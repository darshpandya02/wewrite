"""Evaluate on the 20 held-out (test) writers.

For every test writer, the 10 prompt characters of session 1 are the style
samples (what a user would draw in the app). The 52 other alphanumerics are
then produced by each method and judged against that writer:

* recognisability: CharCNN top-1 accuracy (62 classes)
* style: WriterCNN accuracy over the 20 test writers, per glyph and per
  writer (log-probabilities summed over the 52 glyphs)
* shape distance: symmetric chamfer distance (mm, box frame) to the writer's
  real session-2 glyph of the same character

Methods: real session-2 glyphs (reference), the style-conditioned model, the
unconditioned model, and a nearest-real-sample baseline that copies the
glyphs of the training writer whose prompt characters are closest in chamfer
distance.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from fontgen import calibrate
from fontgen.infer import load_model
from fontgen.service import DEFAULT_CANDIDATES, DEFAULT_TEMPERATURE, sample
from fontgen.strokes import ALNUM
from training import classifiers as C
from training.batching import Pool
from training.data import ROOT, load
from training.model import HandwritingModel, ModelConfig
from training.render import grid
from training.sample_grid import PROMPT
from training.train import RUNS, eval_groups, evaluate_nll

RESULTS = ROOT / "results"
TARGETS = [c for c in ALNUM if c not in PROMPT]


def log(msg):
    print(msg, flush=True)


def chamfer(a, b):
    pa, pb = np.concatenate(a), np.concatenate(b)
    d = np.linalg.norm(pa[:, None] - pb[None], axis=-1)
    return 0.5 * (d.min(1).mean() + d.min(0).mean())


def train_judges(samples, force=False):
    path = RUNS / "judges.pt"
    test_writers = sorted({s.writer for s in samples if s.split == "test"})
    char_net = C.ConvNet(len(ALNUM))
    writer_net = C.ConvNet(len(test_writers), n_cond=len(ALNUM))
    if path.exists() and not force:
        st = torch.load(path)
        char_net.load_state_dict(st["char"])
        writer_net.load_state_dict(st["writer"])
        return char_net.eval(), writer_net.eval(), test_writers, st["info"]
    t0 = time.time()
    tr = [s for s in samples if s.split != "test" and s.char in C.ALNUM_INDEX]
    log(f"training CharCNN on {len(tr)} glyphs from {len({s.writer for s in tr})} training-pool writers")
    C.fit(char_net, [s.strokes for s in tr], [C.ALNUM_INDEX[s.char] for s in tr], epochs=30, log=log)
    wi = {w: i for i, w in enumerate(test_writers)}
    wtr = [s for s in samples if s.split == "test" and s.rep == 1 and s.char in C.ALNUM_INDEX]
    log(f"training WriterCNN on {len(wtr)} session-1 glyphs of {len(test_writers)} held-out writers")
    C.fit(writer_net, [s.strokes for s in wtr], [wi[s.writer] for s in wtr], cond_chars=[s.char for s in wtr],
          epochs=60, augment=False, log=log)
    info = {"train_seconds": time.time() - t0, "char_train_glyphs": len(tr), "writer_train_glyphs": len(wtr)}
    torch.save({"char": char_net.state_dict(), "writer": writer_net.state_dict(), "info": info}, path)
    return char_net.eval(), writer_net.eval(), test_writers, info


def judge(char_net, writer_net, test_writers, sets):
    """sets: {writer: {char: glyph}} for TARGETS -> metrics dict."""
    glyphs, chars, writers = [], [], []
    for w, d in sets.items():
        for c in TARGETS:
            glyphs.append(d[c] if d[c] else [np.zeros((1, 2))])
            chars.append(c)
            writers.append(w)
    lp_c = C.predict_logp(char_net, glyphs)
    lp_w = C.predict_logp(writer_net, glyphs, cond_chars=chars)
    y_c = np.array([C.ALNUM_INDEX[c] for c in chars])
    wi = {w: i for i, w in enumerate(test_writers)}
    y_w = np.array([wi[w] for w in writers])
    char_acc = float((lp_c.argmax(1) == y_c).mean())
    top5 = float(np.mean([y in np.argsort(-p)[:5] for p, y in zip(lp_c, y_c)]))
    glyph_wacc = float((lp_w.argmax(1) == y_w).mean())
    set_hits = []
    for w in sets:
        m = np.array(writers) == w
        set_hits.append(int(lp_w[m].sum(0).argmax() == wi[w]))
    return {"char_acc": char_acc, "char_top5": top5, "writer_acc_glyph": glyph_wacc,
            "writer_acc_set": float(np.mean(set_hits)), "n_glyphs": len(glyphs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wewrite")
    ap.add_argument("--run", default="final", help="training run whose checkpoint is served")
    ap.add_argument("--uncond", default="uncond")
    ap.add_argument("--uncond-run", default="uncond_nodrop")
    ap.add_argument("--temps", default="0.05,0.15,0.3,0.6,1.0")
    ap.add_argument("--default-temp", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    ap.add_argument("--retrain-judges", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    RESULTS.mkdir(exist_ok=True)
    samples = load()
    char_net, writer_net, test_writers, judge_info = train_judges(samples, a.retrain_judges)

    by = {}
    for s in samples:
        by.setdefault(s.writer, {}).setdefault(s.rep, {})[s.char] = s.strokes
    trainpool = sorted({s.writer for s in samples if s.split != "test"})
    results = {"served_run": a.run, "judges": judge_info, "prompt": "".join(PROMPT), "targets": "".join(TARGETS),
               "n_test_writers": len(test_writers)}

    # Judge sanity check on real data.
    real_test_all = [s for s in samples if s.split == "test" and s.char in C.ALNUM_INDEX]
    lp = C.predict_logp(char_net, [s.strokes for s in real_test_all])
    results["char_acc_real_test_all62"] = float((lp.argmax(1) == np.array([C.ALNUM_INDEX[s.char] for s in real_test_all])).mean())

    methods = {}
    methods["real session-2 glyphs (reference)"] = {w: by[w][2] for w in test_writers}

    # Nearest real sample: the training-pool writer with the closest prompt glyphs.
    nearest = {}
    for w in test_writers:
        d = [np.mean([chamfer(by[w][1][c], by[v][1][c]) for c in PROMPT]) for v in trainpool]
        nearest[w] = trainpool[int(np.argmin(d))]
    methods["nearest real sample (training writer)"] = {w: by[nearest[w]][1] for w in test_writers}
    methods["nearest real sample + calibration"] = {
        w: calibrate.apply(by[nearest[w]][1], calibrate.fit([by[w][1][c] for c in PROMPT],
                                                             [by[nearest[w]][1][c] for c in PROMPT]))
        for w in test_writers}

    model = load_model(a.model)
    uncond = load_model(a.uncond)
    styles = {w: model.encode_style([by[w][1][c] for c in PROMPT], PROMPT) for w in test_writers}

    def gen_sets(m, temp, seed, styled=True, cands=1, calib=False):
        out = {}
        for w in test_writers:
            z = styles[w] if styled else np.zeros(m.cfg["style_dim"], np.float32)
            chars = TARGETS + (PROMPT if calib else [])
            g = dict(zip(chars, sample(z, chars, temp, seed, cands, model=m)))
            if calib:
                params = calibrate.fit([by[w][1][c] for c in PROMPT], [g[c] for c in PROMPT])
                g = calibrate.apply(g, params)
            out[w] = {c: g[c] for c in TARGETS}
        return out

    table = {}
    for name, sets in methods.items():
        r = judge(char_net, writer_net, test_writers, sets)
        r["chamfer_mm"] = float(np.mean([chamfer(sets[w][c], by[w][2][c]) for w in test_writers for c in TARGETS]))
        table[name] = r
        log(f"{name}: {r}")

    def seeded(m, temp, styled, cands=1, calib=False):
        rs = []
        for seed in range(a.seeds):
            sets = gen_sets(m, temp, seed, styled, cands, calib)
            r = judge(char_net, writer_net, test_writers, sets)
            r["chamfer_mm"] = float(np.mean([chamfer(sets[w][c], by[w][2][c]) if sets[w][c] else 99
                                             for w in test_writers for c in TARGETS]))
            rs.append(r)
        return {k: float(np.mean([r[k] for r in rs])) for k in rs[0]} | {
            k + "_std": float(np.std([r[k] for r in rs])) for k in ("char_acc", "writer_acc_glyph", "chamfer_mm")}

    t = a.default_temp
    n = a.candidates
    for label, m, styled, cands, calib in (
        (f"WeWrite: style vector + calibration, best of {n} (served)", model, True, n, True),
        (f"WeWrite: style vector only, best of {n}", model, True, n, False),
        ("WeWrite: style vector only, single sample", model, True, 1, False),
        (f"unconditioned model + calibration, best of {n}", uncond, False, n, True),
        (f"unconditioned model, best of {n}", uncond, False, n, False),
        ("unconditioned model, single sample", uncond, False, 1, False),
    ):
        table[label] = seeded(m, t, styled, cands, calib)
        log(f"{label}: {table[label]}")
    results["main"] = table

    sweep = {}
    for temp in [float(x) for x in a.temps.split(",")]:
        sweep[str(temp)] = seeded(model, temp, True, n, True)
        log(f"T={temp}: {sweep[str(temp)]}")
    results["temperature_sweep"] = sweep

    # NLL on held-out writers as a function of the number of style samples K.
    ck = torch.load(RUNS / a.run / "best.pt", map_location="cpu")
    tm = HandwritingModel(ModelConfig(**ck["cfg"]))
    tm.load_state_dict(ck["state"])
    pool = Pool([s for s in samples if s.split == "test"])
    nll_k = {}
    for k in (1, 3, 5, 10, 20):
        nll_k[str(k)] = evaluate_nll(tm, pool, eval_groups(pool, k, 2), "cpu")[0]
        log(f"test NLL with K={k}: {nll_k[str(k)]:.4f}")
    cku = torch.load(RUNS / a.uncond_run / "best.pt", map_location="cpu")
    um = HandwritingModel(ModelConfig(**cku["cfg"]))
    um.load_state_dict(cku["state"])
    nll_k["unconditioned"] = evaluate_nll(um, pool, eval_groups(pool, 0, 2), "cpu")[0]
    results["test_nll_vs_k"] = nll_k


    # Inference cost of the served NumPy model on this machine (1 process).
    zs = list(styles.values())
    t0 = time.perf_counter()
    for w in test_writers[:5]:
        model.encode_style([by[w][1][c] for c in PROMPT], PROMPT)
    enc_ms = (time.perf_counter() - t0) / 5 * 1000
    from fontgen.strokes import CHARSET

    t0 = time.perf_counter()
    for z in zs[:5]:
        sample(z, CHARSET, t, 0, n, model=model)
    gen_ms = (time.perf_counter() - t0) / 5 * 1000
    results["local_inference_ms"] = {"encode_10_samples": enc_ms, f"generate_78_glyphs_best_of_{n}": gen_ms}

    (RESULTS / "evaluation.json").write_text(json.dumps(results, indent=1))

    # Figure: prompt, real session 2, generated, nearest baseline for three writers.
    rows, labels = [], []
    show = list("AEGHQRWbefmqrz2378")
    for w in test_writers[:4]:
        g = dict(zip(show + PROMPT, sample(styles[w], show + PROMPT, t, 0, n, model=model)))
        g = calibrate.apply(g, calibrate.fit([by[w][1][c] for c in PROMPT], [g[c] for c in PROMPT]))
        gen = {c: g[c] for c in show}
        rows += [[by[w][1][c] for c in PROMPT], [by[w][2][c] for c in show], [gen[c] for c in show],
                 [by[nearest[w]][1][c] for c in show]]
        labels += [f"{w} style samples", f"{w} real", f"{w} WeWrite", f"{w} nearest writer"]
    grid(rows, labels, scale=4).save(RESULTS / "samples_test_writers.png")
    log("saved results/evaluation.json and results/samples_test_writers.png")


if __name__ == "__main__":
    main()
