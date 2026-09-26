"""Render real vs generated glyphs for a few held-out writers."""

from __future__ import annotations

import argparse

from fontgen.infer import load_model
from fontgen.strokes import ALNUM
from training.data import ROOT, load
from training.render import grid

PROMPT = list("adghkstyBM")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="wewrite")
    ap.add_argument("--out", default=str(ROOT / "results" / "samples.png"))
    ap.add_argument("--temp", type=float, default=0.15)
    ap.add_argument("--writers", default="UJI_W12,UPV_W21,UPV_W36")
    ap.add_argument("--chars", default=ALNUM)
    a = ap.parse_args()
    model = load_model(a.name)
    samples = load()
    rows, labels = [], []
    chars = list(a.chars)
    for w in a.writers.split(","):
        s1 = {x.char: x.strokes for x in samples if x.writer == w and x.rep == 1}
        s2 = {x.char: x.strokes for x in samples if x.writer == w and x.rep == 2}
        z = model.encode_style([s1[c] for c in PROMPT], PROMPT)
        gen = model.generate(chars, z, temperature=a.temp, seed=0)
        rows += [[s2[c] for c in chars], gen]
        labels += [f"{w} real", f"{w} generated"]
    grid(rows, labels, scale=4, width=2).save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
