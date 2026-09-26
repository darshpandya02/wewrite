"""Collect run summaries into results/training_runs.json and print markdown tables."""

import json

from training.data import ROOT
from training.train import RUNS

ABLATION = [
    ("full", "Original recipe: dropout 0.2 on decoder inputs and outputs, LR plateau schedule, early stopping, writer augmentation"),
    ("abl_nodropout", "No dropout"),
    ("hp_outdrop", "Dropout 0.2 on decoder outputs only"),
    ("abl_nosched", "Original recipe without LR scheduling (constant 1e-3)"),
    ("abl_noes", "Original recipe without early stopping (60 epochs, last weights)"),
    ("abl_noaug", "Original recipe without writer augmentation"),
    ("uncond", "Original recipe, unconditioned (no style encoder)"),
]
SWEEP = [
    ("abl_nodropout", "hidden 256, no dropout, style"),
    ("hp_outdrop", "hidden 256, output dropout 0.2, style"),
    ("hp_h384", "hidden 384, no dropout, style (served)"),
    ("uncond_nodrop", "hidden 256, no dropout, unconditioned"),
    ("uncond_h384", "hidden 384, no dropout, unconditioned"),
]


def load(tag):
    p = RUNS / tag / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    out = {}
    print("| Configuration | Val NLL | Test NLL | Best epoch | Epochs run | Train time (s) | Time to best (s) |")
    print("|---|---|---|---|---|---|---|")
    for tag, label in ABLATION:
        s = load(tag)
        if not s:
            continue
        out[tag] = s
        print(f"| {label} | {s['val_nll']:.3f} | {s['test_nll']:.3f} | {s['best_epoch']} | {s['epochs_run']} | "
              f"{s['train_seconds']:.0f} | {s['seconds_to_best']:.0f} |")
    print()
    print("| Configuration | Params | Best val NLL | Test NLL | Epochs | Train time (s) |")
    print("|---|---|---|---|---|---|")
    for tag, label in SWEEP:
        s = load(tag)
        if not s:
            continue
        out[tag] = s
        print(f"| {label} | {s['params']:,} | {s['best_val_nll']:.3f} | {s['test_nll']:.3f} | {s['epochs_run']} | "
              f"{s['train_seconds']:.0f} |")
    (ROOT / "results" / "training_runs.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()


def eval_tables():
    d = json.loads((ROOT / "results" / "evaluation.json").read_text())
    print("| Method | Char accuracy (62-way) | Writer ID, per glyph (20-way) | Writer ID, per writer | Chamfer to real (mm) |")
    print("|---|---|---|---|---|")
    for name, r in d["main"].items():
        std = lambda k: f" ± {r[k + '_std'] * 100:.1f}" if k + "_std" in r else ""  # noqa: E731
        stdc = f" ± {r['chamfer_mm_std']:.3f}" if "chamfer_mm_std" in r else ""
        print(f"| {name} | {r['char_acc'] * 100:.1f}%{std('char_acc')} | {r['writer_acc_glyph'] * 100:.1f}%"
              f"{std('writer_acc_glyph')} | {r['writer_acc_set'] * 100:.0f}% | {r['chamfer_mm']:.3f}{stdc} |")
    print()
    print("| Temperature | Char accuracy | Writer ID per glyph | Writer ID per writer | Chamfer (mm) |")
    print("|---|---|---|---|---|")
    for t, r in d["temperature_sweep"].items():
        print(f"| {t} | {r['char_acc'] * 100:.1f}% | {r['writer_acc_glyph'] * 100:.1f}% | {r['writer_acc_set'] * 100:.0f}% | "
              f"{r['chamfer_mm']:.3f} |")
    print()
    print("| Style samples K | " + " | ".join(d["test_nll_vs_k"]) + " |")
    print("|---|" + "---|" * len(d["test_nll_vs_k"]))
    print("| Test NLL (nats/point) | " + " | ".join(f"{v:.3f}" for v in d["test_nll_vs_k"].values()) + " |")


if __name__ == "__main__":
    eval_tables()
