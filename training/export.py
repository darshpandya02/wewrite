"""Export a PyTorch checkpoint to the NumPy format served by the Django app."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from fontgen.infer import MODEL_DIR
from training.train import RUNS


def export(tag: str, name: str, ckpt: str = "best.pt"):
    state = torch.load(RUNS / tag / ckpt, map_location="cpu")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {k: v.numpy().astype(np.float32) for k, v in state["state"].items()}
    np.savez_compressed(MODEL_DIR / f"{name}.npz", **arrays)
    (MODEL_DIR / f"{name}.json").write_text(json.dumps(state["cfg"], indent=1))
    n = sum(a.size for a in arrays.values())
    print(f"exported {tag}/{ckpt} -> {name}: {n} parameters")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--name", default="wewrite")
    ap.add_argument("--ckpt", default="best.pt")
    a = ap.parse_args()
    export(a.tag, a.name, a.ckpt)
