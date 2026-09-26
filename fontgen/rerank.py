"""Best-of-N selection for generated glyphs.

A small MLP (trained by training/train_reranker.py on real glyphs of the
training-pool writers) scores how much each candidate looks like the requested
character; the most recognisable candidate is kept. It is deliberately a
different network (MLP on a 16x24 raster) from the CNN judge used in the
evaluation (48x32 raster), although both are trained on the same writers.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from fontgen.strokes import BOX_W, CHAR_INDEX

MODEL_DIR = Path(__file__).resolve().parent / "model"
RW, RH = 16, 24


def raster_small(strokes) -> np.ndarray:
    scale = 4
    im = Image.new("L", (RW * scale, RH * scale), 0)
    d = ImageDraw.Draw(im)
    k = RW * scale / BOX_W
    for s in strokes:
        p = [(float(x) * k, float(y) * k) for x, y in np.asarray(s).reshape(-1, 2)]
        if len(p) == 1:
            x, y = p[0]
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=255)
        else:
            d.line(p, fill=255, width=5, joint="curve")
    return np.asarray(im.resize((RW, RH), Image.BOX), dtype=np.float32).ravel() / 255.0


class Reranker:
    def __init__(self, w: dict, classes: str):
        self.w = w
        self.classes = classes
        self.index = {c: i for i, c in enumerate(classes)}

    def logp(self, glyphs) -> np.ndarray:
        x = np.stack([raster_small(g) if g else np.zeros(RW * RH, np.float32) for g in glyphs])
        x = (x - self.w["mean"]) / self.w["std"]
        h = np.maximum(x @ self.w["w1"] + self.w["b1"], 0)
        z = h @ self.w["w2"] + self.w["b2"]
        z -= z.max(1, keepdims=True)
        return z - np.log(np.exp(z).sum(1, keepdims=True))

    def pick(self, chars, candidates):
        """candidates: list (per char) of lists of glyphs -> best glyph per char."""
        flat = [g for cands in candidates for g in cands]
        lp = self.logp(flat)
        out, i = [], 0
        for c, cands in zip(chars, candidates):
            scores = lp[i : i + len(cands), self.index[c]]
            out.append(cands[int(np.argmax(scores))])
            i += len(cands)
        return out


@lru_cache(maxsize=1)
def load_reranker() -> Reranker | None:
    path = MODEL_DIR / "reranker.npz"
    if not path.exists():
        return None
    with np.load(path) as f:
        w = {k: f[k].astype(np.float32) for k in f.files}
    classes = json.loads((MODEL_DIR / "reranker.json").read_text())["classes"]
    assert all(c in CHAR_INDEX for c in classes)
    return Reranker(w, classes)
