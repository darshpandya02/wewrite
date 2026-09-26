"""Independent judges for generated glyphs.

CharCNN: which of the 62 alphanumeric characters is this glyph? Trained on the
40 training-pool writers only, never on generated data.
WriterCNN: which of the 20 held-out writers wrote this glyph (given its
character)? Trained on the held-out writers' session-1 samples only.
Both see the glyph rasterised in the full box frame (position and size kept).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fontgen.strokes import ALNUM
from training.batching import apply_affine, flatten, random_affine
from training.render import raster

H, W = 48, 32
ALNUM_INDEX = {c: i for i, c in enumerate(ALNUM)}


class ConvNet(nn.Module):
    def __init__(self, n_out, n_cond=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.n_cond = n_cond
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(128 * 6 * 4 + n_cond, 256), nn.ReLU(),
                                nn.Dropout(0.3), nn.Linear(256, n_out))

    def forward(self, x, cond=None):
        h = self.conv(x).flatten(1)
        if self.n_cond:
            h = torch.cat([h, cond], 1)
        return self.fc(h)


def rasterize(glyphs, augment=False, rng=None):
    out = np.zeros((len(glyphs), 1, H, W), np.float32)
    for i, g in enumerate(glyphs):
        if augment:
            pts, lift = flatten(g)
            pts = apply_affine(pts, random_affine(rng))
            ends = np.nonzero(lift)[0] + 1
            g = np.split(pts, ends[:-1])
        out[i, 0] = raster(g, W, H)
    return torch.from_numpy(out)


def one_hot(chars):
    m = torch.zeros(len(chars), len(ALNUM))
    m[torch.arange(len(chars)), torch.tensor([ALNUM_INDEX[c] for c in chars])] = 1
    return m


def fit(model, glyphs, labels, cond_chars=None, epochs=30, seed=0, augment=True, log=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    y = torch.tensor(labels)
    n = len(glyphs)
    static = None if augment else rasterize(glyphs)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        x_all = rasterize(glyphs, augment=True, rng=rng) if augment else static
        tot = 0.0
        for s in range(0, n, 128):
            idx = perm[s : s + 128]
            cond = one_hot([cond_chars[i] for i in idx]) if cond_chars is not None else None
            loss = F.cross_entropy(model(x_all[idx], cond), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if log:
            log(f"  epoch {ep + 1}/{epochs} loss {tot / n:.3f}")
    model.eval()
    return model


@torch.no_grad()
def predict_logp(model, glyphs, cond_chars=None):
    model.eval()
    out = []
    for s in range(0, len(glyphs), 256):
        x = rasterize(glyphs[s : s + 256])
        cond = one_hot(cond_chars[s : s + 256]) if cond_chars is not None else None
        out.append(F.log_softmax(model(x, cond), -1))
    return torch.cat(out).numpy()
