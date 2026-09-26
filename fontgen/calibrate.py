"""Geometric calibration of generated glyphs to the user's own samples.

The model also writes the characters the user drew; comparing those with the
user's versions gives one global affine correction (horizontal and vertical
scale about the baseline, slant, and offset) that is applied to every
generated glyph. This transfers size, slant and placement exactly, which the
style vector alone captures only weakly (see results/evaluation.json).
"""

from __future__ import annotations

import numpy as np

from fontgen.strokes import BASELINE_Y, ORIGIN, resample_glyph


def _moments(glyph):
    p = np.concatenate(resample_glyph([np.asarray(s, float).reshape(-1, 2) for s in glyph]))
    c = p.mean(0)
    d = p - c
    var_y = (d[:, 1] ** 2).mean() + 1e-6
    slant = (d[:, 0] * d[:, 1]).mean() / var_y  # dx per unit dy
    return c, d.std(0) + 1e-3, slant


def fit(user_glyphs, generated_glyphs):
    """Return (sx, sy, shear, tx, ty) mapping generated -> user, or None."""
    pairs = [(u, g) for u, g in zip(user_glyphs, generated_glyphs) if u and g]
    if not pairs:
        return None
    mu = [_moments(u) for u, _ in pairs]
    mg = [_moments(g) for _, g in pairs]
    sx = float(np.clip(np.median([a[1][0] / b[1][0] for a, b in zip(mu, mg)]), 0.7, 1.4))
    sy = float(np.clip(np.median([a[1][1] / b[1][1] for a, b in zip(mu, mg)]), 0.7, 1.4))
    shear = float(np.clip(np.median([a[2] - b[2] for a, b in zip(mu, mg)]), -0.5, 0.5))
    moved = [apply_one(np.asarray(b[0])[None], (sx, sy, shear, 0.0, 0.0))[0] for b in mg]
    t = np.median([a[0] - m for a, m in zip(mu, moved)], axis=0)
    tx, ty = float(np.clip(t[0], -3, 3)), float(np.clip(t[1], -2, 2))
    return sx, sy, shear, tx, ty


def apply_one(pts, params):
    sx, sy, shear, tx, ty = params
    pts = np.asarray(pts, float)
    y = BASELINE_Y + sy * (pts[:, 1] - BASELINE_Y)
    x = ORIGIN[0] + sx * (pts[:, 0] - ORIGIN[0]) + shear * (y - BASELINE_Y)
    return np.stack([x + tx, y + ty], 1)


def apply(glyphs: dict, params) -> dict:
    if params is None:
        return glyphs
    return {c: [apply_one(s, params) for s in g] for c, g in glyphs.items()}
