"""Stroke geometry shared by training, inference and the web backend.

Coordinates are millimetres in the UJI acquisition-box frame: origin at the
top-left corner of a 13.6 x 20.4 mm box, x to the right, y downwards. The box
has two writing guides, 7.5 mm (x-height line) and 12.7 mm (baseline) from
the top. The drawing canvas in the web app uses the same frame, so user
samples and training samples live in one coordinate system.

A glyph is a list of strokes; a stroke is an (n, 2) float array.
The model works on the "stroke-5" sequence format of sketch-rnn:
each row is (dx, dy, p_down, p_up, p_end), where the pen state tells what
happens after that point. The first offset is taken from the box centre so
the sequence also encodes where on the box the glyph starts.
"""

from __future__ import annotations

import numpy as np

BOX_W = 13.6
BOX_H = 20.4
XHEIGHT_Y = 7.5
BASELINE_Y = 12.7
ORIGIN = np.array([BOX_W / 2, BOX_H / 2], dtype=np.float64)

ALNUM = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
)
PUNCT = ".,;:?!'\"()%-@$<>"
CHARSET = ALNUM + PUNCT
CHAR_INDEX = {c: i for i, c in enumerate(CHARSET)}

RESAMPLE_STEP = 0.5  # mm between consecutive points after resampling
MAX_POINTS = 160


def _dedupe(stroke: np.ndarray) -> np.ndarray:
    if len(stroke) < 2:
        return stroke
    keep = np.ones(len(stroke), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(stroke, axis=0)) > 1e-9, axis=1)
    return stroke[keep]


def resample_stroke(stroke: np.ndarray, step: float = RESAMPLE_STEP) -> np.ndarray:
    """Resample a polyline to (roughly) equal arc-length spacing."""
    stroke = _dedupe(np.asarray(stroke, dtype=np.float64).reshape(-1, 2))
    if len(stroke) == 0:
        return stroke
    if len(stroke) == 1:
        return stroke.copy()
    seg = np.linalg.norm(np.diff(stroke, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    length = cum[-1]
    if length < 0.3 * step:
        return stroke[:1].copy()
    n = max(1, int(round(length / step)))
    targets = np.linspace(0.0, length, n + 1)
    x = np.interp(targets, cum, stroke[:, 0])
    y = np.interp(targets, cum, stroke[:, 1])
    return np.stack([x, y], axis=1)


def resample_glyph(strokes, step: float = RESAMPLE_STEP):
    out = []
    for s in strokes:
        r = resample_stroke(s, step)
        if len(r):
            out.append(r)
    return out


def glyph_to_seq(strokes) -> np.ndarray:
    """Strokes (absolute mm) -> stroke-5 array including the end token."""
    rows = []
    prev = ORIGIN.copy()
    for s in strokes:
        s = np.asarray(s, dtype=np.float64)
        for i, p in enumerate(s):
            last = i == len(s) - 1
            d = p - prev
            rows.append([d[0], d[1], 0.0 if last else 1.0, 1.0 if last else 0.0, 0.0])
            prev = p
    rows.append([0.0, 0.0, 0.0, 0.0, 1.0])
    return np.asarray(rows, dtype=np.float32)


def seq_to_glyph(seq: np.ndarray):
    """Stroke-5 array -> list of absolute strokes (stops at the end token)."""
    strokes, cur = [], []
    pos = ORIGIN.copy()
    for row in np.asarray(seq, dtype=np.float64):
        if row[4] > 0.5:
            break
        pos = pos + row[:2]
        cur.append(pos.copy())
        if row[3] > 0.5:  # pen lifted after this point
            strokes.append(np.asarray(cur))
            cur = []
    if cur:
        strokes.append(np.asarray(cur))
    return strokes


def glyph_bbox(strokes):
    pts = np.concatenate([np.asarray(s).reshape(-1, 2) for s in strokes], axis=0)
    return pts.min(axis=0), pts.max(axis=0)


def clip_glyph(strokes, max_points: int = MAX_POINTS):
    """Drop points beyond max_points (the end token takes one more slot)."""
    out, total = [], 0
    for s in strokes:
        room = max_points - total
        if room <= 0:
            break
        out.append(s[:room])
        total += len(out[-1])
    return out
