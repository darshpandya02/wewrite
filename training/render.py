"""Rasterise glyphs: grids for figures and fixed-size images for the classifiers."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from fontgen.strokes import BASELINE_Y, BOX_H, BOX_W, XHEIGHT_Y


def draw_glyph(draw, strokes, x0, y0, scale, width=2, fill=0):
    for s in strokes:
        p = [(x0 + x * scale, y0 + y * scale) for x, y in np.asarray(s).reshape(-1, 2)]
        if len(p) == 1:
            x, y = p[0]
            r = width / 2
            draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)
        else:
            draw.line(p, fill=fill, width=width, joint="curve")


def raster(strokes, w=32, h=48, width=2):
    """Glyph in the full box frame -> float32 (h, w) image in [0, 1], ink = 1."""
    im = Image.new("L", (w * 2, h * 2), 0)
    d = ImageDraw.Draw(im)
    draw_glyph(d, strokes, 0, 0, (w * 2) / BOX_W, width=width * 2, fill=255)
    im = im.resize((w, h), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def grid(rows, labels=None, scale=6, width=2, cols=None):
    """rows: list of lists of glyphs. Returns a PIL image with guide lines per cell."""
    cols = cols or max(len(r) for r in rows)
    cw, ch = int(BOX_W * scale), int(BOX_H * scale)
    pad_left = 150 if labels else 0
    im = Image.new("RGB", (pad_left + cols * cw, len(rows) * ch), "white")
    d = ImageDraw.Draw(im)
    for r, row in enumerate(rows):
        if labels:
            d.text((6, r * ch + ch // 2 - 6), labels[r], fill=(60, 60, 60))
        for c, g in enumerate(row):
            x0, y0 = pad_left + c * cw, r * ch
            for gy, col in ((XHEIGHT_Y, (225, 225, 240)), (BASELINE_Y, (200, 200, 230))):
                d.line([(x0, y0 + gy * scale), (x0 + cw, y0 + gy * scale)], fill=col)
            d.rectangle([x0, y0, x0 + cw - 1, y0 + ch - 1], outline=(235, 235, 235))
            if g:
                draw_glyph(d, g, x0, y0, scale, width=width, fill=(20, 20, 20))
    return im
