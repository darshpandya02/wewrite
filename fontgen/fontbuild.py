"""Assemble a TrueType font from pen strokes.

Each stroke is smoothed (Chaikin corner cutting), thickened into a round-capped
outline with Shapely's buffer, all outlines of a glyph are unioned into clean
non-overlapping polygons, and the polygons are written as TrueType contours
(outer rings clockwise, holes counter-clockwise) with fontTools' FontBuilder.
"""

from __future__ import annotations

import io
import re

import numpy as np
from fontTools.agl import UV2AGL
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from fontgen.strokes import BASELINE_Y, BOX_H, XHEIGHT_Y

UPM = 1000
UNITS_PER_MM = 75.0
SIDE_BEARING = 45
SPACE_WIDTH = 280


def glyph_name(ch: str) -> str:
    return UV2AGL.get(ord(ch), f"uni{ord(ch):04X}")


def chaikin(pts: np.ndarray, iterations: int = 2) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    for _ in range(iterations):
        if len(pts) < 3:
            return pts
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((2 * len(q), 2))
        mid[0::2], mid[1::2] = q, r
        pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts


def to_font_units(pts_mm: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_mm, dtype=np.float64).reshape(-1, 2)
    return np.stack([pts[:, 0] * UNITS_PER_MM, (BASELINE_Y - pts[:, 1]) * UNITS_PER_MM], axis=1)


def outline(strokes_mm, width_mm: float = 0.6, smooth: int = 2):
    """Strokes (mm, box frame) -> shapely (Multi)Polygon in font units."""
    radius = width_mm * UNITS_PER_MM / 2
    shapes = []
    for s in strokes_mm:
        s = np.asarray(s, dtype=np.float64).reshape(-1, 2)
        if len(s) == 0:
            continue
        p = to_font_units(chaikin(s, smooth))
        if len(p) == 1 or np.ptp(p, axis=0).max() < 1e-6:
            shapes.append(Point(p[0]).buffer(radius, quad_segs=6))
        else:
            shapes.append(LineString(p).buffer(radius, quad_segs=6, cap_style="round", join_style="round"))
    if not shapes:
        return None
    geom = unary_union(shapes).simplify(1.0, preserve_topology=True)
    return geom if not geom.is_empty else None


def _polygons(geom):
    if geom is None:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def _draw_ring(pen, coords, dx):
    pts = []
    for x, y in list(coords)[:-1]:  # shapely rings repeat the first point
        q = (int(round(x + dx)), int(round(y)))
        if not pts or q != pts[-1]:
            pts.append(q)
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    if len(pts) < 3:
        return
    pen.moveTo(pts[0])
    for q in pts[1:]:
        pen.lineTo(q)
    pen.closePath()


def build_font(glyphs: dict, family: str = "WeWrite Hand", width_mm: float = 0.6) -> bytes:
    """glyphs: {char: strokes_mm}. Returns TTF bytes."""
    family = re.sub(r"[^A-Za-z0-9 \-]", "", family).strip()[:40] or "WeWrite Hand"
    ps_name = family.replace(" ", "") + "-Regular"
    order = [".notdef", "space"]
    cmap = {32: "space"}
    pen_glyphs, metrics = {}, {}

    nd = TTGlyphPen(None)
    for ring in ([(50, 0), (50, 700), (450, 700), (450, 0)], [(100, 50), (400, 50), (400, 650), (100, 650)]):
        nd.moveTo(ring[0])
        for q in ring[1:]:
            nd.lineTo(q)
        nd.closePath()
    pen_glyphs[".notdef"] = nd.glyph()
    metrics[".notdef"] = (500, 50)
    pen_glyphs["space"] = TTGlyphPen(None).glyph()
    metrics["space"] = (SPACE_WIDTH, 0)

    y_min, y_max = 0, 0
    for ch in sorted(glyphs, key=ord):
        name = glyph_name(ch)
        polys = _polygons(outline(glyphs[ch], width_mm))
        if not polys or name in pen_glyphs:
            continue
        minx = min(p.bounds[0] for p in polys)
        maxx = max(p.bounds[2] for p in polys)
        y_min = min(y_min, min(p.bounds[1] for p in polys))
        y_max = max(y_max, max(p.bounds[3] for p in polys))
        dx = SIDE_BEARING - minx
        pen = TTGlyphPen(None)
        for poly in polys:
            poly = orient(poly, sign=-1.0)  # exterior clockwise, holes counter-clockwise
            _draw_ring(pen, poly.exterior.coords, dx)
            for hole in poly.interiors:
                _draw_ring(pen, hole.coords, dx)
        pen_glyphs[name] = pen.glyph()
        metrics[name] = (int(round(maxx - minx)) + 2 * SIDE_BEARING, SIDE_BEARING)
        order.append(name)
        cmap[ord(ch)] = name

    ascent = int(round(BASELINE_Y * UNITS_PER_MM))
    descent = -int(round((BOX_H - BASELINE_Y) * UNITS_PER_MM))
    fb = FontBuilder(UPM, isTTF=True)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(pen_glyphs)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=ascent, descent=descent)
    fb.setupNameTable({"familyName": family, "styleName": "Regular", "psName": ps_name,
                       "uniqueFontIdentifier": f"WeWrite:{ps_name}", "version": "Version 1.000"})
    fb.setupOS2(sTypoAscender=ascent, sTypoDescender=descent, sTypoLineGap=0,
                usWinAscent=max(ascent, int(y_max)), usWinDescent=max(-descent, int(-y_min)),
                sxHeight=int(round((BASELINE_Y - XHEIGHT_Y) * UNITS_PER_MM)),
                sCapHeight=int(round(6.8 * UNITS_PER_MM)), achVendID="WWRT", fsType=0)
    fb.setupPost()
    buf = io.BytesIO()
    fb.save(buf)
    return buf.getvalue()


def inspect_font(data: bytes) -> dict:
    """Parse a font and report what it contains (used for validation and tests)."""
    font = TTFont(io.BytesIO(data))
    cmap = font.getBestCmap()
    glyf = font["glyf"]
    chars = sorted(chr(u) for u in cmap if u != 32)
    empty = [c for c in chars if glyf[cmap[ord(c)]].numberOfContours == 0]
    return {
        "family": font["name"].getDebugName(1),
        "num_glyphs": len(font.getGlyphOrder()),
        "chars": "".join(chars),
        "empty_glyphs": "".join(empty),
        "units_per_em": font["head"].unitsPerEm,
        "bytes": len(data),
    }
