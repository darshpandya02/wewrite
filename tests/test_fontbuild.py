import io

import numpy as np
from fontTools.pens.areaPen import AreaPen
from fontTools.ttLib import TTFont

from fontgen.fontbuild import build_font, glyph_name, inspect_font


def ring(cx, cy, r, n=40):
    t = np.linspace(0, 2 * np.pi, n)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1)


GLYPHS = {
    "o": [ring(6.8, 10.1, 2.4)],
    "l": [np.array([[6.0, 5.5], [6.2, 12.7]])],
    "i": [np.array([[6.0, 8.0], [6.0, 12.7]]), np.array([[6.0, 6.3]])],
    "A": [np.array([[3, 12.7], [6.8, 5.9], [10.6, 12.7]]), np.array([[4.6, 10.0], [9.0, 10.0]])],
}


def test_font_parses_and_has_glyphs():
    data = build_font(GLYPHS, family="Test Hand")
    info = inspect_font(data)
    assert info["family"] == "Test Hand"
    assert info["chars"] == "Ailo"
    assert info["empty_glyphs"] == ""
    font = TTFont(io.BytesIO(data))
    assert font["head"].unitsPerEm == 1000
    assert "space" in font.getGlyphOrder()


def test_outline_orientation_and_hole():
    font = TTFont(io.BytesIO(build_font(GLYPHS)))
    gs = font.getGlyphSet()
    o = font["glyf"][glyph_name("o")]
    assert o.numberOfContours == 2  # outer ring and counter
    pen = AreaPen(gs)
    gs["o"].draw(pen)
    # TrueType outer contours are clockwise, which AreaPen reports as negative area.
    assert pen.value < 0


def test_dot_is_its_own_contour():
    font = TTFont(io.BytesIO(build_font(GLYPHS)))
    assert font["glyf"]["i"].numberOfContours == 2
