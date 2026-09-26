import io

import pytest
from PIL import Image, ImageDraw

from fontgen.vectorize import VectorizeError, vectorize


def photo(draw_fn, size=(600, 260)):
    im = Image.new("RGB", size, (238, 232, 220))
    draw_fn(ImageDraw.Draw(im))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def test_vectorize_two_letters():
    def draw(d):
        d.line([(100, 40), (100, 200)], fill=(20, 20, 40), width=9)  # "l"
        d.ellipse([230, 110, 320, 200], outline=(20, 20, 40), width=9)  # "o"

    out = vectorize(photo(draw), "lo")
    chars = [s["char"] for s in out["samples"]]
    assert chars == ["l", "o"]
    o = out["samples"][1]["strokes"]
    assert len(o) == 1  # a closed loop traced as one stroke
    xs = [p[0] for p in o[0]]
    ys = [p[1] for p in o[0]]
    assert 0 < min(xs) < max(xs) < 13.6 and 5 < min(ys) < max(ys) < 15


def test_vectorize_label_mismatch():
    def draw(d):
        d.line([(100, 40), (100, 200)], fill=(0, 0, 0), width=9)

    with pytest.raises(VectorizeError):
        vectorize(photo(draw), "abc")
