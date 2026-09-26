import numpy as np

from fontgen.strokes import RESAMPLE_STEP, glyph_to_seq, resample_stroke, seq_to_glyph


def test_resample_spacing_is_uniform():
    t = np.linspace(0, 2 * np.pi, 400)
    circle = np.stack([5 + 2 * np.cos(t), 10 + 2 * np.sin(t)], 1)
    r = resample_stroke(circle)
    d = np.linalg.norm(np.diff(r, axis=0), axis=1)
    assert abs(d.mean() - RESAMPLE_STEP) < 0.05
    assert d.std() < 0.02


def test_short_stroke_becomes_a_dot():
    r = resample_stroke(np.array([[3.0, 4.0], [3.05, 4.02], [3.05, 4.02]]))
    assert r.shape == (1, 2)


def test_seq_roundtrip():
    glyph = [np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 3.5]]), np.array([[5.0, 1.0]])]
    seq = glyph_to_seq(glyph)
    assert seq.shape == (5, 5)
    assert seq[-1, 4] == 1 and seq[2, 3] == 1 and seq[3, 3] == 1
    back = seq_to_glyph(seq)
    assert len(back) == 2
    for a, b in zip(glyph, back):
        np.testing.assert_allclose(a, b, atol=1e-5)
