import json
from pathlib import Path

import numpy as np
import pytest

from fontgen.infer import load_model
from fontgen.strokes import CHARSET

DEMO = json.loads((Path(__file__).resolve().parent.parent / "public" / "demo.json").read_text())


def demo_samples():
    d = next(iter(DEMO.values()))
    chars = list(d)
    return [[np.asarray(s) for s in d[c]] for c in chars], chars


def test_style_vector_shape_and_range():
    m = load_model()
    glyphs, chars = demo_samples()
    z = m.encode_style(glyphs, chars)
    assert z.shape == (m.cfg["style_dim"],)
    assert np.all(np.abs(z) <= 1)


def test_generate_all_characters_deterministic():
    m = load_model()
    glyphs, chars = demo_samples()
    z = m.encode_style(glyphs, chars)
    a = m.generate(list(CHARSET), z, seed=3)
    b = m.generate(list(CHARSET), z, seed=3)
    assert len(a) == len(CHARSET)
    for ga, gb in zip(a, b):
        assert len(ga) == len(gb)
        for sa, sb in zip(ga, gb):
            np.testing.assert_array_equal(sa, sb)


def test_numpy_matches_torch():
    torch = pytest.importorskip("torch")
    from training.model import HandwritingModel, ModelConfig

    m = load_model()
    tm = HandwritingModel(ModelConfig(**m.cfg))
    with np.load(Path(__file__).resolve().parent.parent / "fontgen" / "model" / "wewrite.npz") as f:
        tm.load_state_dict({k: torch.from_numpy(f[k]) for k in f.files})
    tm.eval()
    glyphs, chars = demo_samples()
    z_np = m.encode_style(glyphs, chars)

    from fontgen.strokes import CHAR_INDEX, clip_glyph, glyph_to_seq, resample_glyph

    seqs = [glyph_to_seq(clip_glyph(resample_glyph(g))) for g in glyphs]
    T = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), T, 5)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.from_numpy(s)
    with torch.no_grad():
        zt = tm.encoder(x, torch.tensor([len(s) for s in seqs]), torch.tensor([CHAR_INDEX[c] for c in chars]),
                        torch.zeros(len(seqs), dtype=torch.long), 1)[0].numpy()
    np.testing.assert_allclose(z_np, zt, atol=2e-3)


def test_reranker_prefers_recognisable_candidates():
    from fontgen.rerank import load_reranker
    from fontgen.strokes import BASELINE_Y

    rr = load_reranker()
    assert rr is not None
    t = np.linspace(0, 2 * np.pi, 40)
    o = [np.stack([6.8 + 2.3 * np.cos(t), 10.1 + 2.4 * np.sin(t)], 1)]
    l_ = [np.array([[6.8, 5.8], [6.8, BASELINE_Y]])]
    assert rr.pick(["o"], [[l_, o]])[0] is o
    assert rr.pick(["l"], [[o, l_]])[0] is l_
