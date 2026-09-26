"""Request validation and the generate / build pipeline used by the views."""

from __future__ import annotations

import numpy as np

from fontgen import calibrate
from fontgen.infer import load_model
from fontgen.rerank import load_reranker
from fontgen.strokes import BOX_H, BOX_W, CHAR_INDEX, CHARSET

MAX_SAMPLES = 40
MAX_POINTS_PER_SAMPLE = 4000
COORD_LIMIT = 60.0  # mm; the box is 13.6 x 20.4


class BadRequest(ValueError):
    pass


def parse_strokes(raw) -> list[np.ndarray]:
    if not isinstance(raw, list) or not raw:
        raise BadRequest("strokes must be a non-empty list of point lists")
    out, total = [], 0
    for s in raw:
        arr = np.asarray(s, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) == 0:
            raise BadRequest("each stroke must be a list of [x, y] points")
        if not np.isfinite(arr).all() or np.abs(arr).max() > COORD_LIMIT:
            raise BadRequest("coordinates must be finite millimetres inside the box")
        total += len(arr)
        out.append(arr)
    if total > MAX_POINTS_PER_SAMPLE:
        raise BadRequest("too many points in one sample")
    return out


def parse_samples(raw):
    if not isinstance(raw, list) or not raw:
        raise BadRequest("send at least one handwriting sample")
    if len(raw) > MAX_SAMPLES:
        raise BadRequest(f"at most {MAX_SAMPLES} samples")
    glyphs, chars = [], []
    for item in raw:
        if not isinstance(item, dict):
            raise BadRequest("each sample is {char, strokes}")
        c = item.get("char")
        if not isinstance(c, str) or c not in CHAR_INDEX:
            raise BadRequest(f"unsupported character: {c!r}")
        glyphs.append(parse_strokes(item.get("strokes")))
        chars.append(c)
    return glyphs, chars


def parse_style(raw) -> np.ndarray:
    model = load_model()
    arr = np.asarray(raw, dtype=np.float32)
    if arr.shape != (model.cfg["style_dim"],) or not np.isfinite(arr).all():
        raise BadRequest("style must be the vector returned by /api/style")
    return arr


def encode(glyphs, chars) -> np.ndarray:
    return load_model().encode_style(glyphs, chars)


def _is_degenerate(ch, strokes):
    n = sum(len(s) for s in strokes)
    if n == 0:
        return True
    pts = np.concatenate(strokes)
    span = np.ptp(pts, axis=0).max()
    small_ok = ch in ".,'-\":;!"
    return (n < 3 or span < 0.8) and not small_ok


def sample(style, chars, temperature=0.15, seed=0, candidates=1, model=None):
    """Draw `candidates` samples per character in one batch and keep the one the
    reranker finds most recognisable (plain sampling when candidates == 1)."""
    model = model or load_model()
    chars = list(chars)
    rr = load_reranker()
    if candidates <= 1 or rr is None:
        return model.generate(chars, style, temperature=temperature, seed=seed)
    flat = model.generate([c for c in chars for _ in range(candidates)], style, temperature=temperature, seed=seed)
    groups = [flat[i * candidates : (i + 1) * candidates] for i in range(len(chars))]
    return rr.pick(chars, groups)


DEFAULT_CANDIDATES = 8
DEFAULT_TEMPERATURE = 0.3


def generate(style, chars=CHARSET, temperature=DEFAULT_TEMPERATURE, seed=0, candidates=DEFAULT_CANDIDATES,
             user=None):
    """user: optional (glyphs, chars) of the user's own samples, used for geometric calibration."""
    model = load_model()
    chars = list(chars)
    if user is not None:
        extra = [c for c in user[1] if c not in chars]
        chars = chars + extra
    glyphs = dict(zip(chars, sample(style, chars, temperature, seed, candidates)))
    for attempt in range(1, 4):  # resample the rare empty or collapsed glyph
        bad = [c for c in chars if _is_degenerate(c, glyphs[c])]
        if not bad:
            break
        for c, g in zip(bad, model.generate(bad, style, temperature=temperature, seed=seed + 1000 * attempt)):
            glyphs[c] = g
    if user is not None:
        params = calibrate.fit(user[0], [glyphs[c] for c in user[1]])
        glyphs = calibrate.apply(glyphs, params)
        glyphs = {c: g for c, g in glyphs.items() if c not in extra}
    # Keep any runaway trajectory inside a margin around the box.
    return {c: [np.clip(s, [-4, -4], [BOX_W + 4, BOX_H + 4]) for s in g] for c, g in glyphs.items()}


def glyphs_to_json(glyphs) -> dict:
    return {c: [np.round(np.asarray(s), 2).tolist() for s in strokes] for c, strokes in glyphs.items()}


def parse_glyphs(raw) -> dict:
    if not isinstance(raw, dict) or not raw:
        raise BadRequest("glyphs must map characters to strokes")
    out = {}
    for c, strokes in raw.items():
        if c not in CHAR_INDEX:
            raise BadRequest(f"unsupported character: {c!r}")
        out[c] = parse_strokes(strokes)
    return out
