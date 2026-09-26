"""Turn a photo of handwriting into approximate pen strokes.

Pipeline: grayscale -> Otsu threshold -> connected components -> group
components into characters by horizontal overlap -> Zhang-Suen thinning ->
trace the one-pixel skeleton into polylines -> map pixels into the box frame
(mm) using per-character reference boxes measured on the training data.

A photo has no timing, so stroke order and direction are guesses (top-left
first). The style encoder was trained on real pen trajectories, so photo
input is a weaker style signal than drawing on the canvas.
"""

from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

STATS_FILE = Path(__file__).resolve().parent / "model" / "char_stats.json"
MAX_SIDE = 1400


class VectorizeError(ValueError):
    pass


@lru_cache(maxsize=1)
def char_stats() -> dict:
    return json.loads(STATS_FILE.read_text())


def otsu(gray: np.ndarray) -> float:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
    return float(np.nanargmax(sigma_b))


def binarize(img: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(img).convert("L")
    scale = MAX_SIDE / max(img.size)
    if scale < 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    gray = np.asarray(img, dtype=np.uint8)
    # Flatten uneven lighting with a coarse background estimate.
    bg = np.asarray(img.resize((max(1, img.width // 32), max(1, img.height // 32)), Image.BILINEAR)
                    .resize(img.size, Image.BILINEAR), dtype=np.float64)
    norm = np.clip(gray / np.maximum(bg, 1) * 255, 0, 255).astype(np.uint8)
    t = otsu(norm)
    ink = norm < min(t, 200)
    if ink.mean() > 0.5:  # light ink on dark paper
        ink = ~ink
    return ink


def label_components(mask: np.ndarray):
    """8-connected component labelling (two-pass union-find)."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent = [0]

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        neigh = []
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy and 0 <= xx < w and labels[yy, xx]:
                neigh.append(labels[yy, xx])
        if not neigh:
            parent.append(len(parent))
            labels[y, x] = len(parent) - 1
        else:
            roots = {find(n) for n in neigh}
            r = min(roots)
            labels[y, x] = r
            for o in roots:
                parent[o] = r
    lut = np.array([find(i) for i in range(len(parent))], dtype=np.int32)
    return lut[labels]


def zhang_suen(mask: np.ndarray) -> np.ndarray:
    img = np.pad(mask.astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p = img
            P2, P3, P4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
            P5, P6, P7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            P8, P9 = p[1:-1, :-2], p[:-2, :-2]
            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            B = sum(seq[:8])
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c1, c2 = P2 * P4 * P6, P4 * P6 * P8
            else:
                c1, c2 = P2 * P4 * P8, P2 * P6 * P8
            core = p[1:-1, 1:-1]
            rm = (core == 1) & (B >= 2) & (B <= 6) & (A == 1) & (c1 == 0) & (c2 == 0)
            if rm.any():
                core[rm] = 0
                changed = True
    return img[1:-1, 1:-1].astype(bool)


N4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]
DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _adjacency(pts):
    """8-neighbour graph without redundant diagonal links (a diagonal step is dropped
    when the two pixels are already joined through a shared 4-neighbour)."""
    adj = {}
    for p in pts:
        n = [(p[0] + dy, p[1] + dx) for dy, dx in N4 if (p[0] + dy, p[1] + dx) in pts]
        for dy, dx in DIAG:
            q = (p[0] + dy, p[1] + dx)
            if q in pts and (p[0] + dy, p[1]) not in pts and (p[0], p[1] + dx) not in pts:
                n.append(q)
        adj[p] = n
    return adj


def _edges(adj):
    """Split the skeleton graph into chains between nodes (pixels of degree != 2)."""
    nodes = {p for p, n in adj.items() if len(n) != 2}
    seen, edges = set(), []

    def walk(a, b):
        path = [a, b]
        seen.add(frozenset((a, b)))
        prev, cur = a, b
        while cur not in nodes:
            nxt = [q for q in adj[cur] if q != prev and frozenset((cur, q)) not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            seen.add(frozenset((prev, cur)))
            path.append(cur)
        return path

    for a in sorted(nodes):
        for b in adj[a]:
            if frozenset((a, b)) not in seen:
                edges.append(walk(a, b))
    for p in sorted(adj):  # closed loops with no node at all (e.g. an "o")
        for b in adj[p]:
            if frozenset((p, b)) not in seen:
                nodes.add(p)
                edges.append(walk(p, b))
    return edges


def _direction(path, at_end, k=5):
    seg = path[-k:] if at_end else path[:k][::-1]
    d = np.subtract(seg[-1], seg[0]).astype(float)
    n = np.linalg.norm(d)
    return d / n if n else d


def trace_skeleton(skel: np.ndarray, spur: int = 6):
    """Skeleton pixels -> list of polylines (x, y) in pixel coordinates.

    Chains between junctions are pruned of short spurs and then joined through
    junctions along the straightest continuation, so a crossing stroke stays one stroke.
    """
    pts = set(zip(*[a.tolist() for a in np.nonzero(skel)]))
    if not pts:
        return []
    adj = _adjacency(pts)
    edges = _edges(adj)
    deg = {p: len(n) for p, n in adj.items()}
    if len(edges) > 1:
        edges = [e for e in edges if not (len(e) < spur and (deg[e[0]] == 1) != (deg[e[-1]] == 1))]
    edges = [e for e in edges if len(e) >= 2 or len(edges) == 1]
    incident = {}
    for i, e in enumerate(edges):
        incident.setdefault(e[0], []).append(i)
        incident.setdefault(e[-1], []).append(i)
    used = [False] * len(edges)

    def free_end(i):
        return len(incident[edges[i][0]]) == 1 or len(incident[edges[i][-1]]) == 1

    strokes = []
    while not all(used):
        cands = [i for i in range(len(edges)) if not used[i]]
        cands.sort(key=lambda i: (not free_end(i), min(edges[i][0][0] + edges[i][0][1], edges[i][-1][0] + edges[i][-1][1])))
        i = cands[0]
        e = edges[i]
        a, b = e[0], e[-1]
        if len(incident[b]) == 1 and len(incident[a]) != 1 or (
            (len(incident[a]) == 1) == (len(incident[b]) == 1) and b[0] + b[1] < a[0] + a[1]
        ):
            e = e[::-1]
        used[i] = True
        path = list(e)
        while True:
            end = path[-1]
            din = _direction(path, True)
            best, best_cos = None, 0.0
            for j in incident.get(end, []):
                if used[j]:
                    continue
                cand = edges[j] if edges[j][0] == end else edges[j][::-1]
                cos = float(np.dot(din, _direction(cand, False)))
                if cos > best_cos:
                    best, best_cos, best_path = j, cos, cand
            if best is None:
                break
            used[best] = True
            path += best_path[1:]
        strokes.append(np.array([(x, y) for y, x in path], dtype=np.float64))
    strokes.sort(key=lambda s: s[0, 0] + s[0, 1])
    return strokes


def group_components(labels: np.ndarray, min_area: int):
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    boxes = []
    for i, c in zip(ids.tolist(), counts.tolist()):
        if c < min_area:
            continue
        ys, xs = np.nonzero(labels == i)
        boxes.append([xs.min(), xs.max(), ys.min(), ys.max(), [i]])
    boxes.sort(key=lambda b: b[0])
    groups = []
    for b in boxes:
        if groups:
            g = groups[-1]
            overlap = min(g[1], b[1]) - max(g[0], b[0])
            narrow = min(g[1] - g[0], b[1] - b[0]) + 1
            if overlap > 0.3 * narrow:
                g[0], g[1] = min(g[0], b[0]), max(g[1], b[1])
                g[2], g[3] = min(g[2], b[2]), max(g[3], b[3])
                g[4] += b[4]
                continue
        groups.append(b)
    return groups


def vectorize(image_bytes: bytes, labels: str):
    chars = [c for c in labels if not c.isspace()]
    stats = char_stats()
    unknown = [c for c in chars if c not in stats]
    if unknown:
        raise VectorizeError(f"unsupported characters: {''.join(unknown)}")
    if not chars:
        raise VectorizeError("type the characters shown in the photo, left to right")
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as e:  # noqa: BLE001
        raise VectorizeError("could not read the image") from e
    ink = binarize(img)
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        raise VectorizeError("no ink found")
    # Crop to ink and shrink so the tallest character is ~90 px, which keeps thinning cheap.
    ink = ink[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    target = 90 * 1.0 / max(ink.shape[0], 1)
    if target < 1:
        im = Image.fromarray(ink.astype(np.uint8) * 255).resize(
            (max(1, int(ink.shape[1] * target)), max(1, int(ink.shape[0] * target))), Image.BILINEAR)
        ink = np.asarray(im) > 96
    labels_img = label_components(ink)
    min_area = max(4, int(ink.sum() * 0.002))
    groups = group_components(labels_img, min_area)
    while len(groups) > len(chars):  # merge the pair with the smallest gap
        gaps = [groups[i + 1][0] - groups[i][1] for i in range(len(groups) - 1)]
        i = int(np.argmin(gaps))
        a, b = groups[i], groups.pop(i + 1)
        a[0], a[1], a[2], a[3], a[4] = min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]), a[4] + b[4]
    if len(groups) < len(chars):
        raise VectorizeError(f"found {len(groups)} separate characters but {len(chars)} labels; "
                             "leave clear gaps between characters")

    # Fit one scale and vertical offset (pixels -> mm) from the reference boxes.
    A, rhs = [], []
    for g, c in zip(groups, chars):
        st = stats[c]
        A += [[g[2], 1.0], [g[3], 1.0]]
        rhs += [st["top"], st["bottom"]]
    A, rhs = np.array(A, float), np.array(rhs, float)
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    s, ty = float(sol[0]), float(sol[1])
    if not np.isfinite(s) or s <= 0:
        heights = np.array([g[3] - g[2] + 1 for g in groups], float)
        ref = np.array([stats[c]["bottom"] - stats[c]["top"] for c in chars])
        s = float(np.median(ref / heights))
        ty = float(np.median([stats[c]["bottom"] - s * g[3] for g, c in zip(groups, chars)]))

    samples = []
    for g, c in zip(groups, chars):
        m = np.isin(labels_img, g[4])[g[2]: g[3] + 1, g[0]: g[1] + 1]
        m = np.pad(m, 1)
        paths = trace_skeleton(zhang_suen(m))
        cx_px = (g[0] + g[1]) / 2
        strokes = []
        for p in paths:
            x = (p[:, 0] - 1 + g[0] - cx_px) * s + stats[c]["cx"]
            y = (p[:, 1] - 1 + g[2]) * s + ty
            strokes.append(np.stack([x, y], 1).round(3).tolist())
        if strokes:
            samples.append({"char": c, "strokes": strokes})
    return {"samples": samples, "scale_mm_per_px": s}
