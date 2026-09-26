"""Mini-batch construction: targets plus same-writer reference glyphs."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from fontgen.strokes import BASELINE_Y, CHAR_INDEX, ORIGIN, glyph_to_seq

START = np.array([0, 0, 1, 0, 0], dtype=np.float32)


def flatten(strokes):
    pts = np.concatenate(strokes, axis=0)
    lift = np.zeros(len(pts), dtype=bool)
    ends = np.cumsum([len(s) for s in strokes]) - 1
    lift[ends] = True
    return pts, lift


def to_seq(pts, lift):
    prev = np.vstack([ORIGIN[None], pts[:-1]])
    d = pts - prev
    seq = np.zeros((len(pts) + 1, 5), dtype=np.float32)
    seq[:-1, :2] = d
    seq[:-1, 2] = ~lift
    seq[:-1, 3] = lift
    seq[-1, 4] = 1
    return seq


def random_affine(rng):
    s = rng.uniform(0.88, 1.12)
    asp = rng.uniform(0.92, 1.08)
    shear = rng.uniform(-0.25, 0.25)
    t = np.array([rng.uniform(-0.8, 0.8), rng.uniform(-0.5, 0.5)])
    return s * asp, s / asp, shear, t


def apply_affine(pts, aff):
    sx, sy, shear, t = aff
    out = pts.copy()
    out[:, 0] = ORIGIN[0] + sx * (pts[:, 0] - ORIGIN[0]) - shear * (pts[:, 1] - BASELINE_Y)
    out[:, 1] = BASELINE_Y + sy * (pts[:, 1] - BASELINE_Y)
    return out + t


def pad(seqs):
    T = max(len(s) for s in seqs)
    arr = np.zeros((len(seqs), T, 5), dtype=np.float32)
    lens = np.zeros(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        arr[i, : len(s)] = s
        lens[i] = len(s)
    return arr, lens


class Pool:
    """Samples of one split, indexed by writer."""

    def __init__(self, samples):
        self.samples = samples
        self.flat = [flatten(s.strokes) for s in samples]
        self.chars = np.array([CHAR_INDEX[s.char] for s in samples])
        self.by_writer = defaultdict(list)
        for i, s in enumerate(samples):
            self.by_writer[s.writer].append(i)

    def pick_refs(self, i, k, rng, exclude_same_char=True):
        s = self.samples[i]
        cands = [j for j in self.by_writer[s.writer] if j != i and (not exclude_same_char or self.samples[j].char != s.char)]
        k = min(k, len(cands))
        return list(rng.choice(cands, size=k, replace=False)) if k > 0 else []

    def epoch_groups(self, group_size, rng):
        """Shuffle each writer's samples and cut them into same-writer chunks."""
        chunks = []
        for idxs in self.by_writer.values():
            perm = rng.permutation(idxs)
            chunks += [list(perm[i : i + group_size]) for i in range(0, len(perm), group_size)]
        order = rng.permutation(len(chunks))
        return [chunks[i] for i in order]

    def refs_for_chunk(self, chunk, k, rng):
        chars = {self.samples[i].char for i in chunk}
        w = self.samples[chunk[0]].writer
        cands = [j for j in self.by_writer[w] if self.samples[j].char not in chars]
        k = min(k, len(cands))
        return list(rng.choice(cands, size=k, replace=False)) if k > 0 else []

    def build(self, groups, rng=None, augment=False, device="cpu"):
        """groups: list of (target_indices, reference_indices), one entry per writer group.

        All targets of a group share one reference set and (when augmenting) one random
        affine transform, so the augmentation creates a consistent pseudo-writer.
        """
        tgt_seqs, tgt_group, ref_seqs, ref_chars, ref_group, tgt_idx = [], [], [], [], [], []
        for g, (targets, refs) in enumerate(groups):
            aff = random_affine(rng) if augment else None
            for role, items in (("t", targets), ("r", refs)):
                for j in items:
                    pts, lift = self.flat[j]
                    if aff is not None:
                        pts = apply_affine(pts, aff)
                    seq = to_seq(pts, lift)
                    if role == "t":
                        tgt_seqs.append(seq)
                        tgt_group.append(g)
                        tgt_idx.append(j)
                    else:
                        ref_seqs.append(seq)
                        ref_chars.append(self.chars[j])
                        ref_group.append(g)
        tgt, tlen = pad(tgt_seqs)
        inp = np.concatenate([np.broadcast_to(START, (len(tgt_seqs), 1, 5)), tgt[:, :-1]], axis=1)
        mask = (np.arange(tgt.shape[1])[None] < tlen[:, None]).astype(np.float32)
        batch = {
            "inp": torch.from_numpy(np.ascontiguousarray(inp)).to(device),
            "tgt": torch.from_numpy(tgt).to(device),
            "mask": torch.from_numpy(mask).to(device),
            "char": torch.from_numpy(self.chars[tgt_idx]).to(device),
            "tgt_group": torch.from_numpy(np.array(tgt_group)).to(device),
            "n_groups": len(groups),
            "refs": None,
        }
        if ref_seqs:
            rs, rl = pad(ref_seqs)
            batch["refs"] = {
                "seq": torch.from_numpy(rs).to(device),
                "len": torch.from_numpy(rl),
                "char": torch.from_numpy(np.array(ref_chars)).to(device),
                "group": torch.from_numpy(np.array(ref_group)).to(device),
            }
        return batch


__all__ = ["Pool", "glyph_to_seq", "START"]
