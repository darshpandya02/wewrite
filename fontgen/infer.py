"""NumPy inference for the handwriting model (no PyTorch needed at serve time).

Weights are exported from the PyTorch checkpoint by training/export.py into
fontgen/model/weights.npz plus config.json. tests/test_infer.py checks that
this implementation matches the PyTorch model numerically.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from fontgen.strokes import CHAR_INDEX, MAX_POINTS, glyph_to_seq, resample_glyph, clip_glyph, seq_to_glyph

POS_SCALE = 5.0
MODEL_DIR = Path(__file__).resolve().parent / "model"


def _sigmoid(x):
    return 0.5 * (np.tanh(0.5 * x) + 1.0)


class LSTMLayer:
    def __init__(self, w_ih, w_hh, b_ih, b_hh):
        self.w = np.concatenate([w_ih, w_hh], axis=1).T.astype(np.float32)  # (In+H, 4H)
        self.b = (b_ih + b_hh).astype(np.float32)
        self.h = w_hh.shape[1]

    def step(self, x, h, c):
        g = np.concatenate([x, h], axis=1) @ self.w + self.b
        H = self.h
        i = _sigmoid(g[:, :H])
        f = _sigmoid(g[:, H : 2 * H])
        gg = np.tanh(g[:, 2 * H : 3 * H])
        o = _sigmoid(g[:, 3 * H :])
        c = f * c + i * gg
        h = o * np.tanh(c)
        return h, c

    def run_final(self, x, lens):
        """x (B, T, In) padded; returns the hidden state at each sequence's last valid step."""
        B, T, _ = x.shape
        h = np.zeros((B, self.h), np.float32)
        c = np.zeros((B, self.h), np.float32)
        for t in range(T):
            h2, c2 = self.step(x[:, t], h, c)
            m = (t < lens)[:, None]
            h = np.where(m, h2, h)
            c = np.where(m, c2, c)
        return h


class Model:
    def __init__(self, weights: dict, cfg: dict):
        self.cfg = cfg
        W = weights
        self.char_emb = W["char_emb.weight"]
        self.n_mix = cfg["n_mix"]
        self.layers = cfg["dec_layers"]
        self.hidden = cfg["dec_hidden"]
        self.dec = [
            LSTMLayer(W[f"lstm.weight_ih_l{k}"], W[f"lstm.weight_hh_l{k}"], W[f"lstm.bias_ih_l{k}"], W[f"lstm.bias_hh_l{k}"])
            for k in range(self.layers)
        ]
        self.init_w, self.init_b = W["init.weight"].T, W["init.bias"]
        self.head_w, self.head_b = W["head.weight"].T, W["head.bias"]
        self.use_style = cfg["use_style"]
        self.abs_pos = cfg.get("abs_pos", False)
        self.count = cfg.get("stroke_count", False)
        self.n_in = 5 + (2 if self.abs_pos else 0) + (1 if self.count else 0)
        if self.use_style:
            self.enc_char = W["encoder.char_emb.weight"]
            self.enc_f = LSTMLayer(W["encoder.lstm.weight_ih_l0"], W["encoder.lstm.weight_hh_l0"],
                                   W["encoder.lstm.bias_ih_l0"], W["encoder.lstm.bias_hh_l0"])
            self.enc_b = LSTMLayer(W["encoder.lstm.weight_ih_l0_reverse"], W["encoder.lstm.weight_hh_l0_reverse"],
                                   W["encoder.lstm.bias_ih_l0_reverse"], W["encoder.lstm.bias_hh_l0_reverse"])
            self.proj_w, self.proj_b = W["encoder.proj.weight"].T, W["encoder.proj.bias"]

    def _aux(self, s):
        extra = [s]
        if self.abs_pos:
            extra.append(np.cumsum(s[:, :2], axis=0) / POS_SCALE)
        if self.count:
            extra.append(np.cumsum(s[:, 3:4], axis=0) / 3.0)
        return np.concatenate(extra, axis=1)

    # ---------------------------------------------------------------- style
    def encode_style(self, glyphs, chars) -> np.ndarray:
        """glyphs: list of stroke lists (mm, box frame); chars: list of characters."""
        if not self.use_style or not glyphs:
            return np.zeros(self.cfg["style_dim"], np.float32)
        seqs = [glyph_to_seq(clip_glyph(resample_glyph(g))) for g in glyphs]
        T = max(len(s) for s in seqs)
        R = len(seqs)
        lens = np.array([len(s) for s in seqs])
        x = np.zeros((R, T, self.n_in), np.float32)
        xr = np.zeros((R, T, self.n_in), np.float32)
        for i, s in enumerate(seqs):
            s = self._aux(s)
            x[i, : len(s)] = s
            xr[i, : len(s)] = s[::-1]
        ce = self.enc_char[[CHAR_INDEX[c] for c in chars]][:, None, :].repeat(T, axis=1)
        hf = self.enc_f.run_final(np.concatenate([x, ce], -1), lens)
        hb = self.enc_b.run_final(np.concatenate([xr, ce], -1), lens)
        v = np.concatenate([hf, hb], axis=1) @ self.proj_w + self.proj_b
        return np.tanh(v.mean(axis=0)).astype(np.float32)

    # ------------------------------------------------------------- decoding
    def _init_state(self, cond):
        hc = np.tanh(cond @ self.init_w + self.init_b).reshape(len(cond), 2, self.layers, self.hidden)
        return [hc[:, 0, k] for k in range(self.layers)], [hc[:, 1, k] for k in range(self.layers)]

    def _head(self, h):
        raw = h @ self.head_w + self.head_b
        M = self.n_mix
        pi, mx, my, lsx, lsy, rho = (raw[:, k * M : (k + 1) * M] for k in range(6))
        return pi, mx, my, np.clip(lsx, -7, 4), np.clip(lsy, -7, 4), np.tanh(rho) * 0.999, raw[:, 6 * M :]

    def generate(self, chars, style, temperature=0.15, seed=0, max_len=MAX_POINTS + 1, pen_temperature=1.0):
        """Sample one glyph per character, all characters decoded as one batch.

        temperature sharpens the offset mixture only; the pen state is sampled at
        pen_temperature (1.0 by default), because a sharpened pen distribution
        almost never picks "lift" or "end" and the pen runs on.
        """
        rng = np.random.default_rng(seed)
        B = len(chars)
        z = np.broadcast_to(np.asarray(style, np.float32), (B, self.cfg["style_dim"]))
        cond = np.concatenate([self.char_emb[[CHAR_INDEX[c] for c in chars]], z], axis=1).astype(np.float32)
        hs, cs = self._init_state(cond)
        prev = np.tile(np.array([0, 0, 1, 0, 0], np.float32), (B, 1))
        pos = np.zeros((B, 2), np.float32)
        lifts = np.zeros((B, 1), np.float32)
        out = np.zeros((B, max_len, 5), np.float32)
        out[:, :, 4] = 1  # rows never written stay end tokens
        act = np.arange(B)  # sequences still being written; finished ones are dropped from the batch
        tau = max(float(temperature), 1e-3)
        ptau = max(float(pen_temperature), 1e-3)
        for t in range(max_len):
            n = len(act)
            pos = pos + prev[:, :2]
            lifts = lifts + prev[:, 3:4]
            feats = [prev]
            if self.abs_pos:
                feats.append(pos / POS_SCALE)
            if self.count:
                feats.append(lifts / 3.0)
            x = np.concatenate(feats + [cond], axis=1)
            for k, layer in enumerate(self.dec):
                hs[k], cs[k] = layer.step(x, hs[k], cs[k])
                x = hs[k]
            pi, mx, my, lsx, lsy, rho, pen = self._head(x)
            logits = pi / tau
            logits -= logits.max(1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(1, keepdims=True)
            comp = (p.cumsum(1) > rng.random((n, 1))).argmax(1)
            r = np.arange(n)
            sx = np.exp(lsx[r, comp]) * np.sqrt(tau)
            sy = np.exp(lsy[r, comp]) * np.sqrt(tau)
            rh = rho[r, comp]
            n1, n2 = rng.standard_normal(n), rng.standard_normal(n)
            dx = mx[r, comp] + sx * n1
            dy = my[r, comp] + sy * (rh * n1 + np.sqrt(1 - rh**2) * n2)
            pl = pen / ptau
            pl -= pl.max(1, keepdims=True)
            pp = np.exp(pl)
            pp /= pp.sum(1, keepdims=True)
            state = (pp.cumsum(1) > rng.random((n, 1))).argmax(1)
            if t == max_len - 1:
                state[:] = 2
            nxt = np.zeros((n, 5), np.float32)
            nxt[:, 0], nxt[:, 1] = dx, dy
            nxt[r, 2 + state] = 1
            out[act, t] = nxt
            keep = state != 2
            if not keep.any():
                break
            act, prev, pos, lifts, cond = act[keep], nxt[keep], pos[keep], lifts[keep], cond[keep]
            hs = [h[keep] for h in hs]
            cs = [c[keep] for c in cs]
        return [seq_to_glyph(out[i]) for i in range(B)]


@lru_cache(maxsize=2)
def load_model(name: str = "wewrite") -> Model:
    cfg = json.loads((MODEL_DIR / f"{name}.json").read_text())
    with np.load(MODEL_DIR / f"{name}.npz") as f:
        weights = {k: f[k].astype(np.float32) for k in f.files}
    return Model(weights, cfg)
