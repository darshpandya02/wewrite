"""Style-conditioned LSTM with a mixture-density output (Graves 2013 / sketch-rnn style).

Decoder input at step t: previous stroke-5 point, character embedding and the
writer style vector z. Output: a mixture of M bivariate Gaussians over the
next pen offset plus a 3-way categorical over the pen state.

The style vector comes from a bidirectional LSTM encoder applied to K reference
glyphs written by the same person (each tagged with its character), averaged
over the references. With use_style=False the model is the unconditioned
baseline (z is a constant zero vector).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from fontgen.strokes import CHARSET

POS_SCALE = 5.0


def with_pos(seq, on, count=False):
    """Append the absolute pen position after each step's offset (cumulative sum) and,
    optionally, the number of pen lifts so far (strokes completed / 3)."""
    extra = []
    if on:
        extra.append(torch.cumsum(seq[..., :2], dim=1) / POS_SCALE)
    if count:
        extra.append(torch.cumsum(seq[..., 3:4], dim=1) / 3.0)
    return torch.cat([seq] + extra, dim=-1) if extra else seq


def n_inputs(cfg):
    return 5 + (2 if cfg.abs_pos else 0) + (1 if cfg.stroke_count else 0)


@dataclass
class ModelConfig:
    n_chars: int = len(CHARSET)
    char_dim: int = 32
    style_dim: int = 64
    enc_hidden: int = 128
    dec_hidden: int = 256
    dec_layers: int = 1
    n_mix: int = 20
    dropout: float = 0.2
    use_style: bool = True
    abs_pos: bool = False  # also feed the absolute pen position (mm from box centre / 5)
    stroke_count: bool = False  # also feed the number of strokes completed so far
    input_dropout: bool = True  # dropout on decoder inputs as well as outputs

    def to_dict(self):
        return asdict(self)


class StyleEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.char_emb = nn.Embedding(cfg.n_chars, cfg.char_dim)
        self.abs_pos = cfg.abs_pos
        self.count = cfg.stroke_count
        self.lstm = nn.LSTM(n_inputs(cfg) + cfg.char_dim, cfg.enc_hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * cfg.enc_hidden, cfg.style_dim)
        self.drop = nn.Dropout(cfg.dropout)

    def encode_refs(self, seqs, lens, chars):
        """seqs (R, T, 5), lens (R,), chars (R,) -> per-reference vectors (R, style_dim)."""
        ce = self.char_emb(chars).unsqueeze(1).expand(-1, seqs.shape[1], -1)
        x = torch.cat([with_pos(seqs, self.abs_pos, self.count), ce], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[0], h[1]], dim=-1)
        return self.proj(self.drop(h))

    def forward(self, seqs, lens, chars, group, n_groups):
        """Average reference vectors per target; group (R,) maps each ref to its target index."""
        v = self.encode_refs(seqs, lens, chars)
        z = torch.zeros(n_groups, v.shape[1], device=v.device, dtype=v.dtype)
        z.index_add_(0, group, v)
        cnt = torch.bincount(group, minlength=n_groups).clamp(min=1).unsqueeze(1).to(v.dtype)
        return torch.tanh(z / cnt)


class HandwritingModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.char_emb = nn.Embedding(cfg.n_chars, cfg.char_dim)
        self.encoder = StyleEncoder(cfg) if cfg.use_style else None
        cond = cfg.char_dim + cfg.style_dim
        self.init = nn.Linear(cond, 2 * cfg.dec_layers * cfg.dec_hidden)
        self.lstm = nn.LSTM(n_inputs(cfg) + cond, cfg.dec_hidden, num_layers=cfg.dec_layers, batch_first=True,
                            dropout=cfg.dropout if cfg.dec_layers > 1 else 0.0)
        self.in_drop = nn.Dropout(cfg.dropout if cfg.input_dropout else 0.0)
        self.out_drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.dec_hidden, 6 * cfg.n_mix + 3)

    def style(self, batch):
        """Style vector per target of a batch built by Pool.build."""
        n = batch["inp"].shape[0]
        if self.encoder is None or batch["refs"] is None:
            return torch.zeros(n, self.cfg.style_dim, device=batch["inp"].device)
        r = batch["refs"]
        zg = self.encoder(r["seq"], r["len"], r["char"], r["group"], batch["n_groups"])
        return zg[batch["tgt_group"]]

    def forward(self, inp, chars, z):
        """inp (B, T, 5) teacher-forced inputs -> raw head outputs (B, T, 6M+3)."""
        B, T, _ = inp.shape
        cond = torch.cat([self.char_emb(chars), z], dim=-1)
        hc = torch.tanh(self.init(cond)).view(B, 2, self.cfg.dec_layers, self.cfg.dec_hidden)
        h0 = hc[:, 0].transpose(0, 1).contiguous()
        c0 = hc[:, 1].transpose(0, 1).contiguous()
        x = torch.cat([with_pos(inp, self.cfg.abs_pos, self.cfg.stroke_count), cond.unsqueeze(1).expand(-1, T, -1)], dim=-1)
        out, _ = self.lstm(self.in_drop(x), (h0, c0))
        return self.head(self.out_drop(out))


def mdn_split(raw, n_mix):
    pi, mx, my, lsx, lsy, rho, pen = torch.split(raw, [n_mix] * 6 + [3], dim=-1)
    lsx = lsx.clamp(-7.0, 4.0)
    lsy = lsy.clamp(-7.0, 4.0)
    return F.log_softmax(pi, -1), mx, my, lsx, lsy, torch.tanh(rho) * 0.999, pen


def nll_terms(raw, target, mask, n_mix):
    """Per-sequence summed offset NLL and pen NLL (nats), and point counts.

    target (B, T, 5); mask (B, T) marks valid steps (all points plus the end token).
    Offset likelihood is only scored where the target is a real point (not the end token),
    as in sketch-rnn.
    """
    log_pi, mx, my, lsx, lsy, rho, pen = mdn_split(raw, n_mix)
    dx = target[..., 0:1]
    dy = target[..., 1:2]
    sx, sy = lsx.exp(), lsy.exp()
    zx = (dx - mx) / sx
    zy = (dy - my) / sy
    one_m_r2 = 1 - rho**2
    log_n = -(zx**2 + zy**2 - 2 * rho * zx * zy) / (2 * one_m_r2) - (
        math.log(2 * math.pi) + lsx + lsy + 0.5 * torch.log(one_m_r2)
    )
    log_mix = torch.logsumexp(log_pi + log_n, dim=-1)
    is_point = mask * (1 - target[..., 4])
    off_nll = -(log_mix * is_point).sum(1)
    pen_nll = F.cross_entropy(pen.reshape(-1, 3), target[..., 2:].argmax(-1).reshape(-1), reduction="none")
    pen_nll = (pen_nll.view(mask.shape) * mask).sum(1)
    return off_nll, pen_nll, mask.sum(1)
