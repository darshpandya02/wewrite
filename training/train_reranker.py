"""Train the best-of-N reranker (MLP on 16x24 rasters) on training-pool writers."""

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fontgen.rerank import MODEL_DIR, raster_small
from fontgen.strokes import CHARSET
from training.batching import apply_affine, flatten, random_affine
from training.data import load


def augmented(samples, rng):
    out = []
    for s in samples:
        pts, lift = flatten(s.strokes)
        pts = apply_affine(pts, random_affine(rng))
        out.append(raster_small(np.split(pts, np.nonzero(lift)[0][:-1] + 1)))
    return np.stack(out)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    samples = load()
    tr = [s for s in samples if s.split == "train"]
    va = [s for s in samples if s.split == "val"]
    idx = {c: i for i, c in enumerate(CHARSET)}
    ytr = torch.tensor([idx[s.char] for s in tr])
    xva = torch.from_numpy(np.stack([raster_small(s.strokes) for s in va]))
    yva = torch.tensor([idx[s.char] for s in va])
    base = np.stack([raster_small(s.strokes) for s in tr])
    mean, std = base.mean(0), base.std(0) + 0.05
    net = nn.Sequential(nn.Linear(base.shape[1], 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(CHARSET)))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    norm = lambda x: (x - torch.from_numpy(mean)) / torch.from_numpy(std)  # noqa: E731
    for ep in range(40):
        net.train()
        x = torch.from_numpy(augmented(tr, rng))
        perm = torch.randperm(len(tr))
        for s in range(0, len(tr), 128):
            b = perm[s : s + 128]
            loss = F.cross_entropy(net(norm(x[b])), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            acc = (net(norm(xva)).argmax(1) == yva).float().mean().item()
        print(f"epoch {ep + 1} val-writer accuracy {acc:.3f}", flush=True)
    w = {"mean": mean, "std": std, "w1": net[0].weight.detach().numpy().T, "b1": net[0].bias.detach().numpy(),
         "w2": net[3].weight.detach().numpy().T, "b2": net[3].bias.detach().numpy()}
    np.savez_compressed(MODEL_DIR / "reranker.npz", **{k: v.astype(np.float32) for k, v in w.items()})
    (MODEL_DIR / "reranker.json").write_text(json.dumps({"classes": CHARSET, "val_writer_accuracy": acc}))
    print("saved reranker, val accuracy", acc)


if __name__ == "__main__":
    main()
