"""Train the style-conditioned handwriting model.

Regularisation / schedule switches (all on by default) exist so each one can be
ablated: dropout, ReduceLROnPlateau learning-rate scheduling, early stopping on
held-out-writer validation NLL, and same-affine writer augmentation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from training.batching import Pool
from training.data import ROOT, load
from training.model import HandwritingModel, ModelConfig, nll_terms

RUNS = ROOT / "runs"
EVAL_K = 5


def eval_groups(pool: Pool, k: int, seed: int, chunk: int = 64):
    """Deterministic evaluation groups: every sample of a writer is scored once, conditioned
    on k reference glyphs of the same writer whose characters differ from the target's."""
    rng = np.random.default_rng(seed)
    groups = []
    for w in sorted(pool.by_writer):
        idxs = pool.by_writer[w]
        perm = list(rng.permutation(idxs))
        r1, r2, used = [], [], set()
        for j in perm:
            c = pool.samples[j].char
            if c in used:
                continue
            if len(r1) < k:
                r1.append(j)
                used.add(c)
            elif len(r2) < k:
                r2.append(j)
                used.add(c)
        c1 = {pool.samples[j].char for j in r1}
        t1 = [i for i in idxs if pool.samples[i].char not in c1]
        t2 = [i for i in idxs if pool.samples[i].char in c1]
        for targets, refs in ((t1, r1), (t2, r2)):
            for s in range(0, len(targets), chunk):
                groups.append((targets[s : s + chunk], refs if k > 0 else []))
    return groups


@torch.no_grad()
def evaluate_nll(model, pool: Pool, groups, device, per_batch=4):
    model.eval()
    tot_off = tot_pen = tot_n = 0.0
    for s in range(0, len(groups), per_batch):
        b = pool.build(groups[s : s + per_batch], augment=False, device=device)
        z = model.style(b)
        raw = model(b["inp"], b["char"], z)
        off, pen, n = nll_terms(raw, b["tgt"], b["mask"], model.cfg.n_mix)
        tot_off += off.sum().item()
        tot_pen += pen.sum().item()
        tot_n += n.sum().item()
    return (tot_off + tot_pen) / tot_n, tot_off / tot_n, tot_pen / tot_n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--style-dim", type=int, default=64)
    ap.add_argument("--mix", type=int, default=20)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--groups", type=int, default=8, help="writer groups per batch")
    ap.add_argument("--group-size", type=int, default=8, help="targets per writer group")
    ap.add_argument("--max-epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--no-sched", action="store_true")
    ap.add_argument("--no-es", action="store_true")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--no-style", action="store_true")
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--style-noise", type=float, default=0.0)
    ap.add_argument("--enc-hidden", type=int, default=128)
    ap.add_argument("--no-abs-pos", action="store_true")
    ap.add_argument("--no-stroke-count", action="store_true")
    ap.add_argument("--output-dropout-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    samples = load()
    pools = {k: Pool([s for s in samples if s.split == k]) for k in ("train", "val", "test")}
    val_refs = eval_groups(pools["val"], EVAL_K, 1)
    test_refs = eval_groups(pools["test"], EVAL_K, 2)

    cfg = ModelConfig(dec_hidden=args.hidden, dec_layers=args.layers, style_dim=args.style_dim,
                      n_mix=args.mix, dropout=args.dropout, use_style=not args.no_style,
                      enc_hidden=args.enc_hidden, abs_pos=not args.no_abs_pos,
                      stroke_count=not args.no_stroke_count, input_dropout=not args.output_dropout_only)
    model = HandwritingModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = None if args.no_sched else torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)

    out = RUNS / args.tag
    out.mkdir(parents=True, exist_ok=True)
    history, best, best_epoch, bad = [], float("inf"), -1, 0
    t0 = time.time()
    time_to_best = 0.0
    train_pool = pools["train"]
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        chunks = train_pool.epoch_groups(args.group_size, rng)
        run_loss = run_n = 0.0
        for s in range(0, len(chunks), args.groups):
            groups = []
            for ch in chunks[s : s + args.groups]:
                k = int(rng.integers(1, args.kmax + 1)) if cfg.use_style else 0
                groups.append((ch, train_pool.refs_for_chunk(ch, k, rng)))
            b = train_pool.build(groups, rng, augment=not args.no_aug, device=device)
            z = model.style(b)
            if args.style_noise > 0:
                z = z + args.style_noise * torch.randn_like(z)
            raw = model(b["inp"], b["char"], z)
            off, pen, cnt = nll_terms(raw, b["tgt"], b["mask"], cfg.n_mix)
            loss = (off.sum() + pen.sum()) / cnt.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item() * cnt.sum().item()
            run_n += cnt.sum().item()
        val, val_off, val_pen = evaluate_nll(model, pools["val"], val_refs, device)
        lr = opt.param_groups[0]["lr"]
        if sched is not None:
            sched.step(val)
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_nll": run_loss / run_n, "val_nll": val, "val_off": val_off,
                        "val_pen": val_pen, "lr": lr, "time": elapsed})
        improved = val < best - 1e-4
        if improved:
            best, best_epoch, bad, time_to_best = val, epoch, 0, elapsed
            torch.save({"cfg": cfg.to_dict(), "state": model.state_dict()}, out / "best.pt")
        else:
            bad += 1
        print(f"[{args.tag}] ep {epoch:3d} train {run_loss / run_n:.4f} val {val:.4f} lr {lr:.2e} "
              f"{'*' if improved else ''} {elapsed:.0f}s", flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=1))
        if not args.no_es and bad >= args.patience:
            break
    total = time.time() - t0
    torch.save({"cfg": cfg.to_dict(), "state": model.state_dict()}, out / "last.pt")

    # Early stopping keeps the best checkpoint; without it the final weights are used.
    final_ckpt = "last.pt" if args.no_es else "best.pt"
    state = torch.load(out / final_ckpt, map_location=device)
    model.load_state_dict(state["state"])
    val_final = evaluate_nll(model, pools["val"], val_refs, device)
    test_final = evaluate_nll(model, pools["test"], test_refs, device)
    summary = {
        "tag": args.tag, "args": vars(args), "params": n_params, "epochs_run": len(history),
        "best_epoch": best_epoch, "best_val_nll": best, "train_seconds": total,
        "seconds_to_best": time_to_best, "used_checkpoint": final_ckpt,
        "val_nll": val_final[0], "test_nll": test_final[0], "test_offset_nll": test_final[1],
        "test_pen_nll": test_final[2], "final_train_nll": history[-1]["train_nll"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
