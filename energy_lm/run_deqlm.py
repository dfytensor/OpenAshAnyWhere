"""Head-to-head: standard DEQ-LM (no contraction, IFT backprop) on MiniMind-zh.

    python -m energy_lm.run_deqlm --steps 6000 --baseline

Validates the §2.0 claim: a proper DEQ-LM (engineered cell + full backprop,
no strict contraction) should land far below the strict-contraction EnergyLM/DEQ
CE of 0.80 on the *same* data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime

import numpy as np
import torch

from .deq_lm import DEQLMConfig, DEQLM
from .baseline import BaselineConfig, BaselineTransformer
from .mm_data import build_tokenizer, StreamingBatcher

DATA_PATH = "F:/OpenASH2605/minimind_data/pretrain_t2t_mini.jsonl"
PROMPTS = ["请问", "秋天的", "给我讲一个", "为什么"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--d_model", type=int, default=192)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--res_gain", type=float, default=0.5)
    p.add_argument("--init_std", type=float, default=0.02)
    p.add_argument("--fwd_iters", type=int, default=30)
    p.add_argument("--bwd_iters", type=int, default=30)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--anderson_beta", type=float, default=0.7)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--max_chars", type=int, default=4500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--out_dir", type=str, default="energy_lm/runs_deqlm")
    return p.parse_args()


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} | standard DEQ-LM (no contraction, IFT backprop)")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, stamp)
    os.makedirs(out_dir, exist_ok=True)

    tok = build_tokenizer(DATA_PATH, max_chars=args.max_chars, read_mb=40)
    V = tok.vocab_size
    gen = iter(StreamingBatcher(DATA_PATH, tok, args.seq_len, args.batch, device, seed=args.seed))
    gen_b = iter(StreamingBatcher(DATA_PATH, tok, args.seq_len, args.batch, device, seed=args.seed + 1)) \
        if args.baseline else None

    cfg = DEQLMConfig(
        vocab_size=V, d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
        max_seq_len=args.seq_len, res_gain=args.res_gain, init_std=args.init_std,
        fwd_iters=args.fwd_iters, bwd_iters=args.bwd_iters, tol=args.tol,
        anderson_beta=args.anderson_beta, device=str(device),
    )
    model = DEQLM(cfg).to(device)
    print(f"[model] DEQLM params: {count_params(model):,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # cosine with warmup
    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(max(prog, 0), 1))))

    baseline = None; opt_b = None
    if args.baseline:
        bcfg = BaselineConfig(vocab_size=V, d_model=args.d_model, n_heads=args.n_heads,
                              d_ff=args.d_ff, max_seq_len=args.seq_len, n_layers=2, device=str(device))
        baseline = BaselineTransformer(bcfg).to(device)
        opt_b = torch.optim.Adam(baseline.parameters(), lr=3e-4)
        print(f"[model] baseline params: {count_params(baseline):,}")

    log = {"step": [], "loss": [], "fwd_res": [], "bwd_res": [], "baseline_loss": []}
    t0 = time.time(); bline_ema = float("nan")
    for step in range(1, args.steps + 1):
        x, y = next(gen)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); 
        for g in opt.param_groups: g["lr"] = lr_at(step)
        opt.step()

        if baseline is not None:
            xb, yb = next(gen_b)
            lb = torch.nn.functional.cross_entropy(baseline(xb).reshape(-1, V), yb.reshape(-1))
            opt_b.zero_grad(); lb.backward(); opt_b.step()
            bline_ema = lb.item() if math.isnan(bline_ema) else 0.95 * bline_ema + 0.05 * lb.item()

        if step % 50 == 0 or step == 1:
            log["step"].append(step); log["loss"].append(loss.item())
            log["fwd_res"].append(model.deq.fwd_res); log["bwd_res"].append(model.deq.bwd_res)
            bline = bline_ema if baseline is not None else float("nan")
            log["baseline_loss"].append(bline)
            print(f"step {step:4d} | deqlm_loss {loss.item():6.3f} | fwd_res {model.deq.fwd_res:.1e} "
                  f"bwd_res {model.deq.bwd_res:.1e} | base {bline:6.3f} | {time.time()-t0:.0f}s")
        if step % 750 == 0 or step == args.steps:
            for pr in PROMPTS[:2]:
                print("   ", repr(model.generate(tok, pr, n_new=40, temperature=0.6, top_k=12)))

    print("\n=== final DEQ-LM samples ===")
    for pr in PROMPTS:
        print(f"{pr!r} -> {model.generate(tok, pr, n_new=80, temperature=0.5, top_k=12)!r}")
    if baseline is not None:
        print("\n=== final baseline samples ===")
        for pr in PROMPTS:
            print(f"{pr!r} -> {baseline.generate(tok, pr, n_new=80, temperature=0.5)!r}")

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(out_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=2)
    torch.save({"state_dict": model.state_dict(), "itos": tok.itos},
               os.path.join(out_dir, "deqlm.pt"))
    print(f"[done] artefacts in {out_dir}")


if __name__ == "__main__":
    main()
