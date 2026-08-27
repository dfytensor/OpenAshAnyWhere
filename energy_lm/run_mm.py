"""Train EnergyLM on the MiniMind Chinese corpus (character level).

No backpropagation anywhere in the EnergyLM path.  A backprop baseline of the
same width can be trained for comparison with ``--baseline``.

    python -m energy_lm.run_mm --steps 3000 --baseline
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

from .energy_model import EnergyLMConfig, EnergyRecurrentBlock
from .ep_trainer import EPConfig, EPTrainer, generate as ep_generate
from .deq_trainer import DEQConfig, DEQTrainer
from .baseline import BaselineConfig, BaselineTransformer
from .mm_data import build_tokenizer, StreamingBatcher


DATA_PATH = "F:/OpenASH2605/minimind_data/pretrain_t2t_mini.jsonl"
PROMPTS = ["请问", "秋天的", "给我讲一个", "为什么"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ep", "deq"], default="ep",
                   help="ep=equilibrium propagation (pure local); deq=implicit gradient")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--d_model", type=int, default=192)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--free_steps", type=int, default=16)
    p.add_argument("--clamped_steps", type=int, default=16)
    p.add_argument("--dt", type=float, default=0.45)
    p.add_argument("--res_gain", type=float, default=0.4)
    p.add_argument("--init_scale", type=float, default=0.55)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lr_out", type=float, default=0.15)
    p.add_argument("--lr_emb", type=float, default=0.04)
    p.add_argument("--contractivity", type=float, default=0.6)
    p.add_argument("--use_norm", action="store_true", help="RMSNorm inside the block (can destabilise DEQ contractivity)")
    p.add_argument("--tie_embeddings", action="store_true", help="share input/output embeddings")
    p.add_argument("--anderson", action="store_true", help="Anderson-accelerate relaxation (richer Z*)")
    p.add_argument("--anderson_m", type=int, default=5)
    p.add_argument("--anderson_beta", type=float, default=0.8)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--neumann_k", type=int, default=6)
    p.add_argument("--adjoint", choices=["neumann", "gmres"], default="neumann")
    p.add_argument("--gmres_k", type=int, default=8)
    p.add_argument("--max_chars", type=int, default=4500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--out_dir", type=str, default="energy_lm/runs_mm")
    return p.parse_args()


def count_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, stamp)
    os.makedirs(out_dir, exist_ok=True)

    print("[tokenizer] building char-level tokenizer ...")
    tok = build_tokenizer(DATA_PATH, max_chars=args.max_chars, read_mb=40)
    V = tok.vocab_size

    stream = StreamingBatcher(DATA_PATH, tok, args.seq_len, args.batch, device, seed=args.seed)
    gen = iter(stream)
    gen_b = iter(StreamingBatcher(DATA_PATH, tok, args.seq_len, args.batch, device, seed=args.seed + 1)) \
        if args.baseline else None

    cfg = EnergyLMConfig(
        vocab_size=V, d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
        max_seq_len=args.seq_len, dt=args.dt,
        free_steps=args.free_steps, clamped_steps=args.clamped_steps,
        beta=args.beta, res_gain=args.res_gain, init_scale=args.init_scale,
        use_norm=args.use_norm, tie_embeddings=args.tie_embeddings,
        device=str(device),
    )
    model = EnergyRecurrentBlock(cfg).to(device)
    print(f"[model] EnergyLM params: {count_params(model):,}")

    if args.mode == "deq":
        deqcfg = DEQConfig(
            lr=args.lr, lr_out=args.lr_out, lr_emb=args.lr_emb,
            contractivity=args.contractivity, device=str(device),
            anderson=args.anderson, anderson_m=args.anderson_m,
            anderson_beta=args.anderson_beta, free_steps=args.free_steps,
            total_steps=args.steps, warmup=args.warmup, neumann_k=args.neumann_k,
            adjoint=args.adjoint, gmres_k=args.gmres_k,
        )
        trainer = DEQTrainer(model, deqcfg)
        print("[trainer] DEQ implicit gradient (Neumann-series adjoint, tied emb + RMSNorm + cosine LR)")
    else:
        epcfg = EPConfig(
            lr=args.lr, lr_out=args.lr_out, lr_emb=args.lr_emb, beta=args.beta,
            momentum=0.9, contractivity=args.contractivity, device=str(device),
            anderson=args.anderson, anderson_m=args.anderson_m,
            anderson_beta=args.anderson_beta,
        )
        trainer = EPTrainer(model, epcfg)
        label = "EP + Anderson" if args.anderson else "EP (plain)"
        print(f"[trainer] {label}")

    baseline = None
    opt_b = None
    if args.baseline:
        bcfg = BaselineConfig(
            vocab_size=V, d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
            max_seq_len=args.seq_len, n_layers=1, device=str(device),
        )
        baseline = BaselineTransformer(bcfg).to(device)
        opt_b = torch.optim.Adam(baseline.parameters(), lr=3e-4)
        print(f"[model] baseline params: {count_params(baseline):,}")

    log = {"step": [], "loss": [], "res_free": [], "res_clamped": [],
           "baseline_loss": [], "skips": []}

    t0 = time.time()
    bline_ema = float("nan")
    skip_count = 0
    for step in range(1, args.steps + 1):
        x, y = next(gen)
        info = trainer.update(x, y)
        loss = info["loss"]
        skip_count += info.get("skipped", 0)

        if baseline is not None:
            xb, yb = next(gen_b)
            logits = baseline(xb)
            lb = torch.nn.functional.cross_entropy(logits.reshape(-1, V), yb.reshape(-1))
            opt_b.zero_grad(); lb.backward(); opt_b.step()
            bline_ema = lb.item() if math.isnan(bline_ema) else 0.95 * bline_ema + 0.05 * lb.item()

        if step % 50 == 0 or step == 1:
            log["step"].append(step)
            log["loss"].append(loss)
            log["res_free"].append(info["res_free"])
            log["res_clamped"].append(info.get("res_clamped", 0.0))
            log["skips"].append(skip_count)
            bline = bline_ema if baseline is not None else float("nan")
            log["baseline_loss"].append(bline)
            dt = time.time() - t0
            print(f"step {step:4d} | {args.mode}_loss {loss:6.3f} | res_f {info['res_free']:.1e} "
                  f"| base {bline:6.3f} | skip {skip_count} | {dt:.0f}s")

        if step % 500 == 0 or step == args.steps:
            print("  -- EP sample @", step)
            for pr in PROMPTS[:2]:
                s, _ = ep_generate(model, tok, pr, n_new=40, temperature=0.6, top_k=12)
                print("   ", repr(s))
            if baseline is not None:
                print("  -- baseline sample @", step)
                for pr in PROMPTS[:2]:
                    s = baseline.generate(tok, pr, n_new=40, temperature=0.6)
                    print("   ", repr(s))

    print("\n=== final EP samples ===")
    for pr in PROMPTS:
        s, _ = ep_generate(model, tok, pr, n_new=80, temperature=0.5, top_k=12)
        print(f"{pr!r} -> {s!r}")
    if baseline is not None:
        print("\n=== final baseline samples ===")
        for pr in PROMPTS:
            s = baseline.generate(tok, pr, n_new=80, temperature=0.5)
            print(f"{pr!r} -> {s!r}")

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(out_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=2)
    torch.save({"state_dict": model.state_dict(), "itos": tok.itos},
               os.path.join(out_dir, "energylm_mm.pt"))
    try:
        _plot(log, out_dir, args.baseline)
    except Exception as e:
        print("[plot] failed:", e)
    print(f"[done] artefacts in {out_dir}")


def _plot(log, out_dir, with_baseline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = log["step"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(steps, log["loss"], label="EP train CE")
    if with_baseline and not all(math.isnan(v) for v in log["baseline_loss"]):
        ax[0].plot(steps, log["baseline_loss"], label="backprop CE", alpha=0.8)
    ax[0].set_xlabel("step"); ax[0].set_ylabel("cross-entropy"); ax[0].set_title("MiniMind-zh loss")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(steps, log["res_free"], label="free residual")
    ax[1].plot(steps, log["res_clamped"], label="clamped residual")
    ax[1].set_yscale("log"); ax[1].legend(); ax[1].set_title("Relaxation residual")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "curves.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
