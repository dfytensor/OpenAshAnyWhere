"""Main experiment: train EnergyLM with Equilibrium Propagation and compare
against a backprop baseline of the same size.

Usage (from the repo root)::

    python -m energy_lm.run --steps 1500

The script writes logs and plots to ``energy_lm/runs/<timestamp>/``.
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

from .data import (
    DEFAULT_CORPUS,
    CharTokenizer,
    batch_iter,
    make_sequences,
    save_corpus,
)
from .energy_model import EnergyLMConfig, EnergyRecurrentBlock
from .ep_trainer import EPConfig, EPTrainer, generate
from .baseline import BaselineConfig, BaselineTransformer


# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seq_len", type=int, default=48)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=128)
    p.add_argument("--free_steps", type=int, default=20)
    p.add_argument("--clamped_steps", type=int, default=20)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--res_gain", type=float, default=0.5)
    p.add_argument("--init_scale", type=float, default=0.6)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lr_out", type=float, default=0.05)
    p.add_argument("--lr_emb", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--baseline", action="store_true", help="also train backprop baseline")
    p.add_argument("--out_dir", type=str, default="energy_lm/runs")
    return p.parse_args()


def count_params(m) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def bits_per_char(loss: float) -> float:
    return loss / math.log(2)


# ---------------------------------------------------------------------------
def evaluate_lm(model: EnergyRecurrentBlock, data, seq_len, device, n_batches=8):
    """Average cross-entropy / BPC over random windows using free relaxation."""
    rng = np.random.default_rng(123)
    gen = batch_iter(data, seq_len, 32, rng, device)
    total, count = 0.0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = next(gen)
            X = model.embed(x)
            out = model.relax(X, beta=0.0, steps=max(model.cfg.free_steps, 16))
            logits = model.logits_from_state(out["Z"])
            import torch.nn.functional as Fn
            loss = Fn.cross_entropy(
                logits.reshape(-1, model.cfg.vocab_size), y.reshape(-1)
            ).item()
            total += loss
            count += 1
    return total / max(count, 1)


# ---------------------------------------------------------------------------
def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] using {device}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, stamp)
    os.makedirs(out_dir, exist_ok=True)
    corpus_path = os.path.join(out_dir, "corpus.txt")
    save_corpus(corpus_path, DEFAULT_CORPUS)

    # --- tokenizer / data ------------------------------------------
    tok = CharTokenizer.from_text(DEFAULT_CORPUS)
    V = tok.vocab_size
    data = make_sequences(DEFAULT_CORPUS, tok, args.seq_len + 1, device)
    print(f"[data] vocab={V} chars, corpus={data.numel()} tokens")
    rng = np.random.default_rng(args.seed)

    # --- EnergyLM --------------------------------------------------
    cfg = EnergyLMConfig(
        vocab_size=V,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        dt=args.dt,
        free_steps=args.free_steps,
        clamped_steps=args.clamped_steps,
        beta=args.beta,
        res_gain=args.res_gain,
        init_scale=args.init_scale,
        device=str(device),
    )
    model = EnergyRecurrentBlock(cfg).to(device)
    n_ep = count_params(model)
    print(f"[model] EnergyLM params: {n_ep:,}")

    epcfg = EPConfig(
        lr=args.lr,
        lr_out=args.lr_out,
        lr_emb=args.lr_emb,
        beta=args.beta,
        momentum=args.momentum,
        device=str(device),
    )
    trainer = EPTrainer(model, epcfg)

    # --- optional baseline -----------------------------------------
    baseline = None
    opt_b = None
    if args.baseline:
        bcfg = BaselineConfig(
            vocab_size=V,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            max_seq_len=args.seq_len,
            n_layers=1,
            device=str(device),
        )
        baseline = BaselineTransformer(bcfg).to(device)
        opt_b = torch.optim.Adam(baseline.parameters(), lr=3e-3)
        print(f"[model] baseline params: {count_params(baseline):,}")

    # --- training --------------------------------------------------
    gen = batch_iter(data, args.seq_len, args.batch, rng, device)
    gen_b = batch_iter(data, args.seq_len, args.batch, rng, device) if baseline is not None else None
    log = {"step": [], "loss": [], "bpc": [], "res_free": [], "res_clamped": [],
           "baseline_loss": [], "eval_loss": [], "eval_bpc": [], "skips": []}

    prompt = "the "
    t0 = time.time()
    bline_ema = float("nan")
    skip_count = 0
    for step in range(1, args.steps + 1):
        x, y = next(gen)         # x: (B,T)  y: (B,T) next-token targets
        info = trainer.update(x, y)
        loss = info["loss"]
        skip_count += info.get("skipped", 0)

        # baseline trains every step on its own fresh batch (fair comparison)
        if baseline is not None:
            xb, yb = next(gen_b)
            logits = baseline(xb)
            lb = torch.nn.functional.cross_entropy(
                logits.reshape(-1, V), yb.reshape(-1)
            )
            opt_b.zero_grad()
            lb.backward()
            opt_b.step()
            bline_ema = lb.item() if math.isnan(bline_ema) else 0.9 * bline_ema + 0.1 * lb.item()

        if step % 50 == 0 or step == 1:
            log["step"].append(step)
            log["loss"].append(loss)
            log["bpc"].append(bits_per_char(loss))
            log["res_free"].append(info["res_free"])
            log["res_clamped"].append(info["res_clamped"])
            log["skips"].append(skip_count)

            bline = bline_ema if baseline is not None else float("nan")
            log["baseline_loss"].append(bline)

            dt = time.time() - t0
            print(
                f"step {step:4d} | ep_loss {loss:7.4f} ({bits_per_char(loss):.3f} bpc) | "
                f"res_free {info['res_free']:.2e} res_clmp {info['res_clamped']:.2e} | "
                f"base {bline:7.4f} | skip {skip_count} | {dt:.0f}s"
            )

        # periodic sample + eval
        if step % 250 == 0 or step == args.steps:
            eloss = evaluate_lm(model, data, args.seq_len, device)
            log["eval_loss"].append((step, eloss))
            log["eval_bpc"].append((step, bits_per_char(eloss)))
            sample, gen_info = generate(
                model, tok, prompt, n_new=50, temperature=0.7, top_k=8
            )
            print("    sample:", repr(sample))
            print(f"    eval_loss={eloss:.4f} ({bits_per_char(eloss):.3f} bpc), "
                  f"final_energy={gen_info['energies'][-1]:.4f}")

    # --- final generation -----------------------------------------
    print("\n=== final EnergyLM samples (temp 0.5) ===")
    for pr in ["the ", "a soft ", "the river "]:
        s, _ = generate(model, tok, pr, n_new=80, temperature=0.5, top_k=8)
        print(f"{pr!r} -> {s!r}")

    if baseline is not None:
        print("\n=== final baseline samples (temp 0.5) ===")
        for pr in ["the ", "a soft ", "the river "]:
            s = baseline.generate(tok, pr, n_new=80, temperature=0.5)
            print(f"{pr!r} -> {s!r}")

    # --- save artefacts -------------------------------------------
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(out_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=2)
    torch.save(model.state_dict(), os.path.join(out_dir, "energylm.pt"))
    try:
        _plot(log, out_dir, args.baseline)
    except Exception as e:
        print("[plot] failed:", e)
    print(f"\n[done] artefacts in {out_dir}")


# ---------------------------------------------------------------------------
def _plot(log, out_dir, with_baseline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = log["step"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0, 0]
    ax.plot(steps, log["loss"], label="EP train CE")
    if with_baseline and not all(math.isnan(v) for v in log["baseline_loss"]):
        ax.plot(steps, log["baseline_loss"], label="backprop CE", alpha=0.8)
    ax.set_xlabel("step"); ax.set_ylabel("cross-entropy"); ax.set_title("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, log["bpc"], color="tab:green")
    ax.set_xlabel("step"); ax.set_ylabel("bits/char"); ax.set_title("BPC (EP)")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, log["res_free"], label="free residual")
    ax.plot(steps, log["res_clamped"], label="clamped residual")
    ax.set_xlabel("step"); ax.set_ylabel("residual norm"); ax.set_title("Relaxation convergence")
    ax.set_yscale("log"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if log["eval_loss"]:
        es = [s for s, _ in log["eval_loss"]]
        ev = [v for _, v in log["eval_loss"]]
        ax.plot(es, ev, "-o", color="tab:purple")
        ax.set_xlabel("step"); ax.set_ylabel("eval CE"); ax.set_title("Eval loss")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "curves.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
