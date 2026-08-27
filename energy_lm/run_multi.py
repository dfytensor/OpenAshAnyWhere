"""Train the multi-block hierarchical-equilibrium EnergyLM on MiniMind-zh.

    python -m energy_lm.run_multi --n_blocks 3 --steps 6000 --baseline
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

from .energy_model import EnergyLMConfig
from .multi_model import MultiBlockERB
from .multi_trainer import MultiDEQConfig, MultiDEQTrainer
from .baseline import BaselineConfig, BaselineTransformer
from .mm_data import build_tokenizer, StreamingBatcher

DATA_PATH = "F:/OpenASH2605/minimind_data/pretrain_t2t_mini.jsonl"
PROMPTS = ["请问", "秋天的", "给我讲一个", "为什么"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_blocks", type=int, default=3)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--d_model", type=int, default=192)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--free_steps", type=int, default=22)
    p.add_argument("--gmres_k", type=int, default=8)
    p.add_argument("--anderson_beta", type=float, default=0.7)
    p.add_argument("--dt", type=float, default=0.4)
    p.add_argument("--res_gain", type=float, default=0.5)
    p.add_argument("--init_scale", type=float, default=0.55)
    p.add_argument("--contractivity", type=float, default=0.7)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--lr_out", type=float, default=4e-3)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--max_chars", type=int, default=4500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--out_dir", type=str, default="energy_lm/runs_multi")
    return p.parse_args()


@torch.no_grad()
def generate(model, tok, prompt, n_new=60, temperature=0.6, top_k=12):
    device = model.tok_emb.device
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    max_len = model.cfg.max_seq_len
    for _ in range(n_new):
        ctx = ids[:, -max_len:]
        X = model.embed(ctx)
        out = model.relax_all(X, steps=model.cfg.free_steps if hasattr(model.cfg, "free_steps") else 22,
                              anderson=True, anderson_beta=0.7)
        hL = out["states"][-1]
        logits = (hL @ model.output_weight + model.b_out)[0, -1] / max(temperature, 1e-5)
        if top_k > 0:
            v, _ = torch.topk(logits, top_k); logits[logits < v[-1]] = float("-inf")
        probs = torch.softmax(logits, -1)
        nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt.unsqueeze(0)], 1)
    return tok.decode(ids[0])


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} | blocks={args.n_blocks}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, stamp)
    os.makedirs(out_dir, exist_ok=True)

    tok = build_tokenizer(DATA_PATH, max_chars=args.max_chars, read_mb=40)
    V = tok.vocab_size
    stream = StreamingBatcher(DATA_PATH, tok, args.seq_len, args.batch, device, seed=args.seed)
    gen = iter(stream)
    gen_b = iter(StreamingBatcher(DATA_PATH, tok, args.seq_len, args.batch, device, seed=args.seed + 1)) \
        if args.baseline else None

    cfg = EnergyLMConfig(
        vocab_size=V, d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
        max_seq_len=args.seq_len, dt=args.dt, res_gain=args.res_gain,
        init_scale=args.init_scale, tie_embeddings=True, device=str(device),
    )
    model = MultiBlockERB(cfg, n_blocks=args.n_blocks).to(device)
    print(f"[model] MultiBlockERB params: {count_params(model):,} ({args.n_blocks} blocks)")

    tcfg = MultiDEQConfig(
        lr=args.lr, lr_out=args.lr_out, lr_emb=args.lr,
        contractivity=args.contractivity, free_steps=args.free_steps,
        gmres_k=args.gmres_k, anderson_beta=args.anderson_beta,
        total_steps=args.steps, warmup=args.warmup, device=str(device),
    )
    trainer = MultiDEQTrainer(model, tcfg)

    baseline = None; opt_b = None
    if args.baseline:
        bcfg = BaselineConfig(vocab_size=V, d_model=args.d_model, n_heads=args.n_heads,
                              d_ff=args.d_ff, max_seq_len=args.seq_len,
                              n_layers=args.n_blocks, device=str(device))
        baseline = BaselineTransformer(bcfg).to(device)
        opt_b = torch.optim.Adam(baseline.parameters(), lr=3e-4)
        print(f"[model] baseline ({args.n_blocks} layers) params: {count_params(baseline):,}")

    log = {"step": [], "loss": [], "res_free": [], "baseline_loss": [], "skips": []}
    t0 = time.time(); bline_ema = float("nan"); skip_count = 0
    for step in range(1, args.steps + 1):
        x, y = next(gen)
        info = trainer.update(x, y)
        loss = info["loss"]; skip_count += info.get("skipped", 0)
        if baseline is not None:
            xb, yb = next(gen_b)
            lb = torch.nn.functional.cross_entropy(baseline(xb).reshape(-1, V), yb.reshape(-1))
            opt_b.zero_grad(); lb.backward(); opt_b.step()
            bline_ema = lb.item() if math.isnan(bline_ema) else 0.95 * bline_ema + 0.05 * lb.item()
        if step % 50 == 0 or step == 1:
            log["step"].append(step); log["loss"].append(loss)
            log["res_free"].append(info["res_free"]); log["skips"].append(skip_count)
            bline = bline_ema if baseline is not None else float("nan")
            log["baseline_loss"].append(bline)
            print(f"step {step:4d} | deq_loss {loss:6.3f} | res_f {info['res_free']:.1e} "
                  f"| base {bline:6.3f} | skip {skip_count} | {time.time()-t0:.0f}s")
        if step % 750 == 0 or step == args.steps:
            for pr in PROMPTS[:2]:
                print("   ", repr(generate(model, tok, pr, n_new=40, temperature=0.6, top_k=12)))

    print("\n=== final multi-block DEQ samples ===")
    for pr in PROMPTS:
        print(f"{pr!r} -> {generate(model, tok, pr, n_new=80, temperature=0.5, top_k=12)!r}")
    if baseline is not None:
        print("\n=== final baseline samples ===")
        for pr in PROMPTS:
            print(f"{pr!r} -> {baseline.generate(tok, pr, n_new=80, temperature=0.5)!r}")

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(out_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=2)
    torch.save({"state_dict": model.state_dict(), "itos": tok.itos},
               os.path.join(out_dir, "multi_energylm.pt"))
    print(f"[done] artefacts in {out_dir}")


if __name__ == "__main__":
    main()
