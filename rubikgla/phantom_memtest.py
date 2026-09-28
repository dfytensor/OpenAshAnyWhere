#!/usr/bin/env python3
"""Phantom Gradient 显存/质量测试.
对比: (A) 普通反传穿透30次迭代 (B) 1步 phantom gradient.
测: 峰值显存, 前向一致性, 1000步训练后的 val NLL."""
import sys, os, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v3 import DEQLM, get_batch, V, DEV, SEQ, KMAX, PT_CACHE

HERE = os.path.dirname(os.path.abspath(__file__))
PT_CACHE = r"F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt"
OUT = os.path.join(HERE, "phantom_memtest.json")


def phantom_deq(m, h, n_iter=KMAX):
    """前向 no_grad 解算到 z*, 反向只穿 1 步 (z0 -> z1) 的图."""
    alpha = m.deq.alpha()
    with torch.no_grad():
        z = h
        for _ in range(n_iter):
            z = z + alpha * m.deq._block_fn(z)
        z_star = z
    # phantom: 值 = z*, 梯度流经一步图
    z1 = h + alpha * m.deq._block_fn(h)
    return z_star + (z1 - z1.detach())


def forward_variant(m, x, mode):
    """mode: 'full' 普通反传 | 'phantom' 1步幻影"""
    h = m.bottom_h(x)
    if mode == "full":
        z, res, k = m.deq(h)
    else:
        z = phantom_deq(m, h)
        res, k = 0.0, KMAX
    for layer in m.top:
        z = layer(z)
    return m.head(m.ln_f(z)), res, k


def peak_mem(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    return out, torch.cuda.max_memory_allocated() / 1024**3


def train_mode(mode, steps=1000, seed=0):
    torch.manual_seed(seed); random.seed(seed)
    m = DEQLM().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=6e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=6e-5)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    train = seqs[:int(0.95 * len(seqs))]
    rng = random.Random(42)
    t0 = time.time()
    for st in range(steps):
        x, y = get_batch(train, bs=32, device=DEV, rng=rng)
        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _, _ = forward_variant(m, x, mode)
            loss = F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1), ignore_index=0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            return loss.item()
        if st == 10:
            _, mem = peak_mem(step)
            print("[%s] st10 peak mem = %.2f GB" % (mode, mem), flush=True)
        else:
            step()
        sched.step()
        if st % 200 == 0:
            with torch.no_grad():
                x, y = get_batch(train, bs=32, device=DEV, rng=rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _, _ = forward_variant(m, x, mode)
                lv = F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1), ignore_index=0).item()
            print("[%s] st%d loss=%.4f (%.0fs)" % (mode, st, lv, time.time()-t0), flush=True)
    return m, mem


def eval_nll(m, steps=1000):
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    val = seqs[int(0.95 * len(seqs)):]
    m.eval()
    tot, tok = 0.0, 0
    with torch.no_grad():
        for i in range(20):
            x, y = get_batch(val, bs=32, device=DEV, rng=random.Random(999 + i))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _, _ = m(x)
            l = F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1), reduction="sum")
            tot += l.item(); tok += y.numel()
    m.train()
    return tot / tok


if __name__ == "__main__":
    import json
    results = {}
    for mode in ["full", "phantom"]:
        print("=" * 30, mode, "=" * 30, flush=True)
        m, mem = train_mode(mode)
        nll = eval_nll(m)
        print("[%s] final val NLL = %.4f (1000步)" % (mode, nll), flush=True)
        results[mode] = dict(peak_mem_gb=round(mem, 3), nll_1000=round(nll, 4))
    json.dump(results, open(OUT, "w"), indent=2)
    print("saved", OUT)
