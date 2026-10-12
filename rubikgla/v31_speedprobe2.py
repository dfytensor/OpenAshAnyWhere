#!/usr/bin/env python3
"""v31 训练步细粒度计时: 分段 wall time + CPU 热点 + fused-AdamW 对照.
目标: 找出 326ms wall vs 193ms GPU 的 133ms 去向, 定位加速杠杆.
"""
import sys, os, random, time
import torch
import torch.nn.functional as F

HERE = r"F:\夸克\rubikgla"
sys.path.insert(0, HERE)
import deqlm_v31_multislot as M
from deqlm_v5_30m import get_batch, PT_CACHE

DEV = "cuda"
torch.manual_seed(0); random.seed(0)
seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
train = seqs[: int(0.95 * len(seqs))]
m = M.DeltaLoopLM().to(DEV)
opt = torch.optim.AdamW(m.parameters(), lr=1e-5, weight_decay=0.01, betas=(0.9, 0.95))
rng = random.Random(42)
B = 16


def timed(use_fused=False, n=8):
    if use_fused:
        opt2 = torch.optim.AdamW(m.parameters(), lr=1e-5, weight_decay=0.01,
                                 betas=(0.9, 0.95), fused=True)
    else:
        opt2 = opt
    segs = dict(data=0.0, fwd1=0.0, bwd1=0.0, fwd2=0.0, bwd2=0.0, clip=0.0, step=0.0)
    m.train()
    for it in range(n + 3):
        track = it >= 3
        t0 = time.perf_counter()
        x, y = get_batch(train, bs=B, device=DEV, rng=rng)
        x2, y2 = get_batch(train, bs=B, device=DEV, rng=rng)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        opt2.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(x)
        l1 = F.cross_entropy(lo.float().reshape(-1, M.V), y.reshape(-1), ignore_index=0)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        (l1 * 0.5).backward()
        torch.cuda.synchronize(); t3 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo2 = m(x2)
        l2 = F.cross_entropy(lo2.float().reshape(-1, M.V), y2.reshape(-1), ignore_index=0)
        torch.cuda.synchronize(); t4 = time.perf_counter()
        (l2 * 0.5).backward()
        torch.cuda.synchronize(); t5 = time.perf_counter()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        torch.cuda.synchronize(); t6 = time.perf_counter()
        opt2.step()
        torch.cuda.synchronize(); t7 = time.perf_counter()
        if track:
            segs["data"] += t1 - t0
            segs["fwd1"] += t2 - t1
            segs["bwd1"] += t3 - t2
            segs["fwd2"] += t4 - t3
            segs["bwd2"] += t5 - t4
            segs["clip"] += t6 - t5
            segs["step"] += t7 - t6
    tot = sum(segs.values()) / n * 1000
    print(("fused " if use_fused else "baseline") + " 每步分段 (ms): " +
          "  ".join("%s=%.0f" % (k, v / n * 1000) for k, v in segs.items()) +
          "  总=%.0f" % tot, flush=True)
    return tot


timed(False)
timed(True)

# CPU 热点: profiler 3 步
from torch.profiler import profile, ProfilerActivity
for _ in range(2):
    x, y = get_batch(train, bs=B, device=DEV, rng=rng)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = m(x)
    F.cross_entropy(lo.float().reshape(-1, M.V), y.reshape(-1), ignore_index=0).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    x, y = get_batch(train, bs=B, device=DEV, rng=rng)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = m(x)
    F.cross_entropy(lo.float().reshape(-1, M.V), y.reshape(-1), ignore_index=0).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
