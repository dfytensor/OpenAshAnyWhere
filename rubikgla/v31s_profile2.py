#!/usr/bin/env python3
"""v31s 编译后 CPU/GPU 分解: 还能榨多少?"""
import sys, os, random, time
import torch
import torch.nn.functional as F

HERE = r"F:\夸克\rubikgla"
sys.path.insert(0, HERE)
from deqlm_v5_30m import get_batch, PT_CACHE
import deqlm_v31_spd as Mod

DEV = "cuda"
torch.manual_seed(0); random.seed(0)
seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
train = seqs[: int(0.95 * len(seqs))]
m = Mod.DeltaLoopLM().to(DEV)
opt = torch.optim.AdamW(m.parameters(), lr=4e-4, weight_decay=0.01, betas=(0.9, 0.95), fused=True)
rng = random.Random(42)
m.train()
for st in range(6):  # warmup+compile
    x, y = get_batch(train, bs=16, device=DEV, rng=rng)
    x2, y2 = get_batch(train, bs=16, device=DEV, rng=rng)
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = m(x)
    F.cross_entropy(lo.float().reshape(-1, Mod.V), y.reshape(-1), ignore_index=0).backward()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo2 = m(x2)
    F.cross_entropy(lo2.float().reshape(-1, Mod.V), y2.reshape(-1), ignore_index=0).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()

# wall vs GPU 时间 (10 步)
torch.cuda.synchronize(); t0 = time.perf_counter()
ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
ev0.record()
for st in range(10):
    x, y = get_batch(train, bs=16, device=DEV, rng=rng)
    x2, y2 = get_batch(train, bs=16, device=DEV, rng=rng)
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = m(x)
    F.cross_entropy(lo.float().reshape(-1, Mod.V), y.reshape(-1), ignore_index=0).backward()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo2 = m(x2)
    F.cross_entropy(lo2.float().reshape(-1, Mod.V), y2.reshape(-1), ignore_index=0).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
ev1.record()
torch.cuda.synchronize()
wall = (time.perf_counter() - t0) / 10 * 1000
gpu = ev0.elapsed_time(ev1) / 10
print("v31s: wall=%.0f ms/步  GPU(忙)=%.0f ms  空隙=%.0f ms" % (wall, gpu, wall - gpu), flush=True)

from torch.profiler import profile, ProfilerActivity
x, y = get_batch(train, bs=16, device=DEV, rng=rng)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    x, y = get_batch(train, bs=16, device=DEV, rng=rng)
    x2, y2 = get_batch(train, bs=16, device=DEV, rng=rng)
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = m(x)
    F.cross_entropy(lo.float().reshape(-1, Mod.V), y.reshape(-1), ignore_index=0).backward()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo2 = m(x2)
    F.cross_entropy(lo2.float().reshape(-1, Mod.V), y2.reshape(-1), ignore_index=0).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    torch.cuda.synchronize()
ev = prof.key_averages()
for row in sorted(ev, key=lambda r: -r.self_cpu_time_total)[:8]:
    print("%-40s CPUself=%6.1fms  CUDA=%6.1fms  n=%d" % (
        row.key[:40], row.self_cpu_time_total / 1000, row.self_device_time_total / 1000, row.count))
print("CPU total=%.0fms  CUDA total=%.0fms (整步双微批)" % (
    sum(r.self_cpu_time_total for r in ev) / 1000, sum(r.self_device_time_total for r in ev) / 1000))
