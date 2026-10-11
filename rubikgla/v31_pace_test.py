#!/usr/bin/env python3
"""精确复刻 main() 训练步, 20 步计时. 定位 live 进程 ~900ms vs profiler 326ms 差异来源."""
import sys, os, random, time, math
import torch
import torch.nn.functional as F
import torch.optim as optim

HERE = r"F:\夸克\rubikgla"
sys.path.insert(0, HERE)
import deqlm_v31_multislot as M
from deqlm_v5_30m import get_batch, PT_CACHE

DEV = "cuda"
torch.manual_seed(0)
random.seed(0)
seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
train = seqs[:int(0.95 * len(seqs))]
m = M.DeltaLoopLM().to(DEV)
opt = optim.AdamW(m.parameters(), lr=4e-4, weight_decay=0.01, betas=(0.9, 0.95))
rng = random.Random(42)
B_MICRO = 16

times = []
t_alloc = 0.0
for st in range(24):
    ts = time.perf_counter()
    m.train()
    x, y = get_batch(train, bs=B_MICRO, device=DEV, rng=rng)
    x2, y2 = get_batch(train, bs=B_MICRO, device=DEV, rng=rng)
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = m(x)
    l1 = F.cross_entropy(lo.float().reshape(-1, M.V), y.reshape(-1), ignore_index=0)
    if torch.isfinite(l1):
        (l1 * 0.5).backward()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo2 = m(x2)
    l2 = F.cross_entropy(lo2.float().reshape(-1, M.V), y2.reshape(-1), ignore_index=0)
    if torch.isfinite(l2):
        (l2 * 0.5).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - ts
    times.append(dt * 1000)
    if st % 4 == 0 or st > 19:
        print("step %2d: %6.0f ms  loss=%.4f" % (st, dt * 1000, (0.5 * (l1 + l2)).item()))
print("中位 %d 步后平均: %.0f ms/步" % (len(times) - 8, sum(times[8:]) / len(times[8:])))
