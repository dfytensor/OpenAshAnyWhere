#!/usr/bin/env python3
"""profile 真实训练步: 找 780ms+ 的罪魁. B=16 双微批协议, 5 步采样."""
import sys, os, random, time
import torch
import torch.nn.functional as F
import torch.optim as optim

HERE = r"F:\夸克\rubikgla"
sys.path.insert(0, HERE)
import deqlm_v31_multislot as M
from deqlm_v5_30m import get_batch, PT_CACHE

DEV = "cuda"
torch.manual_seed(0); random.seed(0)
seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
train = seqs[:int(0.95 * len(seqs))]
m = M.DeltaLoopLM().to(DEV)
opt = optim.AdamW(m.parameters(), lr=1e-5)
rng = random.Random(42)
m.train()


def step():
    x, y = get_batch(train, bs=16, device=DEV, rng=rng)
    x2, y2 = get_batch(train, bs=16, device=DEV, rng=rng)
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = m(x)
    l1 = F.cross_entropy(lo.float().reshape(-1, M.V), y.reshape(-1), ignore_index=0)
    (l1 * 0.5).backward()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo2 = m(x2)
    l2 = F.cross_entropy(lo2.float().reshape(-1, M.V), y2.reshape(-1), ignore_index=0)
    (l2 * 0.5).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()


for _ in range(3):
    step()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(5):
    step()
torch.cuda.synchronize()
print("无 profiler 平均: %.0f ms/步" % ((time.time() - t0) / 5 * 1000))

from torch.profiler import profile, ProfilerActivity
for _ in range(2):
    step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
    for _ in range(3):
        step()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=14))
