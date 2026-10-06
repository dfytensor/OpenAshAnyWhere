#!/usr/bin/env python3
"""v31b stability probe: 从 resume@2001 跑 40 步 @峰值lr4e-4, 盯 loss/grad NaN."""
import sys, os, random, math, time
import torch
import torch.nn.functional as F
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deqlm_v31_multislot as M
from deqlm_v5_30m import get_batch, PT_CACHE

DEV = "cuda"
torch.manual_seed(0)
random.seed(0)
seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
train = seqs[:int(0.95 * len(seqs))]
m = M.DeltaLoopLM().to(DEV)
ck = torch.load(os.path.join(HERE, "deqlm31_resume.pth"), map_location="cpu")
m.load_state_dict(ck["model"])
rng = random.Random(42)
rng.setstate(ck["rng"])
opt = optim.AdamW(m.parameters(), lr=4e-4, weight_decay=0.01, betas=(0.9, 0.95))
opt.load_state_dict(ck["opt"])
print("resume st=%d, MA-norm s=%.1f" % (ck["st"], m.pre.ffn.ln.s))
m.train()
t0 = time.time()
for i in range(40):
    for g in opt.param_groups:
        g["lr"] = 4e-4
    x, y = get_batch(train, bs=16, device=DEV, rng=rng)
    x2, y2 = get_batch(train, bs=16, device=DEV, rng=rng)
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
    if not (torch.isfinite(l1) and torch.isfinite(l2)):
        print("%d: NaN SKIP l1=%s l2=%s" % (i, l1.item(), l2.item()))
        opt.zero_grad(set_to_none=True)
        continue
    gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    if i % 5 == 0 or i > 34:
        print("%d: loss=%.4f gn=%.2f (%.0fms)" % (i, (0.5 * (l1 + l2)).item(), gn.item(), (time.time() - t0) / (i + 1) * 1000))
print("PROBE DONE, all finite" )
