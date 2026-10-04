#!/usr/bin/env python3
"""v28 冒烟: 30 步验证构建/速度/loss/显存 + 初始视野."""
import sys, os, time, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
import torch.optim as optim

import deqlm_v29_deltarubik as V
from deqlm_v5_30m import get_batch, PT_CACHE
from deqlm_v19_tuned23m import masked_suffix_eval

torch.manual_seed(0)
random.seed(0)
seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
train, val = seqs[:int(0.95 * len(seqs))], seqs[:5000]

m = V.DeltaLoopLM().to("cuda")
n = sum(p.numel() for p in m.parameters())
print("params: %.2fM (v24=23.19M, %+.1f%%)" % (n / 1e6, (n / 23.19e6 - 1) * 100))

opt = optim.AdamW(m.parameters(), lr=4e-4, weight_decay=0.01, betas=(0.9, 0.95))
rng = random.Random(42)
B = 16

for i in range(2):
    x, y = get_batch(train, bs=B, device="cuda", rng=rng)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = m(x)
    l = F.cross_entropy(lo.float().reshape(-1, V.V), y.reshape(-1), ignore_index=0)
    opt.zero_grad(set_to_none=True)
    l.backward()
    opt.step()
torch.cuda.synchronize()
a0, m0 = masked_suffix_eval(m, val)
print("warmup done, init NLL: %.4f / %.4f" % (a0, m0))

t0 = time.time()
for st in range(30):
    x, y = get_batch(train, bs=B, device="cuda", rng=rng)
    x2, y2 = get_batch(train, bs=B, device="cuda", rng=rng)
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = m(x)
    l1 = F.cross_entropy(lo.float().reshape(-1, V.V), y.reshape(-1), ignore_index=0)
    if torch.isfinite(l1):
        (l1 * 0.5).backward()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo2 = m(x2)
    l2 = F.cross_entropy(lo2.float().reshape(-1, V.V), y2.reshape(-1), ignore_index=0)
    if torch.isfinite(l2):
        (l2 * 0.5).backward()
    if not (torch.isfinite(l1) and torch.isfinite(l2)):
        print("step %d non-finite: %.4f %.4f" % (st, l1, l2))
        continue
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    if st % 10 == 0:
        print("  st=%d loss=%.4f" % (st, (0.5 * (l1 + l2)).item()))
torch.cuda.synchronize()
ms = (time.time() - t0) / 30 * 1000
print("step: %.0f ms (v24=425ms) -> ETA %.1f h" % (ms, ms / 1000 * 39695 / 3600))
print("mem: %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9))
a1, m1 = masked_suffix_eval(m, val)
print("30-step NLL: all=%.4f masked=%.4f (init %.4f)" % (a1, m1, a0))
print("alpha: %.3f / %.3f" % (m.loop_a.alpha().item(), m.loop_c.alpha().item()))

with torch.no_grad():
    r = F.softplus(m.pre.attn.w_rate.bias).cpu()
    eta = torch.sigmoid(m.pre.attn.w_eta.bias).cpu()
    print("rate init: [%.4f, %.3f] eta init: %.2f | 256-step retention max=%.3f"
          % (r.min().item(), r.max().item(), eta.mean().item(), (-r * 256).exp().max().item()))
