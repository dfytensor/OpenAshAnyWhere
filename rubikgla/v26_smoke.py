#!/usr/bin/env python3
"""v26 冒烟: 30 步验证构建/速度/loss/显存."""
import sys, os, time, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
import torch.optim as optim

import deqlm_v26_gla_full as V
from deqlm_v5_30m import get_batch, PT_CACHE
from deqlm_v19_tuned23m import masked_suffix_eval

torch.manual_seed(0)
random.seed(0)
seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
train, val = seqs[:int(0.95 * len(seqs))], seqs[:5000]

m = V.GLAHybrid2LM().to("cuda")
n = sum(p.numel() for p in m.parameters())
print("参数量: %.2fM (v24=23.19M, 差 %+.1f%%)" % (n / 1e6, (n / 1e6 / 23.19 - 1) * 100))

opt = optim.AdamW(m.parameters(), lr=4e-4, weight_decay=0.01, betas=(0.9, 0.95))
rng = random.Random(42)
B = 16

# 预热 compile (2 步)
for i in range(2):
    x, y = get_batch(train, bs=B, device="cuda", rng=rng)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = m(x)
    l = F.cross_entropy(lo.float().reshape(-1, V.V), y.reshape(-1), ignore_index=0)
    opt.zero_grad(set_to_none=True)
    l.backward()
    opt.step()
torch.cuda.synchronize()
print("compile 预热完成, 初始 NLL:", masked_suffix_eval(m, val))

a0, m0 = masked_suffix_eval(m, val)
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
        print("第 %d 步 loss 非有限: %.4f %.4f" % (st, l1, l2))
        continue
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    if st % 10 == 0:
        print("  st=%d loss=%.4f" % (st, (0.5 * (l1 + l2)).item()))
torch.cuda.synchronize()
ms = (time.time() - t0) / 30 * 1000
print("步时: %.0f ms/步 (v24=425ms) -> 全程 ETA %.1f 时" % (ms, ms / 1000 * 39695 / 3600))
print("显存: %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9))
a1, m1 = masked_suffix_eval(m, val)
print("30步后 NLL: 含pad=%.4f 真实=%.4f (初始 %.4f)" % (a1, m1, a0))
print("alpha: %.3f / %.3f" % (m.loop_a.alpha().item(), m.loop_c.alpha().item()))
