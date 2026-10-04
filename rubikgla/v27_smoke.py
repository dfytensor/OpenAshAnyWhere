#!/usr/bin/env python3
"""v27 冒烟: 30 步验证构建/速度/loss/显存."""
import sys, os, time, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
import torch.optim as optim

import deqlm_v27_multiscale as V
from deqlm_v5_30m import get_batch, PT_CACHE
from deqlm_v19_tuned23m import masked_suffix_eval

torch.manual_seed(0)
random.seed(0)
seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
train, val = seqs[:int(0.95 * len(seqs))], seqs[:5000]

m = V.GLAHybrid2LM().to("cuda")
n = sum(p.numel() for p in m.parameters())
print("参数量: %.2fM (v26=23.60M)" % (n / 1e6))

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
print("compile 预热完成, 初始 NLL: %.4f / %.4f" % (a0, m0))

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
print("步时: %.0f ms/步 (v26=529ms) -> 全程 ETA %.1f 时" % (ms, ms / 1000 * 39695 / 3600))
print("显存: %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9))
a1, m1 = masked_suffix_eval(m, val)
print("30步后 NLL: 含pad=%.4f 真实=%.4f (初始 %.4f)" % (a1, m1, a0))
print("alpha: %.3f / %.3f" % (m.loop_a.alpha().item(), m.loop_c.alpha().item()))

# 速率门视野检查: 第0步 init 的 256 步保留率
with torch.no_grad():
    from deqlm_v27_multiscale import multiscale_inv_softplus
    r = F.softplus(m.pre.attn.w_rate.bias).cpu()
    print("init rate: min=%.4f max=%.3f | 256步保留率 max=%.3f (最慢通道)" % (
        r.min().item(), r.max().item(), (-r.max() * 0 + (-r) * 256).exp().max().item()))
