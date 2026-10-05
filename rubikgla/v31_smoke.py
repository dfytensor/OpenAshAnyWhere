#!/usr/bin/env python3
"""v31 smoke: 参数配平 + 前反向 + 数值 + 速率."""
import sys, os, time, math
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deqlm_v31_multislot as M
from deqlm_v5_30m import PT_CACHE

DEV = "cuda"
m = M.DeltaLoopLM().to(DEV)
n = sum(p.numel() for p in m.parameters())
print("params: %.2fM (v24=23.19M, v30=22.92M)" % (n / 1e6))
assert abs(n / 1e6 - 23.19) < 0.25, "param budget off"

seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
tr = seqs[:int(0.95 * len(seqs))]
x, y = M.get_batch(tr, bs=16, device=DEV, rng=__import__("random").Random(1))

m.train()
with torch.autocast("cuda", dtype=torch.bfloat16):
    lo = m(x)
l = torch.nn.functional.cross_entropy(lo.float().reshape(-1, M.V), y.reshape(-1), ignore_index=0)
print("fwd: loss=%.4f finite=%s" % (l.item(), torch.isfinite(l).item()))
l.backward()
gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
print("grad norm: %.3f finite=%s" % (gn.item(), math.isfinite(gn.item())))
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
opt.step()

# 检查: 槽初始 rate 多尺度 + eta 0.12 + MA-norm 幅度
blk = m.loop_a.attn.attn
r = torch.nn.functional.softplus(blk.w_rate.bias)
print("rate init: min=%.4f max=%.4f (floor 5e-4~0.5)" % (r.min().item(), r.max().item()))
print("eta init: %.4f (target 0.12)" % torch.sigmoid(blk.w_eta.bias).mean().item())
print("慢通道 256步保留: %.3f" % float(torch.exp(-r.min() * 256)))
ln = blk.ln
with torch.no_grad():
    z = torch.randn(2, 8, M.D, device=DEV)
    out = ln(z)
    print("MA-norm: out_std=%.3f (纯RMSNorm×2≈2.0)" % out.std().item())

m.eval()
with torch.autocast("cuda", dtype=torch.bfloat16):
    with torch.no_grad():
        lo = m(x)
print("eval fwd: finite=%s" % torch.isfinite(lo).all().item())
torch.cuda.reset_peak_memory_stats()
for i in range(3):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = m(x)
    l = torch.nn.functional.cross_entropy(lo.float().reshape(-1, M.V), y.reshape(-1), ignore_index=0)
    l.backward()
    m.zero_grad(set_to_none=True)
print("peak mem: %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9))
t0 = time.time()
for i in range(5):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = m(x)
    l = torch.nn.functional.cross_entropy(lo.float().reshape(-1, M.V), y.reshape(-1), ignore_index=0)
    l.backward()
    m.zero_grad(set_to_none=True)
torch.cuda.synchronize()
print("train step: %.0fms (v30≈900ms 邻居占用下)" % ((time.time() - t0) / 5 * 1000))
print("SMOKE OK")
