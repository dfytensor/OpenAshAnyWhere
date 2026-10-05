#!/usr/bin/env python3
"""bisect v31 grad NaN: MA-norm? DeltaAttn? scan fn? rho(it>0)?"""
import sys, os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deqlm_v31_multislot as M

DEV = "cuda"
torch.manual_seed(0)
xe = torch.randn(16, 256, M.D, device=DEV)


def report(name, mod, loss):
    bad = [n for n, p in mod.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    print("%-22s loss=%.4f  grad_nan=%s" % (name, loss.item(), bad[:5] if bad else "NONE"))
    mod.zero_grad(set_to_none=True)


# 1) MA-RMSNorm alone
ln = M.MARMSNorm(M.D).to(DEV)
with torch.autocast("cuda", dtype=torch.bfloat16):
    l = ln(xe).float().pow(2).mean()
l.backward()
report("MARMSNorm", ln, l)

# 2) DeltaAttn full (it=0)
att = M.DeltaAttn().to(DEV).train()
with torch.autocast("cuda", dtype=torch.bfloat16):
    l = att._scan_core(xe, 0).float().pow(2).mean()
l.backward()
report("DeltaAttn(it=0)", att, l)

# 3) DeltaAttn (it=3, rho path)
att3 = M.DeltaAttn().to(DEV).train()
with torch.autocast("cuda", dtype=torch.bfloat16):
    l = att3._scan_core(xe, 3).float().pow(2).mean()
l.backward()
report("DeltaAttn(it=3)", att3, l)

# 4) scan fn alone (v31 shapes, 无 compile)
k = torch.randn(16, 256, 10, 32, device=DEV, requires_grad=True)
v = torch.randn(16, 256, 10, 32, device=DEV, requires_grad=True)
rate = torch.rand(16, 256, 10, 32, device=DEV, requires_grad=True) * 0.1
eta = torch.sigmoid(torch.randn(16, 256, 10, 32, device=DEV, requires_grad=True) - 2)
q = torch.randn(16, 256, 10, 32, device=DEV, requires_grad=True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = M.delta_scan_kv(k, v, rate, eta, q)
out.float().pow(2).mean().backward()
print("scan raw: out_finite=%s  k_grad_nan=%s  q_grad_nan=%s" % (
    torch.isfinite(out).all().item(),
    not torch.isfinite(k.grad).all().item(),
    not torch.isfinite(q.grad).all().item()))

# 5) scan fn compiled
M.DELTA_FN2 = torch.compile(M.delta_scan_kv)
k2 = k.detach().clone().requires_grad_(True)
q2 = q.detach().clone().requires_grad_(True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    out2 = M.DELTA_FN2(k2, v.detach(), rate.detach(), eta.detach(), q2)
out2.float().pow(2).mean().backward()
print("scan compiled: out_finite=%s  k_grad_nan=%s" % (
    torch.isfinite(out2).all().item(), not torch.isfinite(k2.grad).all().item()))
