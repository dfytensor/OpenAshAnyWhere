#!/usr/bin/env python3
"""delta_triton 正确性测试: 对照 torch fp32 显式递归 + 大形状速度."""
import sys, os, time
import torch

HERE = r"F:\夸克\rubikgla"
sys.path.insert(0, HERE)
from delta_triton import delta_scan_triton

DEV = "cuda"
torch.manual_seed(0)


def ref_y(k, v, rate, eta, q, kv, slots, kvg, grot=None):
    B, L, G, DH = k.shape
    k = k.detach().float(); v = v.detach().float(); rate = rate.detach().float()
    eta = eta.detach().float(); q = q.detach().float()
    k.requires_grad_(True); v.requires_grad_(True); rate.requires_grad_(True)
    eta.requires_grad_(True); q.requires_grad_(True)
    extras = []
    if grot is not None:
        grot = grot.detach().float().requires_grad_(True)
        extras.append(grot)
    S = torch.zeros(B, G, DH, DH, device=k.device, dtype=torch.float32)
    out = torch.zeros(B, L, G, kvg, DH, device=k.device, dtype=torch.float32)
    for t in range(L):
        kt, vt = k[:, t], v[:, t]
        lam = torch.exp(-rate[:, t])
        if grot is not None:
            R = torch.einsum("bij,bgjk->bgik", grot[:, t], S)
        else:
            R = S
        rr = torch.einsum("bgi,bgij->bgj", kt, R)
        S = lam[..., None] * (R - eta[:, t][..., None] * kt[..., :, None] * rr[..., None, :]) \
            + kt[..., :, None] * vt[..., None, :]
        for g in range(G):
            kvg_i = g // slots
            for m in range(kvg):
                h = kvg_i * kvg + m
                out[:, t, g, m] = torch.einsum("bi,bij->bj", q[:, t, h], S[:, g])
    return out, (k, v, rate, eta, q) + tuple(extras)


B, L, KV, SLOTS, KVG, DH = 2, 64, 5, 2, 2, 32
G = KV * SLOTS
H = KV * KVG
dtype = torch.bfloat16

k = (torch.randn(B, L, G, DH, device=DEV) * 0.3).to(dtype).requires_grad_(True)
v = (torch.randn(B, L, G, DH, device=DEV) * 0.3).to(dtype).requires_grad_(True)
rate = (torch.rand(B, L, G, DH, device=DEV) * 0.3 + 1e-3).to(dtype).requires_grad_(True)
eta = (torch.rand(B, L, G, DH, device=DEV) * 0.3 + 0.05).to(dtype).requires_grad_(True)
q = (torch.randn(B, L, H, DH, device=DEV) * 0.5).to(dtype).requires_grad_(True)

print("== 前向 ==")
import math as _m
# 构造近恒等旋转 G = exp(θ[uwᵀ−wuᵀ]) (θ≤π/8, 单位 u,w)
_u = torch.randn(B, L, DH, device=DEV)
_w = torch.randn(B, L, DH, device=DEV)
_u = _u / _u.norm(dim=-1, keepdim=True)
_w = _w / _w.norm(dim=-1, keepdim=True)
_th = (torch.rand(B, L, 1, device=DEV) * 0.3 + 0.05)
_Mm = _u.unsqueeze(-1) * _w.unsqueeze(-2) - _w.unsqueeze(-1) * _u.unsqueeze(-2)
_GM = _th.unsqueeze(-1) * _Mm
_GM = _GM.float()
_E = torch.eye(DH, device=DEV).view(1, 1, DH, DH)
_P = _GM.clone()
for i in range(2, 7):
    _E = _E + _P
    _P = _P @ _GM / i
grot = _E.detach().clone().requires_grad_(True)   # fp32 旋转 (kernel 内部转 fp32)

y_tri = delta_scan_triton(k, v, rate, eta, q, KV, SLOTS, KVG, grot)
grot_ref = _E.detach().clone().requires_grad_(True)
y_ref, refs = ref_y(k, v, rate, eta, q, KV, SLOTS, KVG, grot=grot_ref)
err = (y_tri.float() - y_ref).abs()
print("y(带旋转): max_abs=%.2e  mean_rel=%.2e  y_ref_std=%.3f"
      % (err.max().item(), (err / (y_ref.abs() + 1e-3)).mean().item(), y_ref.std().item()))
assert err.max().item() < 0.05

print("== 反向 ==")
dy = torch.randn_like(y_tri)
y_tri.backward(dy)
y_ref.backward(dy.float())
ok = True
refsA = [refs[5].grad] + [refs[i].grad for i in range(5)]
namesA = ["dG", "dk", "dv", "drate", "deta", "dq"]
gradsA = [grot.grad] + [k.grad, v.grad, rate.grad, eta.grad, q.grad]
for name, a, b in zip(namesA, gradsA, refsA):
    a = a.float()
    scale = b.abs().mean().item() + 1e-6
    rel_err = (a - b).abs().mean().item() / scale
    bad = rel_err > 0.05
    ok = ok and not bad
    print("%-6s |a|=%.3e |ref|=%.3e  rel=%.2e  %s"
          % (name, a.abs().mean().item(), b.abs().mean().item(), rel_err, "FAIL" if bad else "OK"))
print("反向:", "PASS" if ok else "FAIL")

print("== 速度 (B=16, L=256, 带旋转) ==")
B2, L2 = 16, 256
k3 = torch.randn(B2, L2, G, DH, device=DEV).to(dtype).requires_grad_(True)
v3 = torch.randn(B2, L2, G, DH, device=DEV).to(dtype).requires_grad_(True)
r3 = (torch.rand(B2, L2, G, DH, device=DEV) * 0.3 + 1e-3).to(dtype).requires_grad_(True)
e3 = (torch.rand(B2, L2, G, DH, device=DEV) * 0.3 + 0.05).to(dtype).requires_grad_(True)
q3 = torch.randn(B2, L2, H, DH, device=DEV).to(dtype).requires_grad_(True)
g3 = torch.eye(DH, device=DEV).view(1, 1, DH, DH).repeat(B2, L2, 1, 1).requires_grad_(True)
for _ in range(3):
    y3 = delta_scan_triton(k3, v3, r3, e3, q3, KV, SLOTS, KVG, g3)
    y3.float().pow(2).mean().backward()
    for x in (k3, v3, r3, e3, q3, g3):
        x.grad = None
torch.cuda.synchronize(); t0 = time.time()
n = 10
for _ in range(n):
    y3 = delta_scan_triton(k3, v3, r3, e3, q3, KV, SLOTS, KVG, g3)
    y3.float().pow(2).mean().backward()
    for x in (k3, v3, r3, e3, q3, g3):
        x.grad = None
torch.cuda.synchronize()
print("triton fwd+bwd 带旋转: %.1f ms (无旋转 3.8ms, chunked-inductor 20ms)" % ((time.time() - t0) / n * 1000))
print("DONE")
