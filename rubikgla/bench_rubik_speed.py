#!/usr/bin/env python3
"""rubik 桥速度基准: HS vs chunked, checkpoint vs 无, fp32."""
import sys, time
import torch

sys.path.insert(0, r"F:\夸克\rubikgla")
from scan import affine_scan, chunked_affine_scan

DEV = "cuda"
B, L, H, dh = 16, 256, 10, 32
torch.manual_seed(0)


def make_inputs():
    A0 = torch.randn(B, L, H, dh, dh, device=DEV) * 0.05
    G = A0  # 视作已算好的 G
    lam = torch.sigmoid(torch.randn(1, H, dh, 1, device=DEV)) * 0.5
    k = torch.randn(B, L, H, dh, device=DEV)
    v = torch.randn(B, L, H, dh, device=DEV)
    W_ = (k.unsqueeze(-1) * v.unsqueeze(-2)).requires_grad_(True)
    Ag = (lam * G).detach().requires_grad_(True)
    return Ag, W_


def bench(name, fn, iters=5):
    Ag, W_ = make_inputs()
    out = fn(Ag, W_)
    out.float().pow(2).mean().backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        Ag2, W2 = make_inputs()
        o = fn(Ag2, W2)
        o.float().pow(2).mean().backward()
    torch.cuda.synchronize()
    print("%-28s %6.0f ms/次 (fwd+bwd)" % (name, (time.time() - t0) / iters * 1000))


bench("HS affine_scan", lambda A, W: affine_scan(A, W)[1])
bench("chunked chunk=64", lambda A, W: chunked_affine_scan(A, W, chunk=64)[1])
bench("chunked chunk=128", lambda A, W: chunked_affine_scan(A, W, chunk=128)[1])
