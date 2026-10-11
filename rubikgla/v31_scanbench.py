#!/usr/bin/env python3
"""scan variant bench @ v31 shapes (B=16,L=256,G=10,dh=32) fwd+bwd."""
import sys, os, time
import torch
sys.path.insert(0, r"F:\夸克\rubikgla")
from scan import affine_scan, chunked_affine_scan

DEV = "cuda"
torch.manual_seed(0)
B, L, G, dh = 16, 256, 10, 32
A = (torch.rand(B, L, G, dh, dh, device=DEV) * 0.2).to(torch.bfloat16)
W = (torch.randn(B, L, G, dh, dh, device=DEV) * 0.1).to(torch.bfloat16)


def bench(tag, fn):
    Af = A.clone().requires_grad_(True)
    Wf = W.clone().requires_grad_(True)
    for _ in range(3):
        _, S = fn(Af, Wf)
        S.float().pow(2).mean().backward()
        Af.grad = None; Wf.grad = None
    torch.cuda.synchronize(); t0 = time.time()
    n = 10
    for _ in range(n):
        _, S = fn(Af, Wf)
        S.float().pow(2).mean().backward()
        Af.grad = None; Wf.grad = None
    torch.cuda.synchronize()
    print("%-28s %7.1f ms fwd+bwd" % (tag, (time.time() - t0) / n * 1000))


bench("affine(HS) raw", affine_scan)
bench("chunked c32 raw", lambda a, w: chunked_affine_scan(a, w, chunk=32))
bench("chunked c64 raw", lambda a, w: chunked_affine_scan(a, w, chunk=64))
C_HS = torch.compile(affine_scan)
C_C32 = torch.compile(lambda a, w: chunked_affine_scan(a, w, chunk=32))
C_C64 = torch.compile(lambda a, w: chunked_affine_scan(a, w, chunk=64))
bench("affine(HS) compiled", C_HS)
bench("chunked c32 compiled", C_C32)
bench("chunked c64 compiled", C_C64)
