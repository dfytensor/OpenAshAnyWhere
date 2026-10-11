#!/usr/bin/env python3
"""上限测量: (a) 地板(扫描恒等) (b) max-autotune chunked scan. 小 B=8 降内存, 与训练共存."""
import sys, os, time, random
import torch
import torch.nn.functional as F
import torch.optim as optim

HERE = r"F:\夸克\rubikgla"
sys.path.insert(0, HERE)
import deqlm_v31_multislot as M
from deqlm_v5_30m import get_batch, PT_CACHE
from scan import chunked_affine_scan

DEV = "cuda"
torch.manual_seed(0); random.seed(0)
seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
train = seqs[:int(0.95 * len(seqs))]
KV, S, DH = M.KV_H, M.SLOTS, M.DH


def fn_zero(k, v, rate, eta, q):
    B, L, G, dh = k.shape
    return torch.zeros(B, L, q.shape[2] * dh, device=k.device, dtype=torch.float32)


def fn_chunked(k, v, rate, eta, q):
    B, L, G, dh = k.shape
    H_ = q.shape[2]
    kn = k.to(torch.bfloat16)
    lam = torch.exp(-rate).to(torch.bfloat16)
    eta_ = eta.to(torch.bfloat16)
    I = torch.eye(dh, device=k.device, dtype=torch.bfloat16).view(1, 1, 1, dh, dh)
    A = lam.unsqueeze(-1) * (I - eta_.unsqueeze(-1) * (kn.unsqueeze(-1) * kn.unsqueeze(-2)))
    W = kn.unsqueeze(-1) * v.to(torch.bfloat16).unsqueeze(-2)
    _, S_ = chunked_affine_scan(A, W, chunk=32)
    Ss = S_.float().view(B, L, KV, S, dh, dh).sum(3)
    qg = q.float().view(B, L, KV, H_ // KV, dh)
    y = torch.einsum("blgpq,blgmq->blgmp", Ss, qg)
    return y.reshape(B, L, H_ * dh)


def run(tag, fn, compile_fn, n=12):
    torch.manual_seed(0)
    m = M.DeltaLoopLM().to(DEV)
    M.DELTA_FN = compile_fn(fn)
    opt = optim.AdamW(m.parameters(), lr=1e-5)
    rng = random.Random(42)
    m.train()
    for w in range(3):
        x, y = get_batch(train, bs=8, device=DEV, rng=rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(x)
        F.cross_entropy(lo.float().reshape(-1, M.V), y.reshape(-1), ignore_index=0).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n):
        x, y = get_batch(train, bs=8, device=DEV, rng=rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(x)
        l = F.cross_entropy(lo.float().reshape(-1, M.V), y.reshape(-1), ignore_index=0)
        l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print("%-30s %6.0f ms/步 (B=8)" % (tag, (time.time() - t0) / n * 1000))
    del m
    torch.cuda.empty_cache()


run("地板: 扫描恒等(compile)", fn_zero, lambda f: torch.compile(f))
run("chunked 默认模式", fn_chunked, lambda f: torch.compile(f))
run("chunked max-autotune-nocg", fn_chunked, lambda f: torch.compile(f, mode="max-autotune-no-cudagraphs"))
