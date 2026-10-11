#!/usr/bin/env python3
"""端到端步时对照: v31 现状(HS) vs chunked-c32 vs +其它优化."""
import sys, os, time, random, math
import torch
import torch.nn.functional as F
import torch.optim as optim

HERE = r"F:\夸克\rubikgla"
sys.path.insert(0, HERE)
import deqlm_v31_multislot as M
from deqlm_v5_30m import get_batch, PT_CACHE

DEV = "cuda"
torch.manual_seed(0); random.seed(0)
seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
train = seqs[:int(0.95 * len(seqs))]
m = M.DeltaLoopLM().to(DEV)
opt = optim.AdamW(m.parameters(), lr=1e-5)
rng = random.Random(42)


def run(tag, n=16):
    m.train()
    for w in range(3):  # warmup
        x, y = get_batch(train, bs=16, device=DEV, rng=rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(x)
        F.cross_entropy(lo.float().reshape(-1, M.V), y.reshape(-1), ignore_index=0).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n):
        x, y = get_batch(train, bs=16, device=DEV, rng=rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(x)
        l = F.cross_entropy(lo.float().reshape(-1, M.V), y.reshape(-1), ignore_index=0)
        l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print("%-24s %6.0f ms/步" % (tag, (time.time() - t0) / n * 1000))


run("v31 现状 (HS scan)")

# 换 chunked c32
from scan import chunked_affine_scan

def _chunked_fn(k, v, rate, eta, q):
    B, L, G, dh = k.shape
    H_ = q.shape[2]
    kn = k.float()
    lam = torch.exp(-rate.float())
    I = torch.eye(dh, device=k.device, dtype=torch.float32).view(1, 1, 1, dh, dh)
    A = lam.unsqueeze(-1) * (I - eta.float().unsqueeze(-1) * (kn.unsqueeze(-1) * kn.unsqueeze(-2)))
    W = kn.unsqueeze(-1) * v.float().unsqueeze(-2)
    _, S = chunked_affine_scan(A.to(torch.bfloat16), W.to(torch.bfloat16), chunk=32)
    qg = q.float().view(B, L, M.KV_H, H_ // M.KV_H, dh)
    qg = qg.unsqueeze(3).expand(B, L, M.KV_H, M.SLOTS, H_ // M.KV_H, dh).reshape(B, L, G, H_ // M.KV_H, dh)
    y = torch.einsum("blgpq,blgmq->blgmp", S.float(), qg)
    y = y.view(B, L, M.KV_H, M.SLOTS, H_ // M.KV_H, dh).sum(3)
    return y.reshape(B, L, H_ * dh)

M.DELTA_FN = torch.compile(_chunked_fn)
run("chunked-c32 scan")
