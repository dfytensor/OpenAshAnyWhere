#!/usr/bin/env python3
"""v31 提速变体对照: chunked + 槽求和提前 + bf16 构造. 校验 loss 一致性."""
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


def fn_base(k, v, rate, eta, q):
    B, L, G, dh = k.shape
    H_ = q.shape[2]
    kn = k.float()
    lam = torch.exp(-rate.float())
    I = torch.eye(dh, device=k.device, dtype=torch.float32).view(1, 1, 1, dh, dh)
    A = lam.unsqueeze(-1) * (I - eta.float().unsqueeze(-1) * (kn.unsqueeze(-1) * kn.unsqueeze(-2)))
    W = kn.unsqueeze(-1) * v.float().unsqueeze(-2)
    _, S_ = chunked_affine_scan(A.to(torch.bfloat16), W.to(torch.bfloat16), chunk=32)
    qg = q.float().view(B, L, KV, H_ // KV, dh)
    qg = qg.unsqueeze(3).expand(B, L, KV, S, H_ // KV, dh).reshape(B, L, G, H_ // KV, dh)
    y = torch.einsum("blgpq,blgmq->blgmp", S_.float(), qg)
    y = y.view(B, L, KV, S, H_ // KV, dh).sum(3)
    return y.reshape(B, L, H_ * dh)


def fn_sumfirst(k, v, rate, eta, q):
    # 槽求和提前: Σ_s S_s 再单次 einsum (线性恒等)
    B, L, G, dh = k.shape
    H_ = q.shape[2]
    kn = k.float()
    lam = torch.exp(-rate.float())
    I = torch.eye(dh, device=k.device, dtype=torch.float32).view(1, 1, 1, dh, dh)
    A = lam.unsqueeze(-1) * (I - eta.float().unsqueeze(-1) * (kn.unsqueeze(-1) * kn.unsqueeze(-2)))
    W = kn.unsqueeze(-1) * v.float().unsqueeze(-2)
    _, S_ = chunked_affine_scan(A.to(torch.bfloat16), W.to(torch.bfloat16), chunk=32)
    Ss = S_.float().view(B, L, KV, S, dh, dh).sum(3)          # (B,L,KV,dh,dh)
    qg = q.float().view(B, L, KV, H_ // KV, dh)
    y = torch.einsum("blgpq,blgmq->blgmp", Ss, qg)            # (B,L,KV,KVG,dh)
    return y.reshape(B, L, H_ * dh)


def fn_bf16(k, v, rate, eta, q):
    # + bf16 直接构造 (省 fp32 大张量)
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


def run(tag, fn, n=16):
    torch.manual_seed(0)
    m = M.DeltaLoopLM().to(DEV)
    M.DELTA_FN = torch.compile(fn)
    opt = optim.AdamW(m.parameters(), lr=1e-5)
    rng = random.Random(42)
    m.train()
    losses = []
    for w in range(3):
        x, y = get_batch(train, bs=16, device=DEV, rng=rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(x)
        l = F.cross_entropy(lo.float().reshape(-1, M.V), y.reshape(-1), ignore_index=0)
        losses.append(l.item())
        l.backward(); opt.step(); opt.zero_grad(set_to_none=True)
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
    print("%-22s %6.0f ms/步  loss3=%s" % (tag, (time.time() - t0) / n * 1000,
          ["%.4f" % v for v in losses]))


run("chunked-base", fn_base)
run("sumfirst", fn_sumfirst)
run("sumfirst+bf16", fn_bf16)
print("基准: 现状HS=859ms (同脚本早前测)")
