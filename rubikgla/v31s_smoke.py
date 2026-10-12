#!/usr/bin/env python3
"""v31s 加速对照: v31(基线) vs v31s(整站编译+fused) — 数值平行性 + 步时."""
import sys, os, random, time
import torch
import torch.nn.functional as F

HERE = r"F:\夸克\rubikgla"
sys.path.insert(0, HERE)
from deqlm_v5_30m import get_batch, PT_CACHE

DEV = "cuda"
seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
train = seqs[: int(0.95 * len(seqs))]


def build(modname):
    torch.manual_seed(0)
    random.seed(0)
    import importlib
    Mod = importlib.import_module(modname)
    m = Mod.DeltaLoopLM().to(DEV)
    return m, Mod


def run_steps(m, Mod, n, lr=4e-4, fused=False, warm=0, tag=""):
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95), fused=fused)
    rng = random.Random(42)
    m.train()
    losses = []
    t0 = None
    for st in range(n + warm):
        x, y = get_batch(train, bs=16, device=DEV, rng=rng)
        x2, y2 = get_batch(train, bs=16, device=DEV, rng=rng)
        if st == warm:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(x)
        l1 = F.cross_entropy(lo.float().reshape(-1, Mod.V), y.reshape(-1), ignore_index=0)
        (l1 * 0.5).backward()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo2 = m(x2)
        l2 = F.cross_entropy(lo2.float().reshape(-1, Mod.V), y2.reshape(-1), ignore_index=0)
        (l2 * 0.5).backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        if st >= warm:
            losses.append((0.5 * (l1 + l2)).item())
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n * 1000
    if tag:
        print("%s: %d 步计时 %.0f ms/步" % (tag, n, dt), flush=True)
    return losses


print("== 数值平行性 (3 步, 同初始权重/同批流) ==", flush=True)
m1, M1 = build("deqlm_v31_multislot")
L1 = run_steps(m1, M1, 3, warm=1, tag="v31   基线")
del m1; torch.cuda.empty_cache()

m2, M2 = build("deqlm_v31_spd")
print("v31s: 首次前向将触发编译 (~几分钟)...", flush=True)
L2 = run_steps(m2, M2, 3, warm=1, fused=True, tag="v31s  编译版")
print("loss 对照:")
for i, (a, b) in enumerate(zip(L1, L2)):
    print("  step%d: v31=%.4f  v31s=%.4f  Δ=%+.5f" % (i, a, b, b - a))
print("== 步时 (8 步) ==", flush=True)
run_steps(m2, M2, 8, fused=True, tag="v31s  编译+fused")
del m2; torch.cuda.empty_cache()
m3, M3 = build("deqlm_v31_multislot")
run_steps(m3, M3, 8, tag="v31   基线(复测)")
print("DONE")
