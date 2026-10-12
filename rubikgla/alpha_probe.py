#!/usr/bin/env python3
"""α 自由度探针: 冻结模型, 只优化每迭代 α_j (或共享 α 对照), 看 NLL 改善空间.
v31/v24 各跑: (a) 共享 α 重调 (b) 10 个独立 α_j. 训练 300 步在 train 切分, eval 在 val.
"""
import sys, os, random, math, json, types
import torch
import torch.nn.functional as F

HERE = r"F:\夸克\rubikgla"
sys.path.insert(0, HERE)
from deqlm_v5_30m import get_batch, PT_CACHE
from deqlm_v19_tuned23m import masked_suffix_eval

DEV = "cuda"
torch.manual_seed(0)
seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
train, val = seqs[: int(0.95 * len(seqs))], seqs[:5000]

import deqlm_v24_hybrid2fast as M24
import deqlm_v31_multislot as M31


def logit(p):
    return math.log(p / (1 - p))


def probe(name, cls, ckpt, has_it):
    m = cls()
    m.load_state_dict(torch.load(os.path.join(HERE, ckpt), map_location="cpu"))
    m = m.to(DEV).eval()
    a0, c0 = m.loop_a.alpha().item(), m.loop_c.alpha().item()
    ka, kc = m.loop_a.kmax, m.loop_c.kmax
    print("== %s: 训练 α=(%.4f, %.4f) k=(%d,%d)" % (name, a0, c0, ka, kc), flush=True)

    _, base = masked_suffix_eval(m, val)
    print("%s 基线 NLL=%.4f" % (name, base), flush=True)

    for p_ in m.parameters():
        p_.requires_grad_(False)

    for mode in ("shared", "per_iter"):
        n_a = 1 if mode == "shared" else ka
        n_c = 1 if mode == "shared" else kc
        th_a = torch.tensor([logit(a0)] * n_a, device=DEV, requires_grad=True)
        th_c = torch.tensor([logit(c0)] * n_c, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([th_a, th_c], lr=5e-3)

        def fwd(block, th, chk):
            def f(z0):
                z = z0
                al = torch.sigmoid(th)
                for j in range(block.kmax):
                    a_j = al[0] if th.numel() == 1 else al[j]
                    if chk:
                        z = z + a_j * block.attn(z, it=j)
                    else:
                        z = z + a_j * block.attn(z)
                return z
            return f

        m.loop_a.forward = fwd(m.loop_a, th_a, has_it)
        m.loop_c.forward = fwd(m.loop_c, th_c, has_it)

        rng = random.Random(123)
        m.train()
        for st in range(300):
            x, y = get_batch(train, bs=16, device=DEV, rng=rng)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = m(x)
            l = F.cross_entropy(lo.float().reshape(-1, M31.V), y.reshape(-1), ignore_index=0)
            l.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        m.eval()
        _, after = masked_suffix_eval(m, val)
        al_a = torch.sigmoid(th_a).tolist()
        al_c = torch.sigmoid(th_c).tolist()
        print("%s [%s] NLL=%.4f (Δ=%.4f) α_a=%s" % (
            name, mode, after, after - base,
            ["%.3f" % v for v in al_a]), flush=True)
        print("   α_c=%s" % ["%.3f" % v for v in al_c], flush=True)


probe("v31", M31.DeltaLoopLM, "deqlm31_pt_full.pth", True)
probe("v24", M24.Hybrid2FastLM, "deqlm24_pt_full.pth", False)
print("DONE")
