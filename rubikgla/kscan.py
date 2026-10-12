#!/usr/bin/env python3
"""k 扫描探针: 循环迭代利用率曲线 (anytime 属性).
截断 loop_a/loop_c 的迭代数 k=1..10, 测掩码后缀 NLL(k).
判读: NLL(5)≈NLL(10) → 动态早退有空间; 陡峭 → 迭代承重不可跳.
"""
import sys, os, json
import torch

HERE = r"F:\夸克\rubikgla"
sys.path.insert(0, HERE)
from deqlm_v5_30m import PT_CACHE
from deqlm_v19_tuned23m import masked_suffix_eval

DEV = "cuda"
torch.manual_seed(0)
seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
val = seqs[:5000]

import deqlm_v24_hybrid2fast as M24
import deqlm_v31_multislot as M31

KS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
out = {}


def scan(name, mod, cls):
    m = cls()
    m.load_state_dict(torch.load(os.path.join(HERE, "deqlm%s_pt_full.pth" % name.split("v")[1]), map_location="cpu"))
    m = m.to(DEV).eval()
    res = {}
    for k in KS:
        m.loop_a.kmax = k
        m.loop_c.kmax = k
        a, mk = masked_suffix_eval(m, val)
        res[k] = mk
        print("%s 对称 k=%2d: masked=%.4f" % (name, k, mk), flush=True)
    # 非对称: 只截 loop_a / 只截 loop_c (在 k=3,5 两个点)
    for k in (3, 5):
        m.loop_a.kmax = k
        m.loop_c.kmax = 10
        a, mk = masked_suffix_eval(m, val)
        res["a%d_c10" % k] = mk
        print("%s 只截A k=%2d: masked=%.4f" % (name, k, mk), flush=True)
        m.loop_a.kmax = 10
        m.loop_c.kmax = k
        a, mk = masked_suffix_eval(m, val)
        res["a10_c%d" % k] = mk
        print("%s 只截C k=%2d: masked=%.4f" % (name, k, mk), flush=True)
    del m
    torch.cuda.empty_cache()
    return res


out["v24"] = scan("v24", None, M24.Hybrid2FastLM)
out["v31"] = scan("v31", None, M31.DeltaLoopLM)
json.dump(out, open(os.path.join(HERE, "kscan_results.json"), "w"), indent=2)
print("saved -> kscan_results.json")
