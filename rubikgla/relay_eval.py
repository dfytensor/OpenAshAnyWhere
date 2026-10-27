#!/usr/bin/env python3
"""接力实验评测: 加载 4 个 checkpoint, 只跑 [129,256) NLL 三连."""
import sys, os, random, json
import torch
import torch.nn.functional as F

HERE = r"F:\夸克\rubikgla"
sys.path.insert(0, HERE)
from deqlm_v5_30m import PT_CACHE
import deqlm_v35_relay as M

DEV = "cuda"
torch.manual_seed(0)


def eval_second_half(mB, mA, val, bs=32, with_state=True, full_model=None):
    if mB is not None:
        mB.eval()
    if full_model is not None:
        full_model.eval()
    tot, tok = 0.0, 0
    n = min(len(val), 1000)
    for i in range(0, n, bs):
        chunk = val[i:i + bs]
        xs, ys = [], []
        for s in chunk:
            t = s[128:256]
            xs.append(t[:-1])
            ys.append(t[1:])
        Ln = min(len(x) for x in xs)
        x = torch.stack([x[:Ln] for x in xs]).to(DEV)
        y = torch.stack([y[:Ln] for y in ys]).to(DEV)
        with torch.no_grad():
            if full_model is not None:
                xf = torch.stack([s[:255] for s in chunk]).to(DEV)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo = full_model(xf)
                lg = lo.float()[:, 128:255, :]
            elif with_state:
                st = mA.capture_states(torch.stack([s[:128] for s in chunk]).to(DEV))
                st = [s.to(torch.bfloat16) for s in st]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo = mB(x, s0_list=st)
                lg = lo.float()
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo = mB(x)
                lg = lo.float()
        l = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="sum")
        tot += l.item()
        tok += y.numel()
    return tot / tok


seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
seqs = [s for s in seqs if len(s) >= 257]
tr_end = int(0.95 * len(seqs))
va = seqs[tr_end:][:1000]
print("val:", len(va))

A = M.DeltaLoopLM().to(DEV)
A.load_state_dict(torch.load(os.path.join(HERE, "relay_A.pth"), map_location="cpu"))
A.eval()
for p in A.parameters():
    p.requires_grad_(False)

Bm = M.DeltaLoopLM().to(DEV)
Bm.load_state_dict(torch.load(os.path.join(HERE, "relay_B.pth"), map_location="cpu"))
Bm.eval()
B0 = M.DeltaLoopLM().to(DEV)
B0.load_state_dict(torch.load(os.path.join(HERE, "relay_B0.pth"), map_location="cpu"))
B0.eval()
Fm = M.DeltaLoopLM().to(DEV)
Fm.load_state_dict(torch.load(os.path.join(HERE, "relay_Full.pth"), map_location="cpu"))
Fm.eval()

n_state = eval_second_half(Bm, A, va, with_state=True)
n_none = eval_second_half(B0, None, va, with_state=False)
n_full = eval_second_half(None, None, va, full_model=Fm)
res = dict(B_with_state=n_state, B0_no_state=n_none, Full_ref=n_full,
           state_gain_vs_B0=n_none - n_state, residual_vs_Full=n_state - n_full)
print(json.dumps(res, indent=2), flush=True)
json.dump(res, open(os.path.join(HERE, "relay_min_results.json"), "w"), indent=2)
print("SAVED")
