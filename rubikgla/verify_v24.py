#!/usr/bin/env python3
"""高精度复评: v20 vs v24, 200 批同种子, 确认 0.004 差距是否真实."""
import sys, os, random
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v19_tuned23m import masked_suffix_eval
from deqlm_v24_hybrid2fast import Hybrid2FastLM
from deqlm_v20_tunedloop import TunedTriLoopLM

HERE = os.path.dirname(os.path.abspath(__file__))
PT_VAL = r"F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt"
V, DEV = 23005, "cuda"


@torch.no_grad()
def eval_big(m, val, nb=200, seed=777, bs=16):
    m.eval()
    tot, tok = 0.0, 0
    rng = random.Random(seed)
    for i in range(nb):
        xs = []
        for _ in range(bs):
            s = val[rng.randrange(len(val))][:256]
            xs.append(F.pad(s, (0, 256 - s.numel())))
        x = torch.stack(xs).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(x).float()
        ys = x[:, 1:]
        l = F.cross_entropy(lo[:, :-1].reshape(-1, V), ys.reshape(-1), reduction="none").view(x.shape[0], -1)
        ls = l[:, 192:255]
        msk = x[:, 193:256] != 0
        tot += ls[msk].sum().item()
        tok += msk.sum().item()
    return tot / tok, tot, tok


if __name__ == "__main__":
    seqs = torch.load(PT_VAL, map_location="cpu", weights_only=True)
    val = seqs[:5000]
    m20 = TunedTriLoopLM().to(DEV)
    m20.load_state_dict(torch.load(os.path.join(HERE, "deqlm20_pt_full.pth"), map_location=DEV, weights_only=True))
    n20, t20, k20 = eval_big(m20, val)
    print("v20:  NLL = %.4f  (tokens=%d)" % (n20, k20), flush=True)
    del m20
    torch.cuda.empty_cache()
    m24 = Hybrid2FastLM().to(DEV)
    m24.load_state_dict(torch.load(os.path.join(HERE, "deqlm24_pt_full.pth"), map_location=DEV, weights_only=True))
    n24, t24, k24 = eval_big(m24, val)
    print("v24:  NLL = %.4f  (tokens=%d)" % (n24, k24), flush=True)
    print("差距 v24-v20 = %+.4f  (%s)" % (n24 - n20, "v24 胜" if n24 < n20 else "v20 胜"))
    # binomial-style粗略显著性: 每 token NLL 独立性近似, 用每批均值做样本
