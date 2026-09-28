#!/usr/bin/env python3
"""v8 (SFT 后) 掩码评测: 含pad / 仅真实token 后缀 NLL, 完成终评矩阵."""
import sys, os, random, json
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
PT_VAL = r"F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt"
OUT = os.path.join(HERE, "matrix30_final.json")
V, DEV = 23005, "cuda"
P, Q = 192, 64


@torch.no_grad()
def batches(val, nb=60, seed=777, bs=16):
    rng = random.Random(seed)
    for i in range(nb):
        xs = []
        for _ in range(bs):
            s = val[rng.randrange(len(val))][:256]
            xs.append(F.pad(s, (0, 256 - s.numel())))
        yield torch.stack(xs).to(DEV)


if __name__ == "__main__":
    from deqlm_v5_30m import DEQLM30
    seqs = torch.load(PT_VAL, map_location="cpu", weights_only=True)
    val = seqs[:5000]
    m = DEQLM30().to(DEV)
    m.load_state_dict(torch.load(os.path.join(HERE, "deqlm8_sft_full.pth"), map_location=DEV, weights_only=True))
    m.eval()
    tot_a, tok_a, tot_m, tok_m = 0.0, 0, 0.0, 0
    for x in batches(val):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _, _ = m(x)
        lo = out.float()
        ys = x[:, 1:]
        l = F.cross_entropy(lo[:, :-1].reshape(-1, V), ys.reshape(-1), reduction="none").view(x.shape[0], -1)
        ls = l[:, 192:255]
        ysuf = x[:, 193:256]
        msk = ysuf != 0
        tot_a += ls.sum().item(); tok_a += ls.numel()
        tot_m += ls[msk].sum().item(); tok_m += msk.sum().item()
    print("DEQ-LM v8 (SFT): 含pad=%.4f | 真实=%.4f" % (tot_a / tok_a, tot_m / tok_m), flush=True)
    results = json.load(open(os.path.join(HERE, "matrix30_masked.json"), encoding="utf-8"))
    results["deqlm_v8_sft"] = dict(all=round(tot_a / tok_a, 4), masked=round(tot_m / tok_m, 4), real_frac=round(tok_m / tok_a, 3))
    json.dump(results, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("saved", OUT)
