#!/usr/bin/env python3
"""终评: DEQ-LM v7 (23M, Phantom, 39695步) vs CEDLR-Hybrid2-30M PT, 同口径后缀 NLL (预测 x[193:256])."""
import sys, os, random
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
HERE = os.path.dirname(os.path.abspath(__file__))
PT_VAL = r"F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt"
CF = r"F:\OpenASH2605\copyfirst_redesign"
OUT = os.path.join(HERE, "deqlm7_vs_cedlr.json")


@torch.no_grad()
def batches(val, nb=30, seed=777, bs=32):
    rng = random.Random(seed)
    for i in range(nb):
        xs = []
        for _ in range(bs):
            s = val[rng.randrange(len(val))][:256]
            xs.append(F.pad(s, (0, 256 - s.numel())))
        yield torch.stack(xs).to(DEV)


if __name__ == "__main__":
    import json
    V, DEV = 23005, "cuda"
    seqs = torch.load(PT_VAL, map_location="cpu", weights_only=True)
    val = seqs[:5000]
    results = {}

    # ── CEDLR-Hybrid2-30M PT ──
    from bench_ced30 import CED30, P, Q
    m2 = CED30().to(DEV)
    sd = torch.load(os.path.join(CF, "cedlr30_pt_full_v2.pth"), map_location=DEV, weights_only=True)
    m2.load_state_dict(sd)
    m2.eval()
    tot, tok = 0.0, 0
    for x in batches(val):
        ys = x[:, P + 1:P + Q]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m2(x)
        lo = lo.float()
        l = F.cross_entropy(lo[:, :-1].reshape(-1, V), ys.reshape(-1), reduction="none").view(x.shape[0], -1)
        tot += l.sum().item(); tok += l.numel()
    results["cedlr30_pt"] = round(tot / tok, 4)
    print("CEDLR-Hybrid2-30M PT: suffix NLL = %.4f" % (tot / tok), flush=True)
    del m2
    torch.cuda.empty_cache()

    # ── DEQ-LM v7 ──
    from deqlm_v5_30m import DEQLM30
    m = DEQLM30().to(DEV)
    m.load_state_dict(torch.load(os.path.join(HERE, "deqlm7_full.pth"), map_location=DEV, weights_only=True))
    m.eval()
    for k in [1, 8, 16, 30]:
        tot, tok = 0.0, 0
        for x in batches(val):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = m.bottom_h(x)
                alpha = m.deq.alpha()
                z = h
                for kk in range(k):
                    z = z + alpha * m.deq._block_fn(z)
                hh = z
                for layer in m.top:
                    hh = layer(hh)
                lo = m.head(m.ln_f(hh)).float()
            ys = x[:, 1:]
            l = F.cross_entropy(lo[:, :-1].reshape(-1, V), ys.reshape(-1), reduction="none").view(x.shape[0], -1)
            l = l[:, 192:255]          # 预测 x[193:256], 与 CED 对齐
            tot += l.sum().item(); tok += l.numel()
        results["deqlm7_k%d" % k] = round(tot / tok, 4)
        print("DEQ-LM v7 k=%2d: suffix NLL = %.4f" % (k, tot / tok), flush=True)

    json.dump(results, open(OUT, "w"), indent=2)
    print("saved", OUT)
