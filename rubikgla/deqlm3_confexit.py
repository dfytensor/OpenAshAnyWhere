#!/usr/bin/env python3
"""修正版置信度早退评测: 锁定时捕获该位置的 logits (v3 训练结果不变)."""
import sys, os, json, random
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v3 import DEQLM, get_batch, V, DEV, KMAX, SEQ, PT_CACHE

HERE = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0)
random.seed(0)
seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
val = seqs[int(0.95 * len(seqs)):]
m = DEQLM().to(DEV)
m.load_state_dict(torch.load(os.path.join(HERE, "deqlm3_final.pth"), map_location=DEV, weights_only=True))
m.eval()

@torch.no_grad()
def conf_exit(m, taus, nb=20, seed=999):
    res = {t: [0.0, 0.0, 0] for t in taus}
    for i in range(nb):
        x, y = get_batch(val, bs=32, device=DEV, rng=random.Random(seed + i))
        h = m.bottom_h(x)
        Bc, L = x.shape
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for t in taus:
                locked = torch.zeros(Bc, L, dtype=torch.bool, device=DEV)
                final_lg = None
                k_used = torch.full((Bc, L), float(KMAX), device=DEV)
                zz = h.clone()
                for k in range(1, KMAX + 1):
                    zz = zz + m.deq.alpha() * m.deq._block_fn(zz)
                    hh = zz
                    for layer in m.top:
                        hh = layer(hh)
                    lg = m.head(m.ln_f(hh)).float()
                    if final_lg is None:
                        final_lg = lg.clone()
                    mp = torch.softmax(lg, -1).max(-1).values
                    newly = (~locked) & (mp > t)
                    if newly.any():
                        final_lg[newly] = lg[newly]
                        k_used[newly] = k
                        locked |= newly
                    if locked.all():
                        break
                lsum = F.cross_entropy(final_lg.reshape(-1, V), y.reshape(-1), reduction="sum").item()
                res[t][0] += lsum
                res[t][1] += k_used.sum().item()
                res[t][2] += y.numel()
    return {t: (round(v[0] / v[2], 4), round(v[1] / v[2], 2)) for t, v in res.items()}

taus = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
out = conf_exit(m, taus, nb=20)
print("tau  -> NLL, avg_k (kappa=1.0 全部跑满 KMAX=30: NLL=4.3599, k=30)")
full_nll = 0.0
for t in taus:
    nll, avgk = out[t]
    print("tau=%.2f: NLL=%.4f avg_k=%.2f" % (t, nll, avgk))
json.dump({str(t): out[t] for t in taus}, open(os.path.join(HERE, "deqlm3_confexit.json"), "w"), indent=2)
print("saved deqlm3_confexit.json")
