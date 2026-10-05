#!/usr/bin/env python3
"""v30 gate probe: eta/rate 学成了什么样 (对照 v28: eta_mean=0.53≈init, pre站rate中位0.62, 慢通道保留0.18)."""
import sys, os, random
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deqlm_v30_kv4 as M30
from deqlm_v5_30m import PT_CACHE

DEV = "cuda"
m = M30.DeltaLoopLM()
sd = torch.load(os.path.join(HERE, "deqlm30_pt_full.pth"), map_location="cpu")
m.load_state_dict(sd)
m = m.to(DEV).eval()
print("params: %.2fM" % (sum(p.numel() for p in m.parameters()) / 1e6))

seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
val = [s for s in seqs[:5000] if len(s) >= 256][:64]

eta_all, rate_all, rho_all = [], [], []
captured = {}


def mk_hook(name):
    def hook(mod, inp, out):
        h = mod.ln(inp[0])
        e = torch.sigmoid(mod.w_eta(h)).detach()
        r = torch.nn.functional.softplus(mod.w_rate(h)).detach()
        eta_all.append(e.mean().item())
        rate_all.append(r.mean().item())
        captured.setdefault(name, []).append((e.mean().item(), r.median().item()))
    return hook


hooks = []
for name, mod in [("pre", m.pre), ("loop_a", m.loop_a), ("loop_c", m.loop_c), ("post", m.post)] if hasattr(m, "pre") else []:
    pass

# 兜底: 遍历找所有 DeltaAttn
for nm, mod in m.named_modules():
    if type(mod).__name__ == "DeltaAttn":
        hooks.append(mod.register_forward_hook(mk_hook(nm)))

with torch.no_grad():
    for s in val[:16]:
        x = torch.tensor(s[:256], device=DEV).unsqueeze(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m(x)

import statistics as st
eta_v = [v for v in eta_all]
print("eta mean overall: %.4f  (v28=0.53, init 0.12)" % (sum(eta_v) / len(eta_v)))
print("rate mean overall: %.4f  floor=0.0005" % (sum(rate_all) / len(rate_all)))
for name, vals in captured.items():
    es = [a for a, b in vals]
    rs = [b for a, b in vals]
    print("%-8s eta=%.4f  rate_med=%.4f  256步保留=%.3f" % (
        name, sum(es) / len(es), st.median(rs),
        float(torch.exp(torch.tensor(-st.median(rs) * 256)))))
