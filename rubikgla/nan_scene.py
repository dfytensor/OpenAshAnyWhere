#!/usr/bin/env python3
"""NaN 第一现场插桩 (后台版): 逐层 hook + 梯度检查, 记录第一个出问题的算子."""
import sys, os, torch, random
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_ced30 import CED30, V, DEV, P, Q

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nan_scene.log")
SFT = r"F:\rowcol_llm\sft_cached_256.pt"
PT_CKPT = r"F:\OpenASH2605\copyfirst_redesign\cedlr30_pt_step16000.pth"

def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def main():
    random.seed(0)
    torch.manual_seed(0)
    cache = torch.load(SFT, map_location="cpu", weights_only=True)
    G, T = cache["grids"], cache["tgts"]
    NS = G.shape[0]
    m = CED30().to(DEV)
    m.load_state_dict(torch.load(PT_CKPT, map_location=DEV, weights_only=True))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.01)
    events = []

    def mk_hook(name):
        def hook(mod, inp, out):
            if events:
                return
            o = out[0] if isinstance(out, tuple) else out
            if not torch.isfinite(o.float()).all():
                events.append((name, "out non-finite"))
        return hook

    for i, l in enumerate(m.enc):
        l.register_forward_hook(mk_hook(f"enc[{i}].{type(l).__name__}"))
    for i, l in enumerate(m.dec):
        l.register_forward_hook(mk_hook(f"dec[{i}].{type(l).__name__}"))
    for i, l in enumerate(m.dec_kv):
        l.register_forward_hook(mk_hook(f"dec_kv[{i}]"))

    for st in range(4000):
        idx = torch.randint(0, NS, (32,))
        seq = (G[idx] - 34).clamp(min=0).to(DEV).reshape(32, 256)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(seq)
        lo = lo.float()
        if not torch.isfinite(lo).all():
            events.append(("logits", "non-finite"))
        ys = seq[:, 193:]
        loss = F.cross_entropy(lo[:, :-1].reshape(-1, V), ys.reshape(-1))
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0).item()
        bad_g = [n for n, p in m.named_parameters()
                 if p.grad is not None and not torch.isfinite(p.grad).all()]
        if (gn != gn or bad_g) and len(events) < 3:
            events.append(("grads", "nan=%s gn=%.2e" % (bad_g[:4], gn)))
        if events:
            log("第一现场 st%d: %s" % (st, events))
            for e in events:
                log("  %s %s" % e)
            break
        opt.step()
        if st % 200 == 0:
            log("st%d loss=%.3f gn=%.2f" % (st, loss.item(), gn))
    else:
        log("4000 步无事故 (该 seed 下未复现; 需换 seed 或更长程)")
    log("END")

if __name__ == "__main__":
    main()
