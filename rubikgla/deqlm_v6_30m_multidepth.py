#!/usr/bin/env python3
"""DEQ-LM v6: 23M 多深度监督 — 验证 v4 的'权衡定律'是否随规模保持.
v4 (d128): 多深度监督后 k=1 即达 4.40 (深度价值坍缩), 早退免费.
v6 (d320): 如果同样坍缩 → 定律跨规模成立; 如果 k=1 仍明显差 → 规模保留深度价值."""
import sys, os, time, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v5_30m import DEQLM30, get_batch, V, DEV, SEQ, KMAX, PT_CACHE

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "deqlm6.log")
OUT = os.path.join(HERE, "deqlm6_results.json")
CKPT = os.path.join(HERE, "deqlm6_final.pth")
SUPERVISED_KS = [1, 2, 4, 8, 16, 30]
STEPS = 3000

def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def forward_multidepth(m, x):
    h = m.bottom_h(x)
    alpha = m.deq.alpha()
    out = {}
    z = h
    for k in range(1, KMAX + 1):
        z = z + alpha * m.deq._block_fn(z)
        if k in SUPERVISED_KS:
            hh = z
            for layer in m.top:
                hh = layer(hh)
            out[k] = m.head(m.ln_f(hh))
    return out


def main():
    torch.manual_seed(0)
    random.seed(0)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    train, val = seqs[:int(0.95 * len(seqs))], seqs[int(0.95 * len(seqs)):]
    m = DEQLM30().to(DEV)
    log("v6 (d320 多深度) 参数: %.2fM" % (sum(p.numel() for p in m.parameters()) / 1e6))
    opt = optim.AdamW(m.parameters(), lr=4e-4, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=4e-5)

    t0 = time.time()
    rng = random.Random(42)
    for st in range(STEPS):
        m.train()
        x, y = get_batch(train, bs=32, device=DEV, rng=rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs = forward_multidepth(m, x)
            losses = [F.cross_entropy(outs[k].float().reshape(-1, V), y.reshape(-1), ignore_index=0)
                      for k in SUPERVISED_KS]
            loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            log("st%d non-finite -> skip" % st)
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if st % 200 == 0:
            ks_str = " ".join("%.2f" % l.item() for l in losses)
            log("st%d mean=%.4f [%s] (%.0fms/step)" % (st, loss.item(), ks_str, (time.time()-t0)/(st+1)*1000))

    m.eval()
    torch.save(m.state_dict(), CKPT)

    log("\n=== 各深度独立 val NLL (v6 d320 多深度) ===")
    results = dict(nll_by_k={}, staircase_exit={})
    with torch.no_grad():
        for k in SUPERVISED_KS:
            tot, tok = 0.0, 0
            for i in range(20):
                x, y = get_batch(val, bs=32, device=DEV, rng=random.Random(999 + i))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outs = forward_multidepth(m, x)
                l = F.cross_entropy(outs[k].float().reshape(-1, V), y.reshape(-1), reduction="sum")
                tot += l.item(); tok += y.numel()
            results["nll_by_k"][str(k)] = round(tot / tok, 4)
            log("  k=%2d: NLL=%.4f" % (k, tot / tok))

    log("\n=== 阶梯置信度早退 ===")
    with torch.no_grad():
        for tau in [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]:
            tot_l, tot_k, tot_tok = 0.0, 0.0, 0
            for i in range(20):
                x, y = get_batch(val, bs=32, device=DEV, rng=random.Random(999 + i))
                Bc, L = x.shape
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outs = forward_multidepth(m, x)
                locked = torch.zeros(Bc, L, dtype=torch.bool, device=DEV)
                final_lg = None
                k_used = torch.full((Bc, L), float(KMAX), device=DEV)
                for k in SUPERVISED_KS:
                    lg = outs[k].float()
                    if final_lg is None:
                        final_lg = lg.clone()
                    mp = torch.softmax(lg, -1).max(-1).values
                    newly = (~locked) & (mp > tau)
                    if newly.any():
                        final_lg[newly] = lg[newly]
                        k_used[newly] = float(k)
                        locked |= newly
                    if locked.all():
                        break
                lsum = F.cross_entropy(final_lg.reshape(-1, V), y.reshape(-1), reduction="sum").item()
                tot_l += lsum; tot_k += k_used.sum().item(); tot_tok += y.numel()
            nll, avgk = tot_l / tot_tok, tot_k / tot_tok
            results["staircase_exit"][str(tau)] = [round(nll, 4), round(avgk, 2)]
            log("  tau=%.2f: NLL=%.4f avg_k=%.2f" % (tau, nll, avgk))

    json.dump(results, open(OUT, "w"), indent=2)
    log("结果已存 " + OUT)


if __name__ == "__main__":
    main()
