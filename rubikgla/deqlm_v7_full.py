#!/usr/bin/env python3
"""DEQ-LM v7: 23M + Phantom Gradient + 全量 PT (39,695 步, 对齐 CEDLR-Hybrid2-30M 训练量).
单深度训练 (k=30), 阶段性存 checkpoint, 末段评估: 全序列 NLL + 后缀32 NLL (CEDLR 协议参照) + k 扫描."""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v5_30m import DEQLM30, get_batch, V, DEV, SEQ, KMAX, PT_CACHE

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "deqlm7.log")
OUT = os.path.join(HERE, "deqlm7_results.json")
CKPT = os.path.join(HERE, "deqlm7_full.pth")
STEPS = 39695
WARMUP = 1000


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def phantom_forward(m, x, kmax=KMAX):
    """前向 no_grad 迭代 kmax 次, 反向穿 1 步图 (省显存)."""
    h = m.bottom_h(x)
    alpha = m.deq.alpha()
    with torch.no_grad():
        z = h
        res = 0.0
        for _ in range(kmax):
            z_new = z + alpha * m.deq._block_fn(z)
            z = z_new
        z_star = z
    z1 = h + alpha * m.deq._block_fn(h)
    z = z_star + (z1 - z1.detach())
    for layer in m.top:
        z = layer(z)
    return m.head(m.ln_f(z))


def main():
    torch.manual_seed(0)
    random.seed(0)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    train, val = seqs[:int(0.95 * len(seqs))], seqs[int(0.95 * len(seqs)):]
    log("train %d val %d, 目标 %d 步" % (len(train), len(val), STEPS))
    m = DEQLM30().to(DEV)
    log("v7 参数: %.2fM (Phantom 反向)" % (sum(p.numel() for p in m.parameters()) / 1e6))
    opt = optim.AdamW(m.parameters(), lr=4e-4, weight_decay=0.01, betas=(0.9, 0.95))

    def lr_at(st):
        if st < WARMUP:
            return 4e-4 * st / WARMUP
        p = (st - WARMUP) / max(1, STEPS - WARMUP)
        return 4e-5 + 0.5 * (4e-4 - 4e-5) * (1 + math.cos(math.pi * p))

    t0 = time.time()
    rng = random.Random(42)
    skip = 0
    for st in range(STEPS):
        for g in opt.param_groups:
            g["lr"] = lr_at(st)
        m.train()
        x, y = get_batch(train, bs=32, device=DEV, rng=rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = phantom_forward(m, x)
        loss = F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1), ignore_index=0)
        if not torch.isfinite(loss):
            skip += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if st % 500 == 0:
            el = time.time() - t0
            eta_min = (STEPS - st) * el / max(1, st) / 60
            log("st%d/%d loss=%.4f gn=%.2f (%.0fms/步, ETA %.0f分)" % (st, STEPS, loss.item(), gn, el / max(1, st) * 1000, eta_min))
        if (st + 1) % 10000 == 0 or st == STEPS - 1:
            torch.save(m.state_dict(), CKPT)
    log("完成, 跳过 %d 步" % skip)
    torch.save(m.state_dict(), CKPT)

    # ═══ 评估 ═══
    m.eval()
    log("\n=== k 扫描 (全序列 256 token NLL) ===")
    results = dict(nll_by_k={}, suffix_nll_by_k={})
    with torch.no_grad():
        for k in [1, 2, 4, 8, 16, 30]:
            tot, tok, stot, stok = 0.0, 0, 0.0, 0
            for i in range(20):
                x, y = get_batch(val, bs=32, device=DEV, rng=random.Random(999 + i))
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
                l_all = F.cross_entropy(lo.reshape(-1, V), y.reshape(-1), reduction="none").reshape(y.shape)
                mask = y != 0
                tot += l_all[mask].sum().item(); tok += mask.sum().item()
                # 后缀 32 (CEDLR 协议参照: 给前缀, 预测最后 32 个)
                lsuf = l_all[:, -32:]
                msuf = mask[:, -32:]
                stot += lsuf[msuf].sum().item(); stok += msuf.sum().item()
            results["nll_by_k"][str(k)] = round(tot / tok, 4)
            results["suffix_nll_by_k"][str(k)] = round(stot / stok, 4)
            log("  k=%2d: 全序列 NLL=%.4f | 后缀32 NLL=%.4f" % (k, tot / tok, stot / stok))

    # 残差探针
    import numpy as np
    residuals, nlls = [], []
    with torch.no_grad():
        for i in range(100):
            x, y = get_batch(val, bs=1, device=DEV, rng=random.Random(7000 + i))
            out, res, _ = m(x)
            mask = y != 0
            if mask.sum() < 8:
                continue
            l = F.cross_entropy(out.float().reshape(-1, V), y.reshape(-1), reduction="none").reshape(y.shape)
            nlls.append(l[mask].mean().item()); residuals.append(res)
    r, n_ = np.array(residuals), np.array(nlls)
    corr = float(np.corrcoef(r, n_)[0, 1]) if r.std() > 0 else 0.0
    log("残差-NLL 相关: %.3f" % corr)
    results["probe"] = dict(corr=round(corr, 3))
    json.dump(results, open(OUT, "w"), indent=2)
    log("结果已存 " + OUT)


if __name__ == "__main__":
    main()
