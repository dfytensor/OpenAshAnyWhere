#!/usr/bin/env python3
"""v10: 绑定深度 + 全反传 + 39,695 步 (无 Phantom).
2x2 分解: {tied, plain} x {full-bp, phantom} @ 39,695 步
  v7  = tied   + phantom  -> 真实 3.867
  v9  = plain  + full-bp  -> 真实 2.536
  v10 = tied   + full-bp  -> ? (本次)
  (plain+phantom 不适用: 无迭代块)"""
import sys, os, time, math, json, random
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v5_30m import DEQLM30, get_batch, V, DEV, PT_CACHE

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "deqlm10.log")
OUT = os.path.join(HERE, "deqlm10_results.json")
CKPT = os.path.join(HERE, "deqlm10_pt_full.pth")
PT_STEPS = 39695
WARMUP = 1000


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


@torch.no_grad()
def masked_suffix_eval(m, val, nb=60, seed=777, bs=16):
    m.eval()
    tot_a, tok_a, tot_m, tok_m = 0.0, 0, 0.0, 0
    rng = random.Random(seed)
    for i in range(nb):
        xs = []
        for _ in range(bs):
            s = val[rng.randrange(len(val))][:256]
            xs.append(F.pad(s, (0, 256 - s.numel())))
        x = torch.stack(xs).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _, _ = m(x)
        lo = out.float()
        ys = x[:, 1:]
        l = F.cross_entropy(lo[:, :-1].reshape(-1, V), ys.reshape(-1), reduction="none").view(x.shape[0], -1)
        ls = l[:, 192:255]
        msk = x[:, 193:256] != 0
        tot_a += ls.sum().item(); tok_a += ls.numel()
        tot_m += ls[msk].sum().item(); tok_m += msk.sum().item()
    m.train()
    return round(tot_a / tok_a, 4), round(tot_m / tok_m, 4)


def main():
    torch.manual_seed(0)
    random.seed(0)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    train, val = seqs[:int(0.95 * len(seqs))], seqs[:5000]
    m = DEQLM30().to(DEV)
    log("v10: tied + 全反传, %d 步" % PT_STEPS)
    opt = optim.AdamW(m.parameters(), lr=4e-4, weight_decay=0.01, betas=(0.9, 0.95))

    def lr_at(st):
        if st < WARMUP:
            return 4e-4 * st / WARMUP
        p_ = (st - WARMUP) / max(1, PT_STEPS - WARMUP)
        return 4e-5 + 0.5 * (4e-4 - 4e-5) * (1 + math.cos(math.pi * p_))

    t0 = time.time()
    rng = random.Random(42)
    for st in range(PT_STEPS):
        for g in opt.param_groups:
            g["lr"] = lr_at(st)
        m.train()
        x, y = get_batch(train, bs=32, device=DEV, rng=rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, residual, k_used = m(x)      # 全反传 (不用 phantom)
        loss = F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1), ignore_index=0)
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if st % 2000 == 0:
            log("%d/%d loss=%.4f (%.0fms/步, ETA %.0f分)" % (st, PT_STEPS, loss.item(), (time.time() - t0) / (st + 1) * 1000, (PT_STEPS - st) * (time.time() - t0) / (st + 1) / 60))
    torch.save(m.state_dict(), CKPT)
    a, mk = masked_suffix_eval(m, val)
    log("v10 完成: 含pad=%.4f 真实=%.4f" % (a, mk))
    json.dump(dict(all=a, masked=mk), open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
