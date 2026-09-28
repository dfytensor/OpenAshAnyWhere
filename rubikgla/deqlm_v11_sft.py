#!/usr/bin/env python3
"""v11: v10 (tied+全反传 PT, 真实token 2.508 全场最优) + SFT (同 CEDLR/v8 协议, 全反传).
预期: 含pad 指标大幅改善 (v10 PT 从未学 pad), 真实 token 保住 ~2.5-2.9."""
import sys, os, time, json, random
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v5_30m import DEQLM30, V, DEV

HERE = os.path.dirname(os.path.abspath(__file__))
SFT_CACHE = r"F:\rowcol_llm\sft_cached_256.pt"
PT_VAL = r"F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt"
IN_CKPT = os.path.join(HERE, "deqlm10_pt_full.pth")
OUT_CKPT = os.path.join(HERE, "deqlm11_sft_full.pth")
LOG = os.path.join(HERE, "deqlm11.log")
OUT = os.path.join(HERE, "deqlm11_results.json")
P, Q, B = 192, 64, 32
SFT_STEPS = 28304


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
    m = DEQLM30().to(DEV)
    m.load_state_dict(torch.load(IN_CKPT, map_location=DEV, weights_only=True))
    cache = torch.load(SFT_CACHE, map_location="cpu", weights_only=True)
    G = cache["grids"]
    N_S = G.shape[0]
    val = torch.load(PT_VAL, map_location="cpu", weights_only=True)[:5000]
    log("v11 SFT: v10 PT + 全反传, %d 步" % SFT_STEPS)
    a0, mk0 = masked_suffix_eval(m, val)
    log("初始: 含pad=%.4f 真实=%.4f" % (a0, mk0))

    opt = optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.01)
    t0 = time.time()
    for st in range(SFT_STEPS):
        m.train()
        idx = torch.randint(0, N_S, (B,))
        seq = (G[idx] - 34).clamp(min=0).to(DEV).reshape(B, 256)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _, _ = m(seq)          # 全反传
        ys = seq[:, P + 1:P + Q]
        loss = F.cross_entropy(logits.float()[:, P:P + Q - 1].reshape(-1, V), ys.reshape(-1))
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if st % 2000 == 0:
            log("SFT %d/%d loss=%.4f (%.0fms/步, ETA %.0f分)" % (st, SFT_STEPS, loss.item(), (time.time() - t0) / (st + 1) * 1000, (SFT_STEPS - st) * (time.time() - t0) / (st + 1) / 60))
    torch.save(m.state_dict(), OUT_CKPT)
    a1, mk1 = masked_suffix_eval(m, val)
    log("FINAL: 含pad=%.4f 真实=%.4f (初始 %.4f/%.4f)" % (a1, mk1, a0, mk0))
    json.dump(dict(init=dict(all=a0, masked=mk0), final=dict(all=a1, masked=mk1)), open(OUT, "w"), indent=2)
    log("v11 完成")


if __name__ == "__main__":
    main()
