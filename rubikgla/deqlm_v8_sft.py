#!/usr/bin/env python3
"""DEQ-LM v8: v7 (39695步 PT) + SFT, 完全对齐 CEDLR SFT 协议.
数据: sft_cached_256.pt 网格展平, (G-34).clamp, loss 仅后缀列 (含 pad), 28304 步, lr 1e-4."""
import sys, os, time, math, json, random
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v5_30m import DEQLM30, V, DEV, KMAX

HERE = os.path.dirname(os.path.abspath(__file__))
SFT_CACHE = r"F:\rowcol_llm\sft_cached_256.pt"
PT_VAL = r"F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt"
IN_CKPT = os.path.join(HERE, "deqlm7_full.pth")
OUT_CKPT = os.path.join(HERE, "deqlm8_sft_full.pth")
LOG = os.path.join(HERE, "deqlm8.log")
OUT = os.path.join(HERE, "deqlm8_results.json")
P, Q, B = 192, 64, 32


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def phantom_forward(m, x, kmax=KMAX):
    h = m.bottom_h(x)
    alpha = m.deq.alpha()
    with torch.no_grad():
        z = h
        for _ in range(kmax):
            z = z + alpha * m.deq._block_fn(z)
        z_star = z
    z1 = h + alpha * m.deq._block_fn(h)
    z = z_star + (z1 - z1.detach())
    for layer in m.top:
        z = layer(z)
    return m.head(m.ln_f(z))


@torch.no_grad()
def suffix_nll(m, val, nb=20, seed=777):
    m.eval()
    tot, tok = 0.0, 0
    rng = random.Random(seed)
    for i in range(nb):
        xs = []
        for _ in range(32):
            s = val[rng.randrange(len(val))][:256]
            xs.append(F.pad(s, (0, 256 - s.numel())))
        x = torch.stack(xs).to(DEV)
        ys = x[:, P + 1:P + Q]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(x)[0]
        lo = lo.float()
        l = F.cross_entropy(lo[:, P:P + Q - 1].reshape(-1, V), ys.reshape(-1), reduction="sum")
        tot += l.item(); tok += ys.numel()
    m.train()
    return tot / tok


def main():
    torch.manual_seed(0)
    random.seed(0)
    m = DEQLM30().to(DEV)
    m.load_state_dict(torch.load(IN_CKPT, map_location=DEV, weights_only=True))
    cache = torch.load(SFT_CACHE, map_location="cpu", weights_only=True)
    G = cache["grids"]
    N_S = G.shape[0]
    STEPS = (N_S + B - 1) // B
    val = torch.load(PT_VAL, map_location="cpu", weights_only=True)[:5000]
    log("v8 SFT: %d 步 (1 epoch), 对齐 CEDLR 协议" % STEPS)
    v0 = suffix_nll(m, val)
    log("初始 suffix NLL (含pad) = %.4f" % v0)

    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.01)
    backup = {k: v.clone() for k, v in m.state_dict().items()}
    t0 = time.time()
    nf = 0
    for st in range(STEPS):
        m.train()
        idx = torch.randint(0, N_S, (B,))
        seq = (G[idx] - 34).clamp(min=0).to(DEV).reshape(B, 256)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = phantom_forward(m, seq)
        logits = logits.float()
        ys = seq[:, P + 1:P + Q]
        loss = F.cross_entropy(logits[:, P:P + Q - 1].reshape(-1, V), ys.reshape(-1))
        if not torch.isfinite(loss):
            nf += 1
            m.load_state_dict(backup)
            log("  st%d non-finite -> 回滚 (%d)" % (st, nf))
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if st % 100 == 0:
            backup = {k: v.clone() for k, v in m.state_dict().items()}
        if st % 1000 == 0:
            el = time.time() - t0
            log("  %d/%d loss=%.4f (%.0fms/步, ETA %.0f分)" % (st, STEPS, loss.item(), el / (st + 1) * 1000, (STEPS - st) * el / (st + 1) / 60))
        if (st + 1) % 10000 == 0:
            torch.save(m.state_dict(), os.path.join(HERE, "deqlm8_step%d.pth" % (st + 1)))

    torch.save(m.state_dict(), OUT_CKPT)
    v1 = suffix_nll(m, val)
    log("FINAL suffix NLL (含pad) = %.4f (初始 %.4f)" % (v1, v0))
    json.dump(dict(init_suffix_nll=round(v0, 4), final_suffix_nll=round(v1, 4), steps=STEPS, nonfinite=nf),
              open(OUT, "w"), indent=2)
    log("v8 完成")


if __name__ == "__main__":
    main()

