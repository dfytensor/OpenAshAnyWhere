#!/usr/bin/env python3
"""② 30M 终评矩阵 (掩码口径): Meta-ASH PT/SFT, CEDLR PT/SFT 全部重测.
指标: 后缀 NLL (预测 x[193:256]) — 含pad / 仅真实token 两种口径."""
import sys, os, random, json
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
HERE = os.path.dirname(os.path.abspath(__file__))
PT_VAL = r"F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt"
CF = r"F:\OpenASH2605\copyfirst_redesign"
OUT = os.path.join(HERE, "matrix30_masked.json")
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


def eval_lm(m, name, suffix_cols):
    """suffix_cols: (lo_col_start, ys_start, ys_end) loss 列与目标切片."""
    m.eval()
    tot_a, tok_a, tot_m, tok_m = 0.0, 0, 0.0, 0
    for x in batches(val):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(x)
            if isinstance(lo, tuple):
                lo = lo[0]
            lo = lo.float()
        ys = x[:, 1:]
        l = F.cross_entropy(lo[:, :-1].reshape(-1, V), ys.reshape(-1), reduction="none").view(x.shape[0], -1)
        ls = l[:, suffix_cols[0]:suffix_cols[0] + 63]
        ysuf = x[:, 193:256]
        msk = ysuf != 0
        tot_a += ls.sum().item(); tok_a += ls.numel()
        tot_m += ls[msk].sum().item(); tok_m += msk.sum().item()
    print("%s: 含pad=%.4f | 真实=%.4f (真实占比 %.1f%%)" % (name, tot_a / tok_a, tot_m / tok_m, 100 * tok_m / tok_a), flush=True)
    return dict(all=round(tot_a / tok_a, 4), masked=round(tot_m / tok_m, 4), real_frac=round(tok_m / tok_a, 3))


if __name__ == "__main__":
    seqs = torch.load(PT_VAL, map_location="cpu", weights_only=True)
    val = seqs[:5000]
    results = {}

    # ── Meta-ASH-30M (全序列 LM, loss 列 192:255) ──
    from meta_ash_30m_recovered import CLM as MetaCLM, MetaMaxState30
    m1 = MetaCLM(MetaMaxState30).to(DEV)
    for name in ("meta_ash30_pt_full", "meta_ash30_sft_full"):
        p = os.path.join(CF, name + ".pth")
        if not os.path.exists(p):
            print("skip", name); continue
        sd = torch.load(p, map_location=DEV, weights_only=True)
        bad = sum(1 for t in sd.values() if not torch.isfinite(t.float()).all())
        m1.load_state_dict(sd)
        r = eval_lm(m1, "%s (NaN组=%d)" % (name, bad), (192, 193, 256))
        results[name] = r
    del m1
    torch.cuda.empty_cache()

    # ── CEDLR-30M (CED 前缀 LM) ──
    from bench_ced30 import CED30
    m2 = CED30().to(DEV)

    @torch.no_grad()
    def eval_ced(m, name):
        m.eval()
        tot_a, tok_a, tot_m, tok_m = 0.0, 0, 0.0, 0
        for x in batches(val):
            ys = x[:, P + 1:P + Q]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = m(x).float()
            l = F.cross_entropy(lo[:, :-1].reshape(-1, V), ys.reshape(-1), reduction="none").view(x.shape[0], -1)
            msk = ys != 0
            tot_a += l.sum().item(); tok_a += l.numel()
            tot_m += l[msk].sum().item(); tok_m += msk.sum().item()
        print("%s: 含pad=%.4f | 真实=%.4f (真实占比 %.1f%%)" % (name, tot_a / tok_a, tot_m / tok_m, 100 * tok_m / tok_a), flush=True)
        return dict(all=round(tot_a / tok_a, 4), masked=round(tot_m / tok_m, 4), real_frac=round(tok_m / tok_a, 3))

    for name in ("cedlr30_pt_full_v2", "cedlr30_sft_full_v2"):
        p = os.path.join(CF, name + ".pth")
        sd = torch.load(p, map_location=DEV, weights_only=True)
        m2.load_state_dict(sd)
        results[name] = eval_ced(m2, name)
    del m2
    torch.cuda.empty_cache()

    json.dump(results, open(OUT, "w"), indent=2, ensure_ascii=False)
    print("saved", OUT)
