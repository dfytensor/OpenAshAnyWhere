#!/usr/bin/env python3
"""30M 终评 v2: Meta-ASH-30M vs CEDLR-Hybrid2-30M, 同一口径 suffix NLL (x[193:256])."""
import sys, os, torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
from bench_ced30 import CED30, V, DEV, P, Q
import random

PT_VAL = r"F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt"
CF = r"F:\OpenASH2605\copyfirst_redesign"


@torch.no_grad()
def suffix_nll_meta(m, val, seed=777, nb=30):
    """Meta 全序列 LM: 全位置 CE, 取后缀列 (预测 x[193:256] -> loss 列 192:255)."""
    m.eval()
    tot, tok = 0.0, 0
    rng = random.Random(seed)
    for i in range(nb):
        xs = []
        for _ in range(32):
            s = val[rng.randrange(len(val))][:256]
            xs.append(F.pad(s, (0, 256 - s.numel())))
        x = torch.stack(xs).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(x)
        lo = out[0] if isinstance(out, tuple) else out
        lo = lo.float()
        ys = x[:, 1:]
        l = F.cross_entropy(lo[:, :-1].reshape(-1, V), ys.reshape(-1),
                            reduction="none").view(x.shape[0], -1)
        l = l[:, P - 1:]                                    # 后缀列 (预测 193..255)
        tot += l.sum().item(); tok += l.numel()
    return tot / tok


@torch.no_grad()
def suffix_nll_ced(m, val, seed=777, nb=30):
    """CED 前缀 LM: logits[:, :-1] vs x[193:256]."""
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
            lo = m(x)
        lo = lo.float()
        l = F.cross_entropy(lo[:, :-1].reshape(-1, V), ys.reshape(-1), reduction="sum")
        tot += l.item(); tok += ys.numel()
    return tot / tok


def main():
    torch.manual_seed(0)
    random.seed(0)
    seqs = torch.load(PT_VAL, map_location="cpu", weights_only=True)
    val = seqs[:5000]

    print("=== Meta-ASH-30M ===", flush=True)
    from meta_ash_30m_recovered import CLM as MetaCLM, MetaMaxState30
    m1 = MetaCLM(MetaMaxState30).to(DEV)
    for name in ("meta_ash30_sft_full", "meta_ash30_pt_full"):
        p = os.path.join(CF, name + ".pth")
        sd = torch.load(p, map_location=DEV, weights_only=True)
        bad = sum(1 for t in sd.values() if not torch.isfinite(t.float()).all())
        m1.load_state_dict(sd)
        nll = suffix_nll_meta(m1, val)
        print("%s: NaN组=%d suffix NLL=%.4f" % (name, bad, nll), flush=True)
    del m1
    torch.cuda.empty_cache()

    print("=== CEDLR-Hybrid2-30M ===", flush=True)
    from bench_ced30 import CED30
    m2 = CED30().to(DEV)
    for name in ("cedlr30_sft_full_v2", "cedlr30_pt_full_v2"):
        p = os.path.join(CF, name + ".pth")
        sd = torch.load(p, map_location=DEV, weights_only=True)
        m2.load_state_dict(sd)
        nll = suffix_nll_ced(m2, val)
        print("%s: NaN组=%d suffix NLL=%.4f" % (name, bad, nll), flush=True)
    del m2
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
