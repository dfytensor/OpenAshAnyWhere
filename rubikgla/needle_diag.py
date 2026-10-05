#!/usr/bin/env python3
"""v28 检索归零的零成本前向诊断 (不重训): 同一 ckpt, 换扫描/门控变体跑针测验.
(a) baseline   = 训练原样 (bf16 扫描 + k 归一化)
(b) fp32       = 扫描改 fp32 (慢速通道 bf16 舍入噪声假说)
(c) nonorm     = 去掉 F.normalize(k) (键幅值信息假说)
(d) both       = fp32 + nonorm
(e) eta012     = η 门强制 0.12 (训练学到的过强覆写假说)
(f) slow10     = 速率门 ×0.1 (衰减过快假说)
对照: v24 原测 (needle_results.json), 随机 mean_rank=11503."""
import sys, os, json, math, random
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deqlm_v28_deltaloop as M28
from deqlm_v5_30m import PT_CACHE
from needle_eval import build_trials, evaluate, DEPTH, N_FACT, N_CTRL

DEV = "cuda"

from scan import affine_scan


def fp32_scan(k, v, rate, eta, q):
    B, L, KV, dh = k.shape
    H_ = q.shape[2]
    kn = F.normalize(k.float(), dim=-1)
    lam = torch.exp(-rate.float())
    I = torch.eye(dh, device=k.device, dtype=torch.float32).view(1, 1, 1, dh, dh)
    A = lam.unsqueeze(-1) * (I - eta.float().unsqueeze(-1) * (kn.unsqueeze(-1) * kn.unsqueeze(-2)))
    W = kn.unsqueeze(-1) * v.float().unsqueeze(-2)
    _, S = affine_scan(A, W)                                     # 全 fp32
    qg = q.float().view(B, L, KV, H_ // KV, dh)
    return torch.einsum("blgpq,blgmq->blgmp", S, qg).reshape(B, L, H_ * dh)


def nonorm_scan(k, v, rate, eta, q):
    B, L, KV, dh = k.shape
    H_ = q.shape[2]
    kn = k.float()                                               # 无归一化
    lam = torch.exp(-rate.float())
    I = torch.eye(dh, device=k.device, dtype=torch.float32).view(1, 1, 1, dh, dh)
    A = lam.unsqueeze(-1) * (I - eta.float().unsqueeze(-1) * (kn.unsqueeze(-1) * kn.unsqueeze(-2)))
    W = kn.unsqueeze(-1) * v.float().unsqueeze(-2)
    _, S = affine_scan(A.to(torch.bfloat16), W.to(torch.bfloat16))
    qg = q.float().view(B, L, KV, H_ // KV, dh)
    return torch.einsum("blgpq,blgmq->blgmp", S.float(), qg).reshape(B, L, H_ * dh)


def both_scan(k, v, rate, eta, q):
    B, L, KV, dh = k.shape
    H_ = q.shape[2]
    kn = k.float()
    lam = torch.exp(-rate.float())
    I = torch.eye(dh, device=k.device, dtype=torch.float32).view(1, 1, 1, dh, dh)
    A = lam.unsqueeze(-1) * (I - eta.float().unsqueeze(-1) * (kn.unsqueeze(-1) * kn.unsqueeze(-2)))
    W = kn.unsqueeze(-1) * v.float().unsqueeze(-2)
    _, S = affine_scan(A, W)
    qg = q.float().view(B, L, KV, H_ // KV, dh)
    return torch.einsum("blgpq,blgmq->blgmp", S, qg).reshape(B, L, H_ * dh)


def collect_sites(m):
    sites = [m.pre.attn, m.post.attn, m.loop_a.attn.attn, m.loop_c.attn.attn]
    return sites


def main():
    torch.manual_seed(0)
    rng = random.Random(42)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    val = [s for s in seqs[:5000] if len(s) >= 256]
    xs, metas = build_trials(val, rng)

    m = M28.DeltaLoopLM()
    m.load_state_dict(torch.load(os.path.join(HERE, "deqlm28_pt_full.pth"), map_location="cpu"))
    m = m.to(DEV).eval()
    sites = collect_sites(m)

    base = M28.DELTA_FN

    def run(name, scan_fn=None, ctx=None):
        if scan_fn is not None:
            M28.DELTA_FN = scan_fn
        if ctx:
            ctx.__enter__()
        with torch.no_grad():
            ranks = evaluate(m, xs, metas)
        if ctx:
            ctx.__exit__(None, None, None)
        M28.DELTA_FN = base
        fact = [r for r, mt in zip(ranks, metas) if mt["fact"]]
        ctrl = [r for r, mt in zip(ranks, metas) if not mt["fact"]]
        acc1 = sum(1 for x in fact if x == 1) / len(fact)
        acc5 = sum(1 for x in fact if x <= 5) / len(fact)
        mrr = sum(1.0 / x for x in fact) / len(fact)
        mr = sum(fact) / len(fact)
        c1 = sum(1 for x in ctrl if x == 1) / len(ctrl)
        print("%-12s acc1=%5.2f%% acc5=%5.2f%% mrr=%.4f meanrank=%6.1f ctrl1=%.2f%%"
              % (name, acc1 * 100, acc5 * 100, mrr, mr, c1 * 100))
        return dict(acc1=acc1, acc5=acc5, mrr=mrr, mean_rank=mr, ctrl1=c1)

    # η 强制 0.12 的上下文: 临时把 w_eta 置常数
    class EtaCtx:
        def __enter__(self):
            self.snap = []
            for s in sites:
                self.snap.append((s.w_eta.weight.detach().clone(), s.w_eta.bias.detach().clone()))
                s.w_eta.weight.data.zero_()
                s.w_eta.bias.data.fill_(math.log(0.12 / 0.88))
        def __exit__(self, *a):
            for s, (w, b) in zip(sites, self.snap):
                s.w_eta.weight.data.copy_(w)
                s.w_eta.bias.data.copy_(b)

    # 速率 ×0.1: softplus(b')=softplus(b)/10
    class SlowCtx:
        def __enter__(self):
            self.snap = []
            for s in sites:
                self.snap.append(s.w_rate.bias.detach().clone())
                r = F.softplus(s.w_rate.bias.data) / 10.0
                s.w_rate.bias.data.copy_(torch.log(torch.expm1(r.clamp(min=1e-8))))
        def __exit__(self, *a):
            for s, snap in zip(sites, self.snap):
                s.w_rate.bias.data.copy_(snap)

    res = {}
    res["a_baseline"] = run("a baseline", M28.DELTA_FN)
    res["b_fp32"] = run("b fp32", fp32_scan)
    res["c_nonorm"] = run("c nonorm", nonorm_scan)
    res["d_both"] = run("d both", both_scan)
    res["e_eta012"] = run("e eta012", None, EtaCtx())
    res["f_slow10"] = run("f slow10", None, SlowCtx())
    M28.DELTA_FN = base

    ref = json.load(open(os.path.join(HERE, "needle_results.json")))
    print("\nref v24: acc1=%.2f%% mrr=%.4f meanrank=%.1f" %
          (sum(ref["v24"]["fact_d%d" % d]["acc1"] for d in DEPTH) / len(DEPTH) * 100,
           sum(ref["v24"]["fact_d%d" % d]["mrr"] for d in DEPTH) / len(DEPTH),
           sum(ref["v24"]["fact_d%d" % d]["mean_rank"] for d in DEPTH) / len(DEPTH)))
    print("ref v28(a): acc1=%.2f%% mrr=%.4f meanrank=%.1f" %
          (sum(ref["v28"]["fact_d%d" % d]["acc1"] for d in DEPTH) / len(DEPTH) * 100,
           sum(ref["v28"]["fact_d%d" % d]["mrr"] for d in DEPTH) / len(DEPTH),
           sum(ref["v28"]["fact_d%d" % d]["mean_rank"] for d in DEPTH) / len(DEPTH)))
    json.dump(res, open(os.path.join(HERE, "needle_diag_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
