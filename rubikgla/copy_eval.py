#!/usr/bin/env python3
"""自然复制测试 (in-distribution copy recall): 针测验的分布内对应.
对位置 p∈[193,255]: 查询 = (x[p-2],x[p-1]), 目标 = x[p].
  A组 repeat: 该 bigram 在 d<p-2 复现过, 且早次续接 v'=x[d+2] 恰等于 x[p] → 模型应召回续接
  B组 novel : bigram 从未出现 → 先验基线
指标: 各组 rank/acc + lift(A−B) = 分布内检索能力. 两模型同输入严格配对."""
import sys, os, json, random
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deqlm_v24_hybrid2fast as M24
import deqlm_v28_deltaloop as M28
from deqlm_v5_30m import PT_CACHE

DEV = "cuda"
N_A, N_B, MAXB = 3000, 3000, 128
BS = 64
RES = os.path.join(HERE, "copy_results.json")


def build(seqs, rng):
    """收集 A(repeat+同续接) / B(novel) 试验."""
    A, B = [], []
    seen_seq = 0
    order = list(range(len(seqs)))
    rng.shuffle(order)
    for i in order:
        if len(A) >= N_A and len(B) >= N_B:
            break
        s = seqs[i]
        if len(s) < 256:
            continue
        x = s[:256].tolist()
        seen_seq += 1
        # bigram -> [(pos, cont)]
        hist = {}
        for t in range(191):
            bg = (x[t], x[t + 1])
            if bg not in hist:
                hist[bg] = []
            if len(hist[bg]) < 4:
                hist[bg].append((t, x[t + 2]))
        for p in range(193, 256):
            bg = (x[p - 2], x[p - 1])
            occ = [d for d, _ in hist.get(bg, []) if d < p - 2 and d <= 191]
            if occ:
                d = occ[-1]
                if x[d + 2] == x[p] and len(A) < N_A:
                    A.append(dict(x=x[:], p=p, d=d, dist=p - d - 2, tgt=x[p]))
            elif len(B) < N_B:
                B.append(dict(x=x[:], p=p, tgt=x[p]))
    return A, B, seen_seq


def evaluate(model, trials, tag, name):
    ranks = []
    with torch.no_grad():
        for i in range(0, len(trials), BS):
            chunk = trials[i:i + BS]
            inp = torch.tensor([t["x"] for t in chunk], device=DEV, dtype=torch.long)
            lg = model(inp)
            for j, t in enumerate(chunk):
                lp = lg[j, t["p"] - 1, :]                   # 预测位置 p
                tgt = t["tgt"]
                r = (lp.argsort(descending=True) == tgt).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(r)
    acc1 = sum(1 for r in ranks if r == 1) / len(ranks)
    acc10 = sum(1 for r in ranks if r <= 10) / len(ranks)
    mrr = sum(1.0 / r for r in ranks) / len(ranks)
    mr = sum(ranks) / len(ranks)
    print("%-6s %-6s n=%d acc1=%6.2f%% acc10=%6.2f%% mrr=%.4f meanrank=%7.1f"
          % (name, tag, len(ranks), acc1 * 100, acc10 * 100, mrr, mr))
    return dict(n=len(ranks), acc1=acc1, acc10=acc10, mrr=mrr, mean_rank=mr)


def main():
    torch.manual_seed(0)
    rng = random.Random(42)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    val = [s for s in seqs[:5000] if len(s) >= 256]
    A, B, ns = build(val, rng)
    print("pool=%d seqs  trials: A(repeat)=%d B(novel)=%d  A dist mean=%.0f"
          % (ns, len(A), len(B), sum(t["dist"] for t in A) / max(1, len(A))))

    ckpts = {"v24": (M24.Hybrid2FastLM, "deqlm24_pt_full.pth"),
             "v28": (M28.DeltaLoopLM, "deqlm28_pt_full.pth")}
    res = {}
    eval_models = {}
    for name, (cls, ck) in ckpts.items():
        m = cls()
        m.load_state_dict(torch.load(os.path.join(HERE, ck), map_location="cpu"))
        m = m.to(DEV).eval()
        
        ra = evaluate(m, A, "repeat", name)
        rb = evaluate(m, B, "novel", name)
        res[name] = dict(repeat=ra, novel=rb, lift_mrr=ra["mrr"] - rb["mrr"],
                         lift_acc1=ra["acc1"] - rb["acc1"])
        eval_models[name] = m
        torch.cuda.empty_cache()

    print("== repeat by distance band ==")
    for n in res:
        rr = res[n].setdefault("by_band", {})
        for lo, hi in ((0, 48), (48, 96), (96, 144), (144, 400)):
            sel = [t for t in A if lo <= t["dist"] < hi]
            if not sel:
                continue
            key = "%d-%d" % (lo, hi)
            r_ = evaluate(eval_models[n], sel, "d" + key, n)
            rr[key] = r_
    print("\n== lift (repeat − novel) = 分布内检索能力 ==")
    for n in res:
        print("%-6s d_mrr=%+.4f  d_acc1=%+.2f%%" % (n, res[n]["lift_mrr"], res[n]["lift_acc1"] * 100))
    json.dump(res, open(RES, "w"), indent=2)
    print("saved ->", RES)


if __name__ == "__main__":
    main()
