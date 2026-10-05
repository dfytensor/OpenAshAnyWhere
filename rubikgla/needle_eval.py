#!/usr/bin/env python3
"""针测验 (needle eval): 精确检索能力对照 v24(softmax) vs v28(delta) — DPSH Phase13 口径.
构造: 真实 val 序列做背景, 深度 d 处植入事实 [k1,k2,v], 序列尾部 [191,192] 放查询 [k1,k2],
      看位置 193 的预测中 v 的排名. 对照: 只有查询没有事实 (bigram 先验基线).
按查询-事实距离分带 (d=4..180, 距离 12..188 token). 两模型 eval() fp32 前向, 严格配对同 trial."""
import sys, os, json, random
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deqlm_v24_hybrid2fast as M24
import deqlm_v28_deltaloop as M28
import deqlm_v30_kv4 as M30
from deqlm_v5_30m import PT_CACHE

DEV = "cuda"
DEPTH = [4, 20, 52, 84, 116, 148, 180]      # 事实位置, 距查询 192-d
N_FACT, N_CTRL = 300, 150                    # 每深度 trial 数
BS = 64
RES = os.path.join(HERE, "needle_results.json")


def build_trials(seqs, rng):
    """返回 (x_batch list, meta list). 每条 256 token."""
    xs, metas = [], []
    vocab_pool = None
    for d in DEPTH:
        for t in range(N_FACT + N_CTRL):
            i = rng.randrange(len(seqs))
            x = seqs[i][:256].clone().tolist()
            if vocab_pool is None:
                pass
            k1, k2, v = rng.randrange(20000), rng.randrange(20000), rng.randrange(20000)
            while len({k1, k2, v}) < 3:
                k1, k2, v = rng.randrange(20000), rng.randrange(20000), rng.randrange(20000)
            has_fact = t >= N_CTRL
            if has_fact:
                x[d], x[d + 1], x[d + 2] = k1, k2, v      # 事实植入
            x[191], x[192] = k1, k2                        # 查询
            xs.append(x)
            metas.append(dict(depth=d, dist=192 - d, fact=has_fact, v=v))
    return xs, metas


def evaluate(model, xs, metas):
    out = []
    with torch.no_grad():
        for i in range(0, len(xs), BS):
            inp = torch.tensor(xs[i:i + BS], device=DEV, dtype=torch.long)
            logits = model(inp)                            # (B,256,V)
            lg = logits[:, 192, :]                          # 预测 193
            ranks = lg.argsort(dim=-1, descending=True)     # (B,V)
            for j in range(lg.shape[0]):
                v = metas[i + j]["v"]
                pos = (ranks[j] == v).nonzero(as_tuple=True)[0].item()
                out.append(pos + 1)                         # 1-based rank
    return out


def main():
    torch.manual_seed(0)
    rng = random.Random(42)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    val = [s for s in seqs[:5000] if len(s) >= 256]
    print("val pool: %d" % len(val))
    xs, metas = build_trials(val, rng)
    print("trials: %d (fact=%d ctrl=%d) x %d depths" % (len(xs), N_FACT, N_CTRL, len(DEPTH)))

    models = {}
    m = M24.Hybrid2FastLM()
    m.load_state_dict(torch.load(os.path.join(HERE, "deqlm24_pt_full.pth"), map_location="cpu"))
    models["v24"] = m.to(DEV).eval()
    m = M28.DeltaLoopLM()
    m.load_state_dict(torch.load(os.path.join(HERE, "deqlm28_pt_full.pth"), map_location="cpu"))
    models["v28"] = m.to(DEV).eval()
    m = M30.DeltaLoopLM()
    m.load_state_dict(torch.load(os.path.join(HERE, "deqlm30_pt_full.pth"), map_location="cpu"))
    models["v30"] = m.to(DEV).eval()

    res = {}
    for name, m in models.items():
        ranks = evaluate(m, xs, metas)
        r = {}
        for d in DEPTH:
            for kind, want in (("fact", True), ("ctrl", False)):
                sel = [rk for rk, mt in zip(ranks, metas) if mt["depth"] == d and mt["fact"] == want]
                if not sel:
                    continue
                key = "%s_d%d" % (kind, d)
                r[key] = dict(
                    acc1=sum(1 for x in sel if x == 1) / len(sel),
                    acc5=sum(1 for x in sel if x <= 5) / len(sel),
                    mrr=sum(1.0 / x for x in sel) / len(sel),
                    mean_rank=sum(sel) / len(sel),
                    n=len(sel))
        res[name] = r

    hdr = "%-6s %s" % ("model", "  ".join("d=%-3d" % d for d in DEPTH))
    print("\n== Fact Acc@1 (真植入, 距离=192-d) ==\n" + hdr)
    for name in models:
        print("%-6s %s" % (name, "  ".join("%5.1f%%" % (res[name]["fact_d%d" % d]["acc1"] * 100) for d in DEPTH)))
    print("\n== Ctrl Acc@1 (无针, bigram 先验) ==")
    for name in models:
        print("%-6s %s" % (name, "  ".join("%5.1f%%" % (res[name]["ctrl_d%d" % d]["acc1"] * 100) for d in DEPTH)))
    print("\n== Fact MRR ==")
    for name in models:
        print("%-6s %s" % (name, "  ".join("%6.3f" % res[name]["fact_d%d" % d]["mrr"] for d in DEPTH)))
    print("\n== Fact mean rank (1/23005=随机) ==")
    for name in models:
        print("%-6s %s" % (name, "  ".join("%6.1f" % res[name]["fact_d%d" % d]["mean_rank"] for d in DEPTH)))

    json.dump(res, open(RES, "w"), indent=2)
    print("\nsaved ->", RES)


if __name__ == "__main__":
    main()
