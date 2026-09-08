#!/usr/bin/env python3
"""
HRC-LLM v0.17 机制验证 · 阶段2: 记忆桥前提在真实文本上的成立性 (OpenASH 30M oracle)

HRC 记忆桥成立的两个前提 (与 CTS 无关, 纯 HRC):
  P1 证据集中度: 长文本中"回答所需证据"集中在少数段 (支持 5-15% 预算)
  P2 选择器有效性: 可计算的打分 (surprise) 能定位证据段, 优于随机

测量 (needle 检索, 因果可执行):
  长文档 doc (2048 tok) 随机埋 K 个"秘密数字段"(唯一四位数标记)
  打分器: 每段 teacher-force 平均 NLL (surprise 信号) / 位置先验
  预算 B%: 保留 top-B 段, 测"秘密段"召回
  对照: oracle(知道位置) / surprise / 均匀 / 随机
判死: P1 由 oracle 曲线给出 (若 oracle 在 B=15% 召回远低于 100% -> 证据不可集中);
      P2 由 surprise vs 随机差距给出 (HRC 文档要求选择器远优于随机).
"""
import sys, os, json, time
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import torch
import numpy as np

DEV = "cuda"
MODEL_CKPT = r"F:\OpenASH2605\copyfirst_redesign\convash30_pt_full.pth"


def load_model():
    from convash30 import ConvASH30, VOCAB
    m = ConvASH30(voc=VOCAB).to(DEV)
    m.load_state_dict(torch.load(MODEL_CKPT, map_location="cpu", weights_only=True))
    m.eval()
    return m, VOCAB


def load_texts(path, n_max=120, min_len=1200):
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                t = json.loads(line).get("text", "")
            except Exception:
                continue
            if len(t) > min_len:
                texts.append(t.replace("\n", ""))
            if len(texts) >= n_max:
                break
    return texts


def doc_with_needles(voc, text, S=32, n_needle=4, seed=0):
    """text -> ids[:2048]; 在随机段位置替换为秘密数字(4位唯一)标记段."""
    import random
    rng = random.RandomState(seed) if hasattr(random, "RandomState") else None
    import numpy.random
    rng = numpy.random.RandomState(seed)
    ids = voc.encode(text)[:2048]
    if len(ids) < 1024:
        return None
    n_seg = len(ids) // S
    seg_idx = [i for i in range(n_seg)]
    chosen = rng.choice(seg_idx, n_needle, replace=False)
    secrets = {}
    for j, si in enumerate(chosen):
        secret = 1000 + rng.randint(0, 9000)
        tok = voc.encode("数字%d" % secret)
        if len(tok) >= S:
            tok = tok[:S]
        a, b = si * S, si * S + S
        ids[a:b] = (tok + [0] * S)[:S]        # pad 到段长
        secrets[si] = secret
    return ids, secrets


def seg_nll_scores(model, ids, S=32):
    """整序列一次前向, 聚合每段平均 NLL (surprise) 与位置."""
    x = torch.tensor([ids], device=DEV)
    import torch.nn.functional as F
    with torch.no_grad():
        lg = F.log_softmax(model(x)[0][0].float(), -1)
    t = torch.tensor(ids, device=DEV)
    nll = -lg.gather(1, t.unsqueeze(1)).squeeze(1).cpu().numpy()   # [L]
    n_seg = len(ids) // S
    scores = np.array([nll[i * S:(i + 1) * S].mean() for i in range(n_seg)])
    return scores


def recall_curve(secrets, scores, B_fracs=(0.05, 0.1, 0.15, 0.25, 0.5), use_pos=False):
    """给定段分数(越高=越该保留), 测秘密段在 top-B 中的比例."""
    n_seg = len(scores)
    n_sec = len(secrets)
    key_idx = list(secrets.keys())
    rows = {}
    for Bf in B_fracs:
        B = max(int(n_seg * Bf), 1)
        sel = set(np.argsort(scores)[::-1][:B].tolist())
        rows[Bf] = len(sel & set(key_idx)) / n_sec
    return rows


def main():
    t0 = time.time()
    model, V = load_model()
    from open_ash_voc import OpenASHVoc
    voc = OpenASHVoc(agent_voc_path=r"F:\OpenASH2605\open_ash_voc_agent.json")
    texts = load_texts(r"F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl")
    print("model ready, texts=%d (%.0fs)" % (len(texts), time.time() - t0), flush=True)

    BF = (0.05, 0.1, 0.15, 0.25, 0.5)
    agg = {(name, Bf): [] for name in ("oracle", "surprise", "uniform", "random") for Bf in BF}
    n_ok = 0
    for i, text in enumerate(texts):
        out = doc_with_needles(voc, text, S=32, n_needle=4, seed=i)
        if out is None:
            continue
        ids, secrets = out
        n_seg = len(ids) // 32
        # oracle 分数: 秘密段分数最高
        sc_oracle = np.zeros(n_seg)
        for si in secrets:
            sc_oracle[si] = 1.0
        # surprise
        try:
            sc_surp = seg_nll_scores(model, ids)
        except Exception:
            continue
        # uniform (位置均匀, 无信息)
        sc_uni = np.arange(n_seg)[::-1] / n_seg
        # 随机 (每样本不同种子)
        rng = np.random.RandomState(i)
        sc_rnd = rng.rand(n_seg)
        for name, sc in (("oracle", sc_oracle), ("surprise", sc_surp),
                         ("uniform", sc_uni), ("random", sc_rnd)):
            for Bf, v in recall_curve(secrets, sc).items():
                agg[(name, Bf)].append(v)
        n_ok += 1
        if n_ok % 20 == 0:
            print("  %d ok (%.0fs)" % (n_ok, time.time() - t0), flush=True)

    print("\n===== HRC 阶段2: 证据集中度 + 选择器 (n=%d 文档, 4 needle/文档) =====" % n_ok, flush=True)
    print("  策略        B=5%%      B=10%%     B=15%%     B=25%%     B=50%%")
    for name in ("oracle", "surprise", "uniform", "random"):
        row = []
        for Bf in (0.05, 0.1, 0.15, 0.25, 0.5):
            v = agg[(name, Bf)]
            row.append("%.3f     " % (sum(v) / len(v) if v else float("nan")))
        print("  %-10s%s" % (name, "  ".join(row)), flush=True)
    torch.save(agg, r"F:\夸克\hrc_validate\hrc_stage2.pt")
    print("用时 %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
