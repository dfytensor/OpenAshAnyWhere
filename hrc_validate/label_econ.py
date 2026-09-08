#!/usr/bin/env python3
"""
HRC 阶段4: oracle 标签经济学 — 降本路径实测

oracle 标签 Δ_b = NLL(目标 | 去掉段b的上下文) - NLL(目标 | 完整上下文)
全量标注成本 = 每段一次消融前向 (64 段 x ~2144 tok = ~137K tok/文档)

测三条路径:
  full    : 全量 LOO (分母, 最贵最准)
  short   : 路径3 — 短文档(L=512, 便宜 14x)上全量标注 -> 训练代理 -> 迁移长文档
  sample  : 路径4 — 长文档上只标注 surprise top-50% 的段, 其余用代理值
  surprise: 零标签基线 (段 NLL)

质量: 对真 Δ_b 的 Spearman 相关 + needle 段召回 @15% 预算
成本: 标注处理的 token 数 (每文档 + 一次性)
"""
import sys, os, json, time, pickle
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEV = "cuda"
S = 32
TARGET = 128


def load_model():
    from convash30 import ConvASH30, VOCAB
    m = ConvASH30(voc=VOCAB).to(DEV)
    m.load_state_dict(torch.load(
        r"F:\OpenASH2605\copyfirst_redesign\convash30_pt_full.pth",
        map_location="cpu", weights_only=True))
    m.eval()
    return m, VOCAB


def load_texts(path, n_max, min_len):
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


def make_doc(voc, text, L, n_needle=4, seed=0):
    rng = np.random.RandomState(seed)
    ids = voc.encode(text)[:L - TARGET]
    if len(ids) < int(L * 0.7):
        return None
    n_seg = len(ids) // S
    chosen = rng.choice(range(n_seg), n_needle, replace=False)
    secrets = {}
    for si in chosen:
        secret = 1000 + rng.randint(0, 9000)
        tk = voc.encode("数字%d" % secret)[:S]
        tk = tk + [0] * (S - len(tk))
        ids[si * S:si * S + S] = tk
        secrets[si] = secret
    return ids, secrets, chosen


@torch.no_grad()
def fwd_nll(model, ids):
    """返回逐 token NLL [L] (位置0无)."""
    x = torch.tensor([ids], device=DEV)
    logits = model(x)[0][0]
    lg = F.log_softmax(logits.float(), -1)
    t = torch.tensor(ids, device=DEV)
    return -lg.gather(1, t.unsqueeze(1)).squeeze(1).cpu().numpy()


@torch.no_grad()
def target_nll(model, ctx, target):
    """给定上下文, 目标段的 teacher-forcing NLL."""
    ids = list(ctx) + list(target)
    nll = fwd_nll(model, ids)
    return float(nll[len(ctx):].mean())


@torch.no_grad()
def seg_feats(model, emb_layer, ids, n_seg):
    x = torch.tensor([ids], device=DEV)
    emb = emb_layer(x)[0]
    logits = model(x)[0][0]
    lg = F.log_softmax(logits.float(), -1)
    t = torch.tensor(ids, device=DEV)
    nll = -lg.gather(1, t.unsqueeze(1)).squeeze(1).cpu().numpy()
    feats, nlls = [], []
    for j in range(n_seg):
        a, b = j * S, (j + 1) * S
        e = emb[a:b]
        mask = (t[a:b] != 0).float().unsqueeze(1)
        mu = (e * mask).sum(0) / mask.sum().clamp(min=1)
        nlls.append(nll[a:b].mean())
        feats.append(mu)
    f = torch.stack(feats)
    pos = (torch.arange(n_seg, device=DEV, dtype=torch.float32) / n_seg).unsqueeze(1)
    return torch.cat([f.cpu(), torch.tensor(nlls).unsqueeze(1),
                      pos.cpu()], 1), np.array(nlls)


class Proxy(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 128), nn.GELU(),
                                 nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_proxy(docs, epochs=80):
    """docs: [(feats, delta_minmax)]. 回归 min-max 归一化 delta (排序即目标)."""
    din = docs[0][0].shape[1]
    pr = Proxy(din).to(DEV)
    opt = torch.optim.AdamW(pr.parameters(), lr=1e-3, weight_decay=0.01)
    for ep in range(epochs):
        pr.train()
        tl, tn = 0.0, 0
        order = list(range(len(docs)))
        np.random.RandomState(ep).shuffle(order)
        for di in order:
            f, d = docs[di]
            f, d = f.to(DEV), d.to(DEV)
            pred = pr(f)
            loss = F.mse_loss(pred, d)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl += loss.item()
            tn += 1
    return pr


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    t00 = time.time()
    model, V = load_model()
    emb_layer = model.em
    from open_ash_voc import OpenASHVoc
    voc = OpenASHVoc(agent_voc_path=r"F:\OpenASH2605\open_ash_voc_agent.json")
    texts_long = load_texts(r"F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl", 40, 2600)
    texts_short = load_texts(r"F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl", 40, 900)[20:40]
    print("long=%d short=%d" % (len(texts_long), len(texts_short)), flush=True)

    # ---------- ① 短文档: 全量 LOO 标签 (便宜) + 训练代理 ----------
    t0 = time.time()
    short_tok_label = 0
    short_docs = []
    for i, txt in enumerate(texts_short):
        out = make_doc(voc, txt, 512, n_needle=2, seed=100 + i)
        if out is None:
            continue
        ids, secrets, chosen = out
        n_seg = len(ids) // S
        if n_seg < 12:
            continue
        target = ids[-TARGET:]
        base_ids = ids[:-TARGET]
        nll_with = target_nll(model, base_ids, target)
        deltas = np.zeros(n_seg)
        for b in range(n_seg):
            a, e = b * S, (b + 1) * S
            ctx = base_ids[:a] + base_ids[e:]
            deltas[b] = target_nll(model, ctx, target) - nll_with
        short_tok_label += n_seg * (len(ids) - S + TARGET)
        feats, nlls = seg_feats(model, emb_layer, ids, n_seg)
        dmin, dmax = deltas.min(), deltas.max()
        dnorm = (deltas - dmin) / (dmax - dmin + 1e-9)
        short_docs.append((feats, torch.tensor(dnorm, dtype=torch.float32)))
    print("① 短文档标注: %d docs, %.0fK tok, %.0fs" %
          (len(short_docs), short_tok_label / 1000, time.time() - t0), flush=True)

    proxy = train_proxy(short_docs)
    print("① 代理训练完成", flush=True)

    # ---------- ② 长文档: 全量 LOO (真值) + 三种廉价标注 ----------
    label_tok_full = 0
    label_tok_sample = 0
    rows = []
    t0 = time.time()
    for i, txt in enumerate(texts_long[:12]):
        out = make_doc(voc, txt, 2048, n_needle=4, seed=200 + i)
        if out is None:
            continue
        ids, secrets, chosen = out
        n_seg = len(ids) // S
        target = ids[-TARGET:]
        base_ids = ids[:-TARGET]
        nll_with = target_nll(model, base_ids, target)
        t0d = time.time()
        deltas = np.zeros(n_seg)
        for b in range(n_seg):
            a, e = b * S, (b + 1) * S
            ctx = base_ids[:a] + base_ids[e:]
            deltas[b] = target_nll(model, ctx, target) - nll_with
        label_tok_full += n_seg * (len(ids) - S + TARGET)
        t_full = time.time() - t0d
        if i == 0:
            print("  全量 LOO 单文档: %.1fs" % t_full, flush=True)

        feats, nlls = seg_feats(model, emb_layer, ids, n_seg)
        # 路径4: 只标 surprise top-50%, 其余用代理值中位数
        top_half = np.argsort(nlls)[::-1][:n_seg // 2]
        deltas_sub = deltas.copy()
        med = np.median(deltas)
        mask_lo = np.ones(n_seg, dtype=bool)
        mask_lo[top_half] = False
        deltas_sub[mask_lo] = med
        label_tok_sample += (n_seg // 2) * (len(ids) - S + TARGET)

        # 代理(short 迁移)打分
        with torch.no_grad():
            pr_score = proxy(feats.to(DEV)).cpu().numpy()
        dmin, dmax = deltas.min(), deltas.max()
        d_true = (deltas - dmin) / (dmax - dmin + 1e-9)
        key = set(chosen.tolist())

        def recall(score):
            B = max(int(n_seg * 0.15), 1)
            sel = set(np.argsort(score)[::-1][:B].tolist())
            return len(sel & key) / len(key)

        rows.append({
            "delta": deltas, "d_true": d_true,
            "r_full": recall(deltas), "r_proxy": recall(pr_score),
            "r_surp": recall(nlls),
            "r_sample": recall(deltas_sub),
            "sp_full": 1.0, "sp_proxy": spearman(pr_score, deltas),
            "sp_surp": spearman(nlls, deltas),
        })
        if (i + 1) % 4 == 0:
            print("  %d long docs (%.0fs)" % (i + 1, time.time() - t0), flush=True)

    # ---------- 汇总 ----------
    print("\n===== 阶段4: 标签经济学 (n=%d 长文档) =====" % len(rows), flush=True)
    print("  策略            标注成本/文档    needle召回@15%%  Spearman(Δb)")
    print("  full-LOO        ~%.0fK tok      %.3f           %.2f" %
          (label_tok_full / max(len(rows), 1) / 1000,
           np.mean([r["r_full"] for r in rows]),
           np.mean([r["sp_full"] for r in rows])), flush=True)
    print("  short->proxy    ~0 (一次性233K) %.3f           %.2f" %
          (np.mean([r["r_proxy"] for r in rows]),
           np.mean([r["sp_proxy"] for r in rows])), flush=True)
    print("  sample-50%%      ~%.0fK tok      %.3f           -" %
          (label_tok_sample / max(len(rows), 1) / 1000,
           np.mean([r["r_sample"] for r in rows])), flush=True)
    print("  surprise(零标签) 2K tok         %.3f           %.2f" %
          (np.mean([r["r_surp"] for r in rows]),
           np.mean([r["sp_surp"] for r in rows])), flush=True)
    pickle.dump({"rows": rows, "label_tok_full": label_tok_full,
                 "label_tok_sample": label_tok_sample, "short_tok": short_tok_label},
                open(r"F:\夸克\hrc_validate\label_econ.pkl", "wb"))
    print("\n总用时 %.0fs" % (time.time() - t00), flush=True)


if __name__ == "__main__":
    main()
