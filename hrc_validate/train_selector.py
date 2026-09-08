#!/usr/bin/env python3
"""
HRC-LLM v0.17 机制验证 · 阶段3: oracle 监督训练段打分器

问题: 廉价 surprise 打分只有 oracle 的 ~30% 能力 (阶段2)。
     文档红线承认"从零端到端训练打分器无正面证据"。
本实验: 用 oracle 标签 (needle 段=1) 监督训练一个小 MLP 打分器,
       段特征 = [token embedding 均值(有效token) + 段NLL + 位置],
       检验 supervised 打分器能否逼近文档硬指标 (B=15% 召回 >= 95%)。

对比: oracle 上界 / supervised / surprise / random
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


def load_model():
    from convash30 import ConvASH30, VOCAB
    m = ConvASH30(voc=VOCAB).to(DEV)
    m.load_state_dict(torch.load(
        r"F:\OpenASH2605\copyfirst_redesign\convash30_pt_full.pth",
        map_location="cpu", weights_only=True))
    m.eval()
    return m, VOCAB


def load_texts(path, n_max=140, min_len=1200):
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


def doc_with_needles(voc, text, n_needle=4, seed=0):
    import numpy.random
    rng = numpy.random.RandomState(seed)
    ids = voc.encode(text)[:2048]
    if len(ids) < 1024:
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
    return ids, secrets


@torch.no_grad()
def seg_features(model, emb_layer, ids, S=32):
    """段特征: [有效 token embedding 均值(640) + 段 NLL(1) + 位置(1)] -> [b, n_seg, 642]"""
    x = torch.tensor([ids], device=DEV)
    emb = emb_layer(x)[0]                                    # [L, 640]
    logits = model(x)[0][0]
    lg = F.log_softmax(logits.float(), -1)
    t = torch.tensor(ids, device=DEV)
    nll = -lg.gather(1, t.unsqueeze(1)).squeeze(1)           # [L]
    n_seg = len(ids) // S
    feats, nlls = [], []
    ids_t = torch.tensor(ids, device=DEV)
    for j in range(n_seg):
        a, b = j * S, (j + 1) * S
        e = emb[a:b]
        mask = (ids_t[a:b] != 0).float().unsqueeze(1)
        denom = mask.sum().clamp(min=1)
        mu = (e * mask).sum(0) / denom
        nlls.append(nll[a:b].mean().item())
        feats.append(mu)
    f = torch.stack(feats)                                    # [n_seg, 640]
    nll_t = torch.tensor(nlls, device=DEV).unsqueeze(1)
    pos = (torch.arange(n_seg, device=DEV, dtype=torch.float32) / n_seg).unsqueeze(1)
    return torch.cat([f, nll_t, pos], 1), torch.tensor(nlls)


class Scorer(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    t0 = time.time()
    model, V = load_model()
    emb_layer = model.em
    from open_ash_voc import OpenASHVoc
    voc = OpenASHVoc(agent_voc_path=r"F:\OpenASH2605\open_ash_voc_agent.json")
    texts = load_texts(r"F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl")
    print("texts=%d (%.0fs)" % (len(texts), time.time() - t0), flush=True)

    docs = []
    for i, txt in enumerate(texts):
        out = doc_with_needles(voc, txt, n_needle=4, seed=i)
        if out is None:
            continue
        ids, secrets = out
        feats, seg_nll = seg_features(model, emb_layer, ids)
        labels = torch.zeros(len(seg_nll))
        for si in secrets:
            labels[si] = 1.0
        docs.append((feats.cpu(), seg_nll.cpu(), labels, len(secrets)))
        if (i + 1) % 20 == 0:
            print("  特征 %d/%d (%.0fs)" % (i + 1, len(texts), time.time() - t0), flush=True)

    n_train = int(len(docs) * 0.8)
    train_docs, val_docs = docs[:n_train], docs[n_train:]
    print("docs=%d train=%d val=%d" % (len(docs), len(train_docs), len(val_docs)), flush=True)

    # 训练
    din = docs[0][0].shape[1]
    scorer = Scorer(din).to(DEV)
    opt = torch.optim.AdamW(scorer.parameters(), lr=3e-4, weight_decay=0.01)
    # 类别不平衡: 正样本 ~6%, pos_weight = neg/pos
    n_pos = sum(int(d[2].sum()) for d in train_docs)
    n_all = sum(len(d[2]) for d in train_docs)
    pos_w = (n_all - n_pos) / max(n_pos, 1)
    print("pos_weight=%.1f" % pos_w, flush=True)
    best_val = -1
    best_state = None
    for ep in range(60):
        scorer.train()
        random_order = list(range(len(train_docs)))
        np.random.RandomState(ep).shuffle(random_order)
        tl = 0.0
        for di in random_order:
            f, nll, lab, _ = train_docs[di]
            f, lab = f.to(DEV), lab.to(DEV)
            logit = scorer(f)
            loss = F.binary_cross_entropy_with_logits(logit, lab, pos_weight=torch.tensor(pos_w, device=DEV))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl += loss.item()
        # val AUC-ish: 平均正样本分 - 负样本分
        scorer.eval()
        margins = []
        with torch.no_grad():
            for f, nll, lab, _ in val_docs:
                s = scorer(f.to(DEV))
                pos_s = s[lab > 0]
                neg_s = s[lab == 0]
                if len(pos_s) and len(neg_s):
                    margins.append((pos_s.mean() - neg_s.mean()).item())
        vm = sum(margins) / len(margins) if margins else 0
        if vm > best_val:
            best_val = vm
            best_state = {k: v.clone() for k, v in scorer.state_dict().items()}
        if ep % 10 == 0:
            print("  ep%d loss=%.4f val_margin=%.4f" % (ep, tl / len(random_order), vm), flush=True)
    scorer.load_state_dict(best_state)

    # 评估: top-B 召回
    def recall_curve(kind, docs_):
        rows = {0.05: [], 0.1: [], 0.15: [], 0.25: [], 0.5: []}
        for f, nll, lab, _ in docs_:
            n_seg = len(lab)
            key = set(torch.nonzero(lab).squeeze(1).tolist())
            if kind == "supervised":
                with torch.no_grad():
                    s = scorer(f.to(DEV)).cpu().numpy()
            elif kind == "surprise":
                s = nll.numpy()
            elif kind == "random":
                s = np.random.RandomState(0).rand(n_seg)
            for Bf in rows:
                B = max(int(n_seg * Bf), 1)
                sel = set(np.argsort(s)[::-1][:B].tolist())
                rows[Bf].append(len(sel & key) / len(key))
        return {Bf: sum(v) / len(v) for Bf, v in rows.items()}

    print("\n=== 验证集召回 (n=%d 文档) ===" % len(val_docs), flush=True)
    print("  策略        B=5%%   B=10%%  B=15%%  B=25%%  B=50%%")
    table = {}
    for kind in ["oracle", "supervised", "surprise", "random"]:
        rows = {}
        for Bf in rows if False else [0.05, 0.1, 0.15, 0.25, 0.5]:
            vals = []
            for f, nll, lab, _ in val_docs:
                n_seg = len(lab)
                key = set(torch.nonzero(lab).squeeze(1).tolist())
                B = max(int(n_seg * Bf), 1)
                if kind == "oracle":
                    sel = set(np.argsort((lab > 0).float().numpy())[::-1][:B])
                elif kind == "supervised":
                    with torch.no_grad():
                        s = scorer(f.to(DEV)).cpu().numpy()
                    sel = set(np.argsort(s)[::-1][:B])
                elif kind == "surprise":
                    sel = set(np.argsort(nll.numpy())[::-1][:B])
                else:
                    sel = set(np.random.RandomState(0).choice(n_seg, B, replace=False).tolist())
                vals.append(len(sel & key) / len(key))
            rows[Bf] = sum(vals) / len(vals)
        table[kind] = rows
        print("  %-10s %.3f  %.3f  %.3f  %.3f  %.3f" %
              (kind, rows[0.05], rows[0.1], rows[0.15], rows[0.25], rows[0.5]), flush=True)

    torch.save({"scorer": scorer.state_dict(), "table": table},
               r"F:\夸克\hrc_validate\selector_v1.pt")
    print("\n用时 %.0fs | 门槛: B=15%% >= 0.95" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
