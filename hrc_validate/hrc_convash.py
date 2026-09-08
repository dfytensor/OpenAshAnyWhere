#!/usr/bin/env python3
"""
HRC-Lite on ConvASH30 (30M): em共享 / 前8层编码 / 记忆桥 / 后8层解码 / head共享

任务: doc[:1920] → 编码+选段+记忆 → 解码 doc[1920:2048]
对比: hrc_mem vs full_ctx(16层AR) vs no_mem(仅窗口)
"""
import sys, os, json, time, math
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEV = "cuda"
S_SEG = 32
TARGET = 128
WINDOW = 64
SPLIT = 8
D = 640


class ScaleShift(nn.Module):
    """轻量可训练适配器."""
    def __init__(self, d):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.shift = nn.Parameter(torch.zeros(d) * 0.01)

    def forward(self, x):
        return x * self.scale + self.shift


class HRCLiteConvASH(nn.Module):
    def __init__(self, convash, split=SPLIT):
        super().__init__()
        self.ca = convash
        self.split = split
        self.d = D
        # 冻结全部原参数
        for p in self.ca.parameters():
            p.requires_grad_(False)
        # 记忆桥
        self.seg_scorer = nn.Linear(D, 1)
        self.mem_proj = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D), nn.Tanh())
        # 只训练记忆桥 (不加 adapter, 排除干扰)
        # 只训练新参数
        self.trainable = list(self.seg_scorer.parameters()) + \
                         list(self.mem_proj.parameters())

    def encode(self, ids):
        """前 8 层跑 doc[:1920], 返回 hidden [b, L, d]."""
        x = self.ca.em(ids)
        state = [None] * SPLIT
        for i in range(SPLIT):
            x1, state[i] = self.ca.decoder_layers[i](x, state[i])
            x = x1 + x
        return x

    def decode(self, x):
        """后 8 层跑 [mem|win|tgt] 序列."""
        state = [None] * (16 - SPLIT)
        for i, layer in enumerate(self.ca.decoder_layers[SPLIT:]):
            x1, state[i] = layer(x, state[i])
            x = x1 + x
        return x

    def forward(self, doc_ids, target_ids):
        b = doc_ids.shape[0]
        h_enc = self.encode(doc_ids)                          # [b, 1920, d]
        n_seg = doc_ids.shape[1] // S_SEG
        seg_repr = h_enc.view(b, n_seg, S_SEG, D).mean(2)     # [b, n_seg, d]
        scores = self.seg_scorer(seg_repr).squeeze(-1)        # [b, n_seg]
        k = max(int(n_seg * 0.15), 1)
        top_idx = scores.topk(k, dim=1).indices
        mem = seg_repr.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, D))
        mem_soft = self.mem_proj(mem) * 0.1              # 缩放到 embedding 量级       # [b, k, d]

        win = self.ca.em(doc_ids[:, -WINDOW:])
        tgt = self.ca.em(target_ids[:, :-1])
        dec_in = torch.cat([mem_soft, win, tgt], dim=1)
        logits = self.decode(dec_in)                          # [b, k+W+127, V]

        offset = self.n_mem_or_k(k) + WINDOW
        pred = logits[:, offset - 1: offset - 1 + TARGET - 1]
        labels = target_ids[:, 1:]
        return pred, labels, scores, mem_soft

    def n_mem_or_k(self, k):
        return k


def main():
    t0 = time.time()
    from convash30 import ConvASH30, VOCAB
    from open_ash_voc import OpenASHVoc
    voc = OpenASHVoc(agent_voc_path=r"F:\OpenASH2605\open_ash_voc_agent.json")
    base = ConvASH30(voc=VOCAB).to(DEV)
    base.load_state_dict(torch.load(
        r"F:\OpenASH2605\copyfirst_redesign\convash30_pt_full.pth",
        map_location="cpu", weights_only=True))
    base.eval()
    model = HRCLiteConvASH(base, split=SPLIT).to(DEV)

    trainable = list(model.seg_scorer.parameters()) + \
                list(model.mem_proj.parameters())
    n_train = sum(p.numel() for p in trainable)
    print("可训练: %.2fM / 全部: %.1fM" % (n_train / 1e6, 29.5), flush=True)

    # 数据
    texts = []
    for p in [r"F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl"]:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    t = json.loads(line).get("text", "").replace("\n", "")
                    if len(t) > 1500:
                        texts.append(t)
                except Exception:
                    pass
                if len(texts) >= 600:
                    break
    print("texts:", len(texts), flush=True)

    docs = []
    for txt in texts:
        ids = voc.encode(txt)[:1920 + TARGET]
        ids = [max(0, min(x, VOCAB - 1)) for x in ids]
        if len(ids) < 1920 + TARGET:
            ids = ids + [0] * (1920 + TARGET - len(ids))
        docs.append(torch.tensor([ids], device=DEV))
    n_train = len(docs) - 50
    train_docs, val_docs = docs[:n_train], docs[n_train:]
    print("train=%d val=%d" % (len(train_docs), len(val_docs)), flush=True)

    # 训练
    opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
    steps = 1500
    model.train()
    t0 = time.time()
    for st in range(steps):
        doc = train_docs[st % len(train_docs)]
        doc_ids = doc[:, :1920]
        target = doc[:, 1920:1920 + TARGET]
        pred, labels, _, _ = model(doc_ids, target)
        loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]).float(),
                               labels.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if st % 200 == 0 or st == steps - 1:
            print("  step %d/%d loss=%.4f (%.0fs)" %
                  (st, steps, loss.item(), time.time() - t0), flush=True)

    # 评估
    model.eval()
    print("\n=== 评估 (val %d docs, %d tok) ===" % (len(val_docs), TARGET), flush=True)
    for kind in ["hrc_mem", "full_ctx", "no_mem"]:
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for doc in val_docs:
                doc_ids = doc[:, :1920]
                target = doc[:, 1920:1920 + TARGET]
                if kind == "hrc_mem":
                    pred, labels, _, _ = model(doc_ids, target)
                elif kind == "full_ctx":
                    logits, _ = model.ca(doc[:, :-1])
                    lg = F.log_softmax(logits[0, 1919:1919 + TARGET - 1].float(), -1)
                    t = target[0, 1:]
                    nll = -lg.gather(1, t.unsqueeze(1)).squeeze(1)
                    tot += nll.mean().item(); cnt += 1
                    continue
                else:
                    win = model.ca.em(doc_ids[:, -WINDOW:])
                    tgt = model.ca.em(target[:, :-1])
                    dec_in = torch.cat([win, tgt], dim=1)
                    logits = model.decode(dec_in)
                    offset = WINDOW
                    pred = logits[:, offset - 1:offset - 1 + TARGET - 1]
                    labels = target[:, 1:]
                loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]).float(),
                                       labels.reshape(-1))
                tot += loss.item(); cnt += 1
        print("  %-10s %.4f nats/tok" % (kind, tot / cnt), flush=True)
    print("\n完成 %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
