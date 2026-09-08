#!/usr/bin/env python3
"""HRC-Lite 训练 + 评估: 记忆桥压缩 vs 全上下文."""
import sys, os, json, time, math
sys.path.insert(0, r"F:\Spark-X2.5-1.7B")
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
sys.path.insert(0, r"F:\夸克\hrc_validate")
import torch
import torch.nn.functional as F
import numpy as np
from hrc_lite import HRCLite, S_SEG, TARGET, WINDOW

DEV = "cuda"


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


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(r"F:\Spark-X2.5-1.7B", trust_remote_code=True)
    spark = AutoModelForCausalLM.from_pretrained(
        r"F:\Spark-X2.5-1.7B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True)
    spark.eval()

    model = HRCLite(spark, split=14, top_frac=0.15, n_mem=10).to(DEV)

    # 可训练参数统计
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in model.parameters())
    print("可训练: %.2fM / 全部: %.2fB (%.2f%%)" %
          (n_train / 1e6, n_all / 1e9, n_train / n_all * 100), flush=True)

    # 数据
    texts = load_texts(r"F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl", 80, 2600)
    print("texts:", len(texts), flush=True)
    docs = []
    for i, txt in enumerate(texts):
        ids = tok.encode(txt, add_special_tokens=False)[:1920 + TARGET]
        if len(ids) < 1920 + TARGET:
            ids = ids + [tok.eos_token_id or 2] * (1920 + TARGET - len(ids))
        docs.append(torch.tensor([ids], device=DEV))
    n_train = int(len(docs) * 0.85)
    train_docs, val_docs = docs[:n_train], docs[n_train:]

    # 训练
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-4, weight_decay=0.01)
    steps = 400
    model.train()
    t0 = time.time()
    for st in range(steps):
        di = st % len(train_docs)
        doc = train_docs[di]
        doc_ids = doc[:, :1920]
        target = doc[:, 1920:1920 + TARGET]
        pred, labels, scores, _ = model(doc_ids, target)
        loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]).float(),
                               labels.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if st % 50 == 0:
            print("  step %d/%d loss=%.4f (%.0fs)" %
                  (st, steps, loss.item(), time.time() - t0), flush=True)

    # 评估: 记忆桥 vs 全上下文 vs 无记忆
    model.eval()
    print("\n=== 评估 (val %d docs, %d token 续写) ===" % (len(val_docs), TARGET), flush=True)
    for kind in ["hrc_mem", "full_ctx", "no_mem"]:
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for doc in val_docs:
                doc_ids = doc[:, :1920]
                target = doc[:, 1920:1920 + TARGET]
                if kind == "hrc_mem":
                    pred, labels, _, _ = model(doc_ids, target)
                elif kind == "full_ctx":
                    # 全上下文 AR: teacher forcing 整序列
                    x = doc[:, :-1]
                    logits = model.spark(x).logits
                    lg = F.log_softmax(logits[0, 1919:1919 + TARGET - 1].float(), -1)
                    t = target[0, 1:]
                    nll = -lg.gather(1, t.unsqueeze(1)).squeeze(1)
                    tot += nll.mean().item(); cnt += 1
                    continue
                else:
                    # 无记忆: 只有窗口
                    win = doc_ids[:, -WINDOW:]
                    tgt_e = model.spark.model.embedding(target[:, :-1])
                    win_e = model.spark.model.embedding(win)
                    dec_in = torch.cat([win_e, tgt_e], dim=1)
                    logits = model.decode(dec_in)
                    offset = WINDOW
                    pred = logits[:, offset - 1:offset - 1 + TARGET - 1]
                    labels = target[:, 1:]
                loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]),
                                       labels.reshape(-1))
                tot += loss.item(); cnt += 1
        print("  %-10s %.4f nats/tok" % (kind, tot / cnt), flush=True)
    print("\n完成 %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
