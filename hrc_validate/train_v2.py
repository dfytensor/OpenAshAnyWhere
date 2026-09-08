#!/usr/bin/env python3
"""HRC-Lite v2: 多样文本 + 更多数据 + 更长训练."""
import sys, os, json, time, math
sys.path.insert(0, r"F:\Spark-X2.5-1.7B")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from hrc_lite import HRCLite, TARGET, WINDOW, S_SEG

DEV = "cuda"


def main():
    tok = AutoTokenizer.from_pretrained(r"F:\Spark-X2.5-1.7B", trust_remote_code=True)
    spark = AutoModelForCausalLM.from_pretrained(
        r"F:\Spark-X2.5-1.7B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True)
    model = HRCLite(spark, split=14, top_frac=0.15, n_mem=10).to(DEV)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print("可训练: %.2fM" % (n_train / 1e6), flush=True)

    # 数据
    texts = []
    for p in [r"F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl",
              r"F:\OpenASH2605\minimind_data\sft_t2t_mini.jsonl"]:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    t = d.get("text", "")
                    if not t and "conversations" in d:
                        t = "".join(x.get("content", "") for x in d["conversations"])
                    if len(t) > 1500:
                        texts.append(t.replace("\n", ""))
                except Exception:
                    pass
                if len(texts) >= 3000:
                    break
    print("可用文本:", len(texts), flush=True)

    rng = np.random.RandomState(42)
    rng.shuffle(texts)
    # 拼接短文本为 2048+ token 的长文档
    docs = []
    buf = []
    buf_len = 0
    for txt in texts:
        ids = tok.encode(txt, add_special_tokens=False)
        buf.append(ids)
        buf_len += len(ids)
        if buf_len >= 1920 + TARGET:
            merged = [t for sub in buf for t in sub][:1920 + TARGET]
            docs.append(torch.tensor([merged], device=DEV))
            buf, buf_len = [], 0
        if len(docs) >= 800:
            break
    print("docs:", len(docs), flush=True)
    n_train = len(docs) - 50
    train_docs, val_docs = docs[:n_train], docs[n_train:]

    # 训练
    opt = torch.optim.AdamW(trainable, lr=5e-5, weight_decay=0.01)
    steps = 2000
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
        if st % 100 == 0 or st == steps - 1:
            print("  step %d/%d loss=%.4f (%.0fs)" %
                  (st, steps, loss.item(), time.time() - t0), flush=True)
        if (st + 1) % 500 == 0:
            torch.save({"model": model.state_dict(), "step": st + 1},
                       r"F:\夸克\hrc_validate\hrc_lite_ckpt.pt")
            print("  ckpt @%d" % (st + 1), flush=True)

    # 评估
    model.eval()
    print("\n=== 评估 (val %d docs, %d tok) ===" % (len(val_docs), TARGET), flush=True)
    for kind in ["hrc_mem", "full_ctx", "no_mem", "no_mem_win128"]:
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for doc in val_docs:
                doc_ids = doc[:, :1920]
                target = doc[:, 1920:1920 + TARGET]
                if kind == "hrc_mem":
                    pred, labels, _, _ = model(doc_ids, target)
                elif kind == "full_ctx":
                    x = doc[:, :-1]
                    logits = model.spark(x).logits
                    lg = F.log_softmax(logits[0, 1919:1919 + TARGET - 1].float(), -1)
                    t = target[0, 1:]
                    nll = -lg.gather(1, t.unsqueeze(1)).squeeze(1)
                    tot += nll.mean().item(); cnt += 1
                    continue
                elif kind == "no_mem":
                    win = doc_ids[:, -WINDOW:]
                    win_e = model.spark.model.embedding(win)
                    tgt_e = model.spark.model.embedding(target[:, :-1])
                    dec_in = torch.cat([win_e, tgt_e], dim=1)
                    logits = model.decode(dec_in)
                    pred = logits[:, WINDOW - 1:WINDOW - 1 + TARGET - 1]
                    labels = target[:, 1:]
                elif kind == "no_mem_win128":
                    win = doc_ids[:, -128:]
                    win_e = model.spark.model.embedding(win)
                    tgt_e = model.spark.model.embedding(target[:, :-1])
                    dec_in = torch.cat([win_e, tgt_e], dim=1)
                    logits = model.decode(dec_in)
                    pred = logits[:, 127:127 + TARGET - 1]
                    labels = target[:, 1:]
                loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]).float(),
                                       labels.reshape(-1))
                tot += loss.item(); cnt += 1
        print("  %-16s %.4f nats/tok" % (kind, tot / cnt), flush=True)
    print("\n完成 %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
