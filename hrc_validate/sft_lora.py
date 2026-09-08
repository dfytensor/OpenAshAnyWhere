#!/usr/bin/env python3
"""HRC-lite LoRA SFT: 骨架->正文 填充能力训练 (Spark 1.7B)."""
import sys, os, json, time
sys.path.insert(0, r"F:\Spark-X2.5-1.7B")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

DEV = "cuda"
MDIR = r"F:\Spark-X2.5-1.7B"
DATA = r"F:\夸克\hrc_validate\hrc_docs.jsonl"
OUT = r"F:\夸克\hrc_validate\lora_hrc"
MARK_SK = "\n【要点】\n"
MARK_BODY = "\n【正文】\n"


def load_samples(path):
    xs, sks, rnds, bodys = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            xs.append(r["x"])
            sks.append(r["sk"])
            rnds.append(r["sk_rnd"])
            bodys.append(r["body"])
    return xs, sks, rnds, bodys


def build_ids(tok, x, sk, body, max_len=1024):
    p = tok.encode(x + MARK_SK + sk + MARK_BODY, add_special_tokens=False)
    b = tok.encode(body, add_special_tokens=False)
    ids = p + b
    if len(ids) > max_len:
        ids = ids[:max_len]
    labels = [-100] * len(p) + ids[len(p):]
    return ids, labels


def main():
    tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MDIR, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model.config.use_cache = False
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_k_v_proj", "out_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    xs, sks, rnds, bodys = load_samples(DATA)
    n_val = 30
    train = list(zip(xs[:n_val * 0 + len(xs) - n_val], sks[:len(xs) - n_val],
                     bodys[:len(xs) - n_val]))
    print("train:", len(train), flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=2e-4, weight_decay=0.01)
    BS, ACC = 2, 8
    steps = 300
    t0 = time.time()
    model.train()
    rng = np.random.RandomState(0)
    step = 0
    while step < steps:
        order = rng.permutation(len(train))
        for i in range(0, len(order) - BS + 1, BS):
            batch = [train[j] for j in order[i:i + BS]]
            ids_l, lab_l = [], []
            for x, sk, body in batch:
                ids, labels = build_ids(tok, x, sk, body)
                ids_l.append(ids)
                lab_l.append(labels)
            ml = max(len(x) for x in ids_l)
            pad = tok.pad_token_id or 2
            xb = torch.full((len(ids_l), ml), pad, dtype=torch.long)
            lb = torch.full((len(ids_l), ml), -100, dtype=torch.long)
            am = torch.zeros((len(ids_l), ml), dtype=torch.long)
            for j, (ii, ll) in enumerate(zip(ids_l, lab_l)):
                xb[j, :len(ii)] = torch.tensor(ii)
                lb[j, :len(ll)] = torch.tensor(ll)
                am[j, :len(ii)] = 1
            xb, lb, am = xb.to(DEV), lb.to(DEV), am.to(DEV)
            out = model(input_ids=xb, attention_mask=am, labels=lb)
            loss = out.loss / ACC
            loss.backward()
            if (step + 1) % ACC == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print("  step %d/%d loss=%.4f (%.0fs)" %
                      (step, steps, out.loss.item(), time.time() - t0), flush=True)
            if (step + 1) % 100 == 0 or step == steps - 1:
                model.save_pretrained(OUT)
                print("  ckpt saved @%d" % (step + 1), flush=True)
            if step >= steps:
                break
    model.save_pretrained(OUT)
    print("saved ->", OUT, flush=True)


if __name__ == "__main__":
    import numpy as np
    main()
