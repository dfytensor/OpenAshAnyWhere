#!/usr/bin/env python3
"""HRC-lite 评估: 三条件正文 NLL 对比 (LoRA 适配后模型).

条件:
  hrc    : x + 骨架要点 + 【正文】 (训练格式, 记忆桥模拟)
  no-sk  : x + 【正文】            (无记忆)
  rnd-sk : x + 随机要点 + 【正文】 (同预算无信息骨架)
  ar     : x + body_<i 逐位 teacher forcing (AR 上界)
"""
import sys, os, json, time
sys.path.insert(0, r"F:\Spark-X2.5-1.7B")
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DEV = "cuda"
MDIR = r"F:\Spark-X2.5-1.7B"
LORA = r"F:\夸克\hrc_validate\lora_hrc"
DATA = r"F:\夸克\hrc_validate\hrc_docs.jsonl"
MAXLEN = 1024
MARK_SK = "\n【要点】\n"
MARK_BODY = "\n【正文】\n"


@torch.no_grad()
def body_nll(model, tok, prefix_text, body_text):
    """body teacher-forcing NLL (mean/token). 前缀不计 loss; 超长截断保 body."""
    p = tok.encode(prefix_text, add_special_tokens=False)
    b = tok.encode(body_text, add_special_tokens=False)
    if len(p) + len(b) > MAXLEN:
        b = b[: max(64, MAXLEN - len(p))]
    ids = p + b
    x = torch.tensor([ids], device=DEV)
    logits = model(x).logits[0]
    lg = F.log_softmax(logits.float(), -1)
    t = torch.tensor(ids, device=DEV)
    nll = -lg.gather(1, t.unsqueeze(1)).squeeze(1)
    return nll[len(p) - 1:].mean().item()


@torch.no_grad()
def gen_sample(model, tok, prefix, n=200):
    ids = tok.encode(prefix, add_special_tokens=False)
    x = torch.tensor([ids[:MAXLEN - n]], device=DEV)
    out = model.generate(x, max_new_tokens=n, do_sample=False,
                         pad_token_id=tok.pad_token_id or 2)
    return tok.decode(out[0, x.shape[1]:], skip_special_tokens=True)


def main():
    tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        MDIR, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, LORA)
    model.eval()

    samples = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    val = samples[-30:]
    print("val samples:", len(val), flush=True)

    agg = {k: [] for k in ("hrc", "no_sk", "rnd_sk", "ar")}
    for i, r in enumerate(val):
        body = r["body"]
        p_hrc = r["x"] + MARK_SK + r["sk"] + MARK_BODY
        p_no = r["x"] + MARK_BODY
        p_rnd = r["x"] + MARK_SK + r["sk_rnd"] + MARK_BODY
        agg["hrc"].append(body_nll(model, tok, p_hrc, body))
        agg["no_sk"].append(body_nll(model, tok, p_no, body))
        agg["rnd_sk"].append(body_nll(model, tok, p_rnd, body))
        # AR 上界: 逐位 teacher forcing (标准 LM loss)
        ids = tok.encode(r["x"] + MARK_BODY + body, add_special_tokens=False)[-MAXLEN:]
        x = torch.tensor([ids], device=DEV)
        with torch.no_grad():
            lg = F.log_softmax(model(x).logits[0].float(), -1)
        t = torch.tensor(ids, device=DEV)
        nll = -lg.gather(1, t.unsqueeze(1)).squeeze(1)
        n_body = len(tok.encode(body, add_special_tokens=False))
        agg["ar"].append(nll[-n_body:].mean().item())
        if (i + 1) % 10 == 0:
            print("  %d/%d" % (i + 1, len(val)), flush=True)

    def m(k):
        v = agg[k]
        return sum(v) / len(v)

    print("\n===== HRC-lite 评估 (n=%d) =====" % len(val), flush=True)
    print("  hrc   (骨架条件)   = %.4f" % m("hrc"), flush=True)
    print("  no-sk (无记忆)     = %.4f" % m("no_sk"), flush=True)
    print("  rnd-sk (随机骨架)  = %.4f" % m("rnd_sk"), flush=True)
    print("  ar    (逐位上界)   = %.4f" % m("ar"), flush=True)
    print("\n  骨架增益 = no_sk - hrc = %.4f nats/tok" % (m("no_sk") - m("hrc")), flush=True)
    print("  距 AR 上界 = %.4f nats/tok" % (m("hrc") - m("ar")), flush=True)

    # 生成样例
    r = val[0]
    print("\n--- 生成样例 (hrc 条件, 贪婪 150 字) ---", flush=True)
    g = gen_sample(model, tok, r["x"] + MARK_SK + r["sk"] + MARK_BODY, 150)
    print(g[:200], flush=True)
    print("\n--- 真实正文开头 ---", flush=True)
    print(r["body"][:200], flush=True)


if __name__ == "__main__":
    main()
