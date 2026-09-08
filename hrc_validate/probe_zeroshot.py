#!/usr/bin/env python3
"""零样本探针: chat 模板格式下, 骨架(要点)是否降低正文 NLL? (Spark 基模型, 无 LoRA)"""
import sys, json, time
sys.path.insert(0, r"F:\Spark-X2.5-1.7B")
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
MDIR = r"F:\Spark-X2.5-1.7B"
MAXLEN = 2048
SYS = "你是续写助手。根据用户提供的开头与要点列表，还原要点之间的完整正文。"


@torch.no_grad()
def body_nll(model, tok, x, sk_list, body, skip_local=0):
    """条件 = chat(系统+用户[开头+要点]) + 助手正文 teacher forcing.
    返回正文逐位 NLL 均值 (跳过正文前 skip_local 个 token, 模拟'已并行解出'的左邻)."""
    user = "开头：" + x + "\n要点：\n" + "\n".join("· " + s for s in sk_list)
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user},
            {"role": "assistant", "content": body}]
    full = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    # 找 assistant 部分起点: 用无 assistant 的前缀长度
    pre = tok.apply_chat_template(msgs[:2], tokenize=True, add_generation_prompt=True)
    off = len(pre) - 1                       # assistant 首 token 的 logits 位置
    ids = torch.tensor([full], device=DEV)
    logits = model(ids).logits[0]
    lg = F.log_softmax(logits.float(), -1)
    t = torch.tensor(full, device=DEV)
    nll = -lg.gather(1, t.unsqueeze(1)).squeeze(1)
    seg = nll[off + skip_local: off + len(body)]
    return seg.mean().item()


def main():
    tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16,
                                             device_map="cuda", trust_remote_code=True)
    m.eval()
    samples = []
    with open(r"F:\夸克\hrc_validate\hrc_docs.jsonl", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    # 只保留正文 <= 400 token 的样本 (总长可控)
    kept = []
    for r in samples:
        bids = tok.encode(r["body"], add_special_tokens=False)[:350]
        if len(bids) < 96:
            continue
        body = tok.decode(bids)
        kept.append((r, body, len(bids)))
    print("可用样本:", len(kept), flush=True)

    agg = {k: [] for k in ("hrc", "no_sk", "rnd_sk")}
    n = 0
    for r, body, blen in kept[:25]:
        sk_list = [s[2:] for s in r["sk"].split("\n") if s.startswith("· ")]
        rnd_list = [s[2:] for s in r["sk_rnd"].split("\n") if s.startswith("· ")]
        a = body_nll(m, tok, r["x"], sk_list, body)
        b = body_nll(m, tok, r["x"], [], body)
        c = body_nll(m, tok, r["x"], rnd_list, body)
        agg["hrc"].append(a)
        agg["no_sk"].append(b)
        agg["rnd_sk"].append(c)
        n += 1
        if n % 10 == 0:
            print("  %d: hrc=%.3f no=%.3f rnd=%.3f" % (n, a, b, c), flush=True)

    def mm(k):
        v = agg[k]
        return sum(v) / len(v)
    print("\n===== 零样本骨架增益 (n=%d, Spark-1.7B chat 格式) =====" % n, flush=True)
    print("  NLL(正文|开头+要点骨架) = %.4f" % mm("hrc"), flush=True)
    print("  NLL(正文|开头)          = %.4f" % mm("no_sk"), flush=True)
    print("  NLL(正文|开头+随机要点) = %.4f" % mm("rnd_sk"), flush=True)
    print("  骨架增益 = %.4f nats/tok (%.1f%% of no_sk)" %
          (mm("no_sk") - mm("hrc"), (mm("no_sk") - mm("hrc")) / mm("no_sk") * 100), flush=True)
    print("  随机骨架增益 = %.4f" % (mm("no_sk") - mm("rnd_sk")), flush=True)


if __name__ == "__main__":
    main()
