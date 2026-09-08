#!/usr/bin/env python3
"""骨架因果效应零样本测试 (全部模型自生成, 无语料污染).

对每个文档:
  A = 模型在 [开头+真骨架] 条件下的续写
  B = 模型在 [开头无骨架] 条件下的续写
  div = NLL(A | 无骨架提示) - NLL(A | 有骨架提示)
     > 0 显著 => 骨架对自身生成的文本有因果支撑 (去掉骨架, 模型不这么写了)
对照: 随机骨架的 div_rnd. 层级骨架 div 应显著 > 随机骨架 div (H0 判据).
"""
import sys, json, time
sys.path.insert(0, r"F:\Spark-X2.5-1.7B")
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
MDIR = r"F:\Spark-X2.5-1.7B"
MAXLEN = 2048
GEN = 220
SYS = "你是续写助手。根据用户提供的开头与要点列表，续写完整正文。"


@torch.no_grad()
def generate(model, tok, x, sk_list):
    user = "开头：" + x + "\n要点：\n" + ("\n".join("· " + s for s in sk_list) if sk_list else "（无）")
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    pre = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    ids = torch.tensor([pre], device=DEV)
    out = model.generate(ids, max_new_tokens=GEN, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    gen = out[0][ids.shape[1]:]
    return pre, gen


@torch.no_grad()
def nll_of(model, pre, gen):
    """生成段在前缀条件下的逐 token NLL 均值."""
    ids = torch.tensor([list(pre) + list(gen)], device=DEV)
    logits = model(ids).logits[0]
    lg = F.log_softmax(logits.float(), -1)
    t = torch.tensor(list(pre) + list(gen), device=DEV)
    nll = -lg.gather(1, t.unsqueeze(1)).squeeze(1)
    return nll[len(pre):].mean().item()


def main():
    tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16,
                                             device_map="cuda", trust_remote_code=True)
    m.eval()
    samples = []
    with open(r"F:\夸克\hrc_validate\hrc_docs.jsonl", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    rows = []
    n = 0
    t0 = time.time()
    for r in samples:
        x = r["x"]
        sk_list = [s[2:] for s in r["sk"].split("\n") if s.startswith("· ")]
        rnd_list = [s[2:] for s in r["sk_rnd"].split("\n") if s.startswith("· ")]
        if len(sk_list) < 4:
            continue
        # 有骨架生成
        pre_sk, A = generate(m, tok, x, sk_list)
        # 无骨架生成
        pre_no, B = generate(m, tok, x, [])
        # 随机骨架生成
        pre_r, C = generate(m, tok, x, rnd_list)
        # 分歧: 把 A 放到无骨架条件下评估
        div_true = nll_of(m, pre_no, A) - nll_of(m, pre_sk, A)
        # 随机骨架对"随机骨架生成"的支撑 (对照)
        div_rnd = nll_of(m, pre_no, C) - nll_of(m, pre_r, C)
        rows.append({"div_true": div_true, "div_rnd": div_rnd,
                     "len": len(A)})
        n += 1
        if n % 5 == 0:
            dm = sum(r2["div_true"] for r2 in rows) / len(rows)
            dr = sum(r2["div_rnd"] for r2 in rows) / len(rows)
            print("  %d: div_true=%.3f div_rnd=%.3f (%.0fs)" % (n, dm, dr, time.time() - t0),
                  flush=True)

    dt = [r["div_true"] for r in rows]
    dr = [r["div_rnd"] for r in rows]
    print("\n===== 骨架因果效应 (n=%d, 模型自生成分歧) =====" % n, flush=True)
    print("  div(真骨架) = %.4f nats/tok" % (sum(dt) / len(dt)), flush=True)
    print("  div(随机骨架) = %.4f nats/tok" % (sum(dr) / len(dr)), flush=True)
    print("  判读: div>0 => 骨架有因果信息; 真骨架 >> 随机骨架 => 层级锚点特异有效",
          flush=True)
    json.dump(rows, open(r"F:\夸克\hrc_validate\causal_probe.json", "w"), indent=1)
    print("done %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
