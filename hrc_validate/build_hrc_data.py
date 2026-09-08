#!/usr/bin/env python3
"""构建 HRC-lite 数据: 长文档流 -> (x, 骨架句集, 正文) 三元组, 存 token ids."""
import sys, json, time
sys.path.insert(0, r"F:\Spark-X2.5-1.7B")
sys.path.insert(0, r"F:\Spark-X2.5-1.7B")
from transformers import AutoTokenizer
import numpy as np

TOKDIR = r"F:\Spark-X2.5-1.7B"
JSONL = r"F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl"
OUT = r"F:\夸克\hrc_validate\hrc_docs.jsonl"

X_LEN = 64
BODY_TGT = 1400          # 正文目标长度 (token)
SK_FRAC = 0.14           # 骨架预算 (占正文)
N_TRAIN, N_VAL = 260, 30


def main():
    tok = AutoTokenizer.from_pretrained(TOKDIR, trust_remote_code=True)
    texts = []
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            try:
                texts.append(json.loads(line)["text"].replace("\n", ""))
            except Exception:
                pass
    print("raw texts:", len(texts), flush=True)

    # 拼接成 ~1600 token 的文档流
    rng = np.random.RandomState(0)
    order = rng.permutation(len(texts))
    docs = []
    cur, cur_len = [], 0
    for i in order:
        t = texts[i]
        t_ids = tok.encode(t, add_special_tokens=False)
        if len(t_ids) > 300:
            continue
        cur.append(t)
        cur_len += len(t_ids) + 1
        if cur_len >= X_LEN + 1500:
            docs.append("\n".join(cur))
            cur, cur_len = [], 0
        if len(docs) >= N_TRAIN + N_VAL + 20:
            break
    print("docs built:", len(docs), flush=True)

    punc = tok.encode("。！？", add_special_tokens=False)
    n_out = 0
    t0 = time.time()
    with open(OUT, "w", encoding="utf-8") as fo:
        for di, doc in enumerate(docs):
            ids = tok.encode(doc, add_special_tokens=False)
            if len(ids) < 800:
                continue
            x = ids[:X_LEN]
            body = ids[X_LEN:X_LEN + BODY_TGT]
            L = len(body)
            # 句子边界 (body 内)
            sb = [i + 1 for i in range(1, L - 1) if body[i] in punc]
            if len(sb) < 6:
                continue
            # 切句: 句 j = (prev_sent_boundary, this_sent_boundary]
            sents = []
            prev = 0
            for b in sb:
                sents.append((prev, b))
                prev = b
            sents.append((prev, L))
            # 层级骨架: 均匀取 ~SK_FRAC 的句子, 取句首 ~8 token
            n_sk = max(4, int(len(sents) * 0.18))
            step = (len(sents) - 1) / (n_sk - 1) if n_sk > 1 else 1
            sk_parts, sk_len = [], 0
            for j in range(n_sk):
                a, b = sents[int(round(j * step))]
                head = body[a:min(a + 8, b)]
                sk_parts.append(head)
                sk_len += len(head)
            # 控制预算: 若超, 减少句
            while sk_len > L * SK_FRAC and n_sk > 3:
                n_sk -= 1
                step = (len(sents) - 1) / (n_sk - 1) if n_sk > 1 else 1
                sk_parts, sk_len = [], 0
                for j in range(n_sk):
                    a, b = sents[int(round(j * step))]
                    head = body[a:min(a + 8, b)]
                    sk_parts.append(head)
                    sk_len += len(head)
            # 随机骨架对照 (同预算): 随机句子
            ridx = np.random.RandomState(di).choice(len(sents), n_sk, replace=False)
            rnd_parts, rnd_len = [], 0
            for j in sorted(ridx):
                a, b = sents[j]
                head = body[a:min(a + 8, b)]
                rnd_parts.append(head)
                rnd_len += len(head)
            # 解码为文本
            def dec(parts):
                return "".join(tok.decode(p) for p in parts)
            rec = {
                "x": tok.decode(x),
                "sk": "\n".join("· " + dec([p]) for p in sk_parts),
                "sk_rnd": "\n".join("· " + dec([p]) for p in rnd_parts),
                "body": tok.decode(body),
                "n_body": L,
                "sk_frac": round(sk_len / L, 3),
            }
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1
            if n_out >= N_TRAIN + N_VAL:
                break
    print("written %d samples (%.0fs)" % (n_out, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
