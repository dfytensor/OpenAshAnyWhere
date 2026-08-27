#!/usr/bin/env python3
"""
Needle-in-a-Haystack test for long-range dependency
  1. Generate a long background text (from real novel)
  2. Insert a "needle" (specific fact) at position X%
  3. At the end, ask a question about the needle
  4. Check if the model can retrieve the answer
  5. Test at different depths (0%, 25%, 50%, 75%, 100%) and context lengths
"""
import os, sys, math, json, torch, torch.nn.functional as F, time, random

ROOT = r"F:\OpenASH2605"
BENCH = os.path.join(ROOT, "experiment_openash_vs_wdlm", "bench")
sys.path.insert(0, ROOT); sys.path.insert(0, BENCH); os.chdir(ROOT)

from open_ash import OpenASH
from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from open_ash_infer import _sp

DEV = "cuda"
CHUNK = 64
STATE_CAP = 150
STATE_DECAY = 0.97

voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1
sp = _sp(voc)

m30 = OpenASH(vs, hidden_size=432, num_heads=8, num_layers=8, model_flag="train")
m30.load_state_dict(torch.load(os.path.join(ROOT, "train_30m_cap_decay", "openash30m_cd_sft_final.pth"), map_location=DEV)["model"])
m30.to(DEV).eval()

m85 = OpenASH(vs, hidden_size=768, num_heads=8, num_layers=12, model_flag="train")
m85.load_state_dict(torch.load(os.path.join(BENCH, "full_sft_768_12.pth"), map_location=DEV))
m85.to(DEV).eval()

for m in [m30, m85]:
    for _ in range(3):
        with torch.no_grad(): m(torch.randint(1, 100, (1, 128), device=DEV), state=None)
torch.cuda.synchronize()
print("Models ready.\n")


NEEDLES = [
    ("我的手机密码是8473，请记住这个密码。", "我的手机密码是什么？", "8473"),
    ("今天是2026年6月12日，天气晴朗。", "今天的日期是哪天？", "2026"),
    ("小明家的猫叫橘子，是一只橘色的胖猫。", "小明家的猫叫什么名字？", "橘子"),
    ("这本书的作者是李华，出版社是清华大学出版社。", "这本书的作者是谁？", "李华"),
    ("密码箱的密码是9527，千万不要忘记。", "密码箱的密码是多少？", "9527"),
    ("会议室在三楼302房间，下午两点开会。", "会议室在哪个房间？", "302"),
    ("张三的银行卡号是6225880137123456。", "张三的银行卡号后四位是多少？", "3456"),
    ("钥匙藏在门口花盆下面第三个位置。", "钥匙藏在哪里？", "花盆"),
]


def generate_response(model, ids, max_new=80, use_cd=False):
    n_layers = len(model.decoder_layers)
    with torch.no_grad():
        states = [None] * n_layers
        generated = ids
        for step in range(max_new):
            ctx = generated[:, -768:]
            if ctx.size(1) <= CHUNK:
                c = ctx
            else:
                c = ctx[:, -CHUNK:]

            h = model.em(c)
            for i, layer in enumerate(model.decoder_layers):
                h2, s = layer(h, states[i])
                h = h2 + h
                states[i] = s
                if use_cd and s is not None:
                    sn = s.norm()
                    if sn > STATE_CAP:
                        s = s * (STATE_CAP / sn)
                    states[i] = s * STATE_DECAY

            logits = model.head_score(h)[:, -1, :] / 0.7
            v, _ = torch.topk(logits, 40)
            logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
            probs = F.softmax(logits, dim=-1)
            nt = torch.multinomial(probs, 1)
            generated = torch.cat([generated, nt], dim=1)
            tok_id = nt.item()
            if tok_id == sp["im_end"]:
                break
    return generated[0].tolist()


def needle_test(model, model_name, use_cd=False):
    novel_path = os.path.join(r"F:\小说\女生小说", "傲世九重天-风凌天下.txt")
    with open(novel_path, encoding="utf-8", errors="ignore") as f:
        novel_text = f.read(1000000)

    novel_tokens = voc.encode(novel_text)
    print(f"  Novel tokens: {len(novel_tokens):,}", flush=True)

    context_lens = [512, 2048, 8192, 32768]
    depths = [0, 25, 50, 75, 100]
    n_trials = 3

    results = {}

    for ctx_len in context_lens:
        if ctx_len > len(novel_tokens):
            continue
        results[ctx_len] = {}

        for depth_pct in depths:
            scores = []
            for trial in range(n_trials):
                needle_stmt, question, answer = random.choice(NEEDLES)
                needle_ids = voc.encode(needle_stmt)
                question_ids = voc.encode(question)

                insert_pos = int(ctx_len * depth_pct / 100)
                insert_pos = max(0, min(insert_pos, ctx_len - len(needle_ids)))

                before = novel_tokens[:insert_pos]
                after = novel_tokens[insert_pos:ctx_len - len(needle_ids)]

                qa_prefix = [sp["im_start"], sp["user"]] + question_ids + [sp["im_end"]]
                qa_prefix += [sp["im_start"], sp["agent"]]

                full_ids = before + needle_ids + after + qa_prefix
                if len(full_ids) > ctx_len:
                    full_ids = full_ids[:ctx_len]
                while len(full_ids) < 64:
                    full_ids.append(0)

                x = torch.tensor([full_ids], dtype=torch.long, device=DEV).clamp(0, vs - 1)
                resp_ids = generate_response(model, x, max_new=60, use_cd=use_cd)
                resp_text = voc.decode(resp_ids[len(full_ids):])
                resp_text = resp_text.strip()[:200]

                hit = 1 if answer in resp_text else 0
                scores.append(hit)

                label = f"{ctx_len//1024}K@{depth_pct}%" if ctx_len >= 1024 else f"{ctx_len}@{depth_pct}%"
                mark = "HIT" if hit else "MISS"
                short_resp = resp_text[:50].replace('\n', ' ')
                print(f"    [{model_name}] {label:>12} trial{trial+1} {mark}  ans={answer}  resp={short_resp}", flush=True)

            accuracy = sum(scores) / len(scores)
            results[ctx_len][depth_pct] = accuracy

    return results


def print_heatmap(results, model_name):
    print(f"\n  [{model_name}] Needle-in-a-Haystack Accuracy")
    print(f"  {'Depth':>8}", end="")
    ctx_lens = sorted(results.keys())
    for cl in ctx_lens:
        lb = f"{cl//1024}K" if cl >= 1024 else str(cl)
        print(f"  {lb:>8}", end="")
    print()
    print(f"  {'-'*(8 + 9*len(ctx_lens))}")

    for d in [0, 25, 50, 75, 100]:
        print(f"  {str(d)+'%':>8}", end="")
        for cl in ctx_lens:
            acc = results.get(cl, {}).get(d, 0)
            print(f"  {acc:>7.0%}", end="")
        print()
    print()


# ============================================================
# Run tests
# ============================================================
print("=" * 80)
print("  Needle-in-a-Haystack Test")
print("  Insert a fact into a novel, ask about it at the end")
print("=" * 80)

print("\n--- Testing 30M-cd (cap+decay trained) ---")
r30 = needle_test(m30, "30M-cd", use_cd=True)
print_heatmap(r30, "30M-cd")

print("\n--- Testing 85M+cd (inference-time cap+decay) ---")
r85 = needle_test(m85, "85M+cd", use_cd=True)
print_heatmap(r85, "85M+cd")

print("\n--- Testing 85M baseline (no cap+decay) ---")
r85b = needle_test(m85, "85M-base", use_cd=False)
print_heatmap(r85b, "85M-base")

print("Done.")
