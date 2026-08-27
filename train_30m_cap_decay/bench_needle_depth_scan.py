"""
Fine-grained needle depth scan to find long-term dependency limit.
Test depths: 100%, 95%, 90%, 85%, 80%, 75%, 70%, 60%, 50%, 40%, 30%, 20%, 10%
At context lengths: 512, 768, 1024, 2048
"""
import os, sys, math, torch, torch.nn.functional as F, time, random
ROOT = r"F:\OpenASH2605"
sys.path.insert(0, ROOT); os.chdir(ROOT)
from open_ash import OpenASH
from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from open_ash_infer import _sp

DEV = "cuda"
CHUNK = 64
CAP = 150
DECAY = 0.97

voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1
sp = _sp(voc)

model = OpenASH(vs, hidden_size=432, num_heads=8, num_layers=8, model_flag="train")
model.load_state_dict(torch.load(os.path.join(ROOT, "train_30m_cap_decay", "openash30m_cd_needle_final.pth"),
                                  map_location=DEV)["model"])
model.to(DEV).eval()
with torch.no_grad():
    model(torch.randint(1, 100, (1, 128), device=DEV), state=None)
torch.cuda.synchronize()
print("30M-cd-needle model loaded\n")

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

novel_path = os.path.join(r"F:\小说\女生小说", "傲世九重天-风凌天下.txt")
with open(novel_path, encoding="utf-8", errors="ignore") as f:
    novel_text = f.read(2000000)
novel_tokens = voc.encode(novel_text)

CTX_LENS = [512, 768, 1024, 2048]
DEPTHS = [100, 95, 90, 85, 80, 75, 70, 60, 50, 40, 30, 20, 10]
N_TRIALS = 15
NL = len(model.decoder_layers)

results = {}

for cl in CTX_LENS:
    results[cl] = {}
    for d in DEPTHS:
        hits = 0
        for trial in range(N_TRIALS):
            needle_stmt, question, answer = random.choice(NEEDLES)
            needle_ids = voc.encode(needle_stmt)
            question_ids = voc.encode(question)

            max_ctx = cl - len(needle_ids) - len(question_ids) - 20
            if max_ctx < 20:
                max_ctx = 20
            context_ids = novel_tokens[:max_ctx]
            insert_pos = int(len(context_ids) * d / 100)
            insert_pos = max(0, min(insert_pos, len(context_ids) - 1))

            before = context_ids[:insert_pos]
            after = context_ids[insert_pos:]

            qa_prefix = [sp["im_start"], sp["user"]] + question_ids + [sp["im_end"]]
            qa_prefix += [sp["im_start"], sp["agent"]]
            full_ids = before + needle_ids + after + qa_prefix

            if len(full_ids) > cl:
                full_ids = full_ids[:cl]
            while len(full_ids) < 64:
                full_ids.append(0)

            needle_pos = len(before)
            total_ctx = len(full_ids)
            dist_to_end = total_ctx - needle_pos

            x = torch.tensor([full_ids], dtype=torch.long, device=DEV).clamp(0, vs - 1)

            with torch.no_grad():
                states = [None] * NL
                generated = x
                for _ in range(60):
                    ctx = generated[:, -768:]
                    c = ctx[:, -CHUNK:] if ctx.size(1) > CHUNK else ctx
                    h = model.em(c)
                    for i, la in enumerate(model.decoder_layers):
                        h2, s = la(h, states[i])
                        h = h2 + h
                        states[i] = s
                        if s is not None:
                            sn = s.norm()
                            if sn > CAP:
                                s = s * (CAP / sn)
                            states[i] = s * DECAY
                    logits = model.head_score(h)[:, -1, :] / 0.7
                    v, _ = torch.topk(logits, 40)
                    logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
                    nt = torch.multinomial(F.softmax(logits, dim=-1), 1)
                    generated = torch.cat([generated, nt], dim=1)
                    if nt.item() == sp["im_end"]:
                        break

                resp_text = voc.decode(generated[0].tolist()[total_ctx:]).strip()[:200]
                hit = 1 if answer in resp_text else 0
                hits += hit

        acc = hits / N_TRIALS
        results[cl][d] = acc
        label = "{}K".format(cl // 1024) if cl >= 1024 else str(cl)
        print("  {:>5} @{:>3}%  ACC={:>5.0%}  ({}/{})  dist_to_end={}".format(
            label, d, acc, hits, N_TRIALS, dist_to_end))
        sys.stdout.flush()
    print()

print("\n" + "=" * 80)
print("  Dependency Limit Scan — Accuracy by Depth")
print("=" * 80)

header = "{:>5}".format("Depth")
for cl in CTX_LENS:
    label = "{}K".format(cl // 1024) if cl >= 1024 else str(cl)
    header += "  {:>6}".format(label)
print(header)
print("-" * (5 + 8 * len(CTX_LENS)))

for d in DEPTHS:
    row = "{:>4}%".format(d)
    for cl in CTX_LENS:
        acc = results[cl].get(d, 0)
        bar = "#" * int(acc * 10)
        row += "  {:>5.0%}".format(acc)
    print(row)

print()
print("  'Depth' = needle position as % of context (100% = near question, 0% = at start)")
print("  acc threshold ~20% suggests effective retrieval distance")
print()

for cl in CTX_LENS:
    cutoff_depth = None
    for d in DEPTHS:
        if results[cl][d] < 0.20:
            cutoff_depth = d
            break
    if cutoff_depth:
        effective_dist = int(cl * cutoff_depth / 100)
        label = "{}K".format(cl // 1024) if cl >= 1024 else str(cl)
        print("  {}: effective retrieval < {} tokens (depth <{}% drops below 20%)".format(
            label, effective_dist, cutoff_depth))
    else:
        label = "{}K".format(cl // 1024) if cl >= 1024 else str(cl)
        print("  {}: full context retrieval > 20% at all tested depths".format(label))

print("\nDone.")
