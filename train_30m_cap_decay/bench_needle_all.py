import os, sys, math, torch, torch.nn.functional as F, time, random
ROOT = r"F:\OpenASH2605"
BENCH = os.path.join(ROOT, "experiment_openash_vs_wdlm", "bench")
sys.path.insert(0, ROOT); sys.path.insert(0, BENCH); os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "wdlm_verification"))

from open_ash import OpenASH
from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from open_ash_infer import _sp
from wdlm_neural import WaveDynamicsLanguageModel

DEV = "cuda"
CHUNK = 64
CAP = 150
DECAY = 0.97

voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1
sp = _sp(voc)

m30 = OpenASH(vs, hidden_size=432, num_heads=8, num_layers=8, model_flag="train")
m30.load_state_dict(torch.load(os.path.join(ROOT,"train_30m_cap_decay","openash30m_cd_sft_final.pth"), map_location=DEV)["model"])
m30.to(DEV).eval()

m58 = OpenASH(vs, hidden_size=640, num_heads=8, num_layers=10, model_flag="train")
m58.load_state_dict(torch.load(os.path.join(BENCH,"openash60m_sft_final.pth"), map_location=DEV)["model"])
m58.to(DEV).eval()

m85 = OpenASH(vs, hidden_size=768, num_heads=8, num_layers=12, model_flag="train")
m85.load_state_dict(torch.load(os.path.join(BENCH,"full_sft_768_12.pth"), map_location=DEV))
m85.to(DEV).eval()

wm = WaveDynamicsLanguageModel(vs, hidden_dim=512, num_layers=10)
_ck = torch.load(os.path.join(BENCH, "wdlm60m_sft_final.pth"), map_location=DEV)
wm.load_state_dict(_ck["model"] if "model" in _ck else _ck)
wm.to(DEV).eval()

for m in [m30, m58, m85, wm]:
    for _ in range(3):
        with torch.no_grad(): m(torch.randint(1,100,(1,128),device=DEV), state=None)
torch.cuda.synchronize()
print("Models ready: OA-30M-cd, OA-58M, OA-85M, WDLM-60M\n")

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


def oa_generate(model, ids, max_new=60, use_cd=False):
    nl = len(model.decoder_layers)
    with torch.no_grad():
        states = [None] * nl
        generated = ids
        for _ in range(max_new):
            ctx = generated[:, -768:]
            c = ctx[:, -CHUNK:] if ctx.size(1) > CHUNK else ctx
            h = model.em(c)
            for i, la in enumerate(model.decoder_layers):
                h2, s = la(h, states[i])
                h = h2 + h
                states[i] = s
                if use_cd and s is not None:
                    sn = s.norm()
                    if sn > CAP: s = s * (CAP / sn)
                    states[i] = s * DECAY
            logits = model.head_score(h)[:, -1, :] / 0.7
            v, _ = torch.topk(logits, 40)
            logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
            nt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            generated = torch.cat([generated, nt], dim=1)
            if nt.item() == sp["im_end"]: break
    return generated[0].tolist()


def wdlm_generate(model, ids, max_new=60, use_cd=False):
    with torch.no_grad():
        state = None
        generated = ids
        for _ in range(max_new):
            ctx = generated[:, -768:]
            out, state_out = model(ctx, state=state)
            if isinstance(state_out, list) and use_cd:
                for i in range(len(state_out)):
                    if state_out[i] is not None:
                        sn = state_out[i].norm()
                        if sn > CAP: state_out[i] = state_out[i] * (CAP / sn)
                        state_out[i] = state_out[i] * DECAY
            state = [s.detach() if s is not None else None for s in state_out]
            logits = out[:, -1, :] / 0.7
            v, _ = torch.topk(logits, 40)
            logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
            nt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            generated = torch.cat([generated, nt], dim=1)
            if nt.item() == 2: break
    return generated[0].tolist()


def run_test(model_name, model, gen_fn, use_cd, n_trials=10):
    novel_path = os.path.join(r"F:\小说\女生小说", "傲世九重天-风凌天下.txt")
    with open(novel_path, encoding="utf-8", errors="ignore") as f:
        novel_text = f.read(1000000)
    novel_tokens = voc.encode(novel_text)

    ctx_lens = [512, 2048, 8192]
    depths = [0, 50, 100]
    results = {}

    for cl in ctx_lens:
        results[cl] = {}
        for d in depths:
            scores = []
            for trial in range(n_trials):
                needle_stmt, question, answer = random.choice(NEEDLES)
                needle_ids = voc.encode(needle_stmt)
                question_ids = voc.encode(question)
                insert_pos = int(cl * d / 100)
                insert_pos = max(0, min(insert_pos, cl - len(needle_ids)))
                before = novel_tokens[:insert_pos]
                after = novel_tokens[insert_pos:cl - len(needle_ids)]
                qa_prefix = [sp["im_start"], sp["user"]] + question_ids + [sp["im_end"]]
                qa_prefix += [sp["im_start"], sp["agent"]]
                full_ids = before + needle_ids + after + qa_prefix
                if len(full_ids) > cl: full_ids = full_ids[:cl]
                while len(full_ids) < 64: full_ids.append(0)

                x = torch.tensor([full_ids], dtype=torch.long, device=DEV).clamp(0, vs-1)
                resp_ids = gen_fn(model, x, max_new=60, use_cd=use_cd)
                resp_text = voc.decode(resp_ids[len(full_ids):]).strip()[:200]
                hit = 1 if answer in resp_text else 0
                scores.append(hit)
                mark = "HIT" if hit else "miss"
                lb = "{}K@{}%".format(cl//1024 if cl>=1024 else cl, d)
                print("    [{:<12}] {:>10} t{:>2} {} ans={:<6} resp={:.50}".format(
                    model_name, lb, trial+1, mark, answer, resp_text.replace('\n',' ')))
                sys.stdout.flush()
            results[cl][d] = sum(scores) / len(scores)
    return results


models_config = [
    ("30M-cd",      m30, oa_generate,   True),
    ("58M+cd",      m58, oa_generate,   True),
    ("85M+cd",      m85, oa_generate,   True),
    ("85M-base",    m85, oa_generate,   False),
    ("WDLM+cd",     wm,  wdlm_generate, True),
    ("WDLM-base",   wm,  wdlm_generate, False),
]

all_results = {}
for name, model, gen_fn, use_cd in models_config:
    print("\n--- {} ---".format(name))
    r = run_test(name, model, gen_fn, use_cd, n_trials=10)
    all_results[name] = r

print("\n" + "=" * 90)
print("  Needle-in-a-Haystack Summary ({} trials each, answer check in response)".format(10))
print("=" * 90)
ctx_lens = [512, 2048, 8192]
print("  {:>12}  {:>8}  {:>8}  {:>8}".format("Model", "512", "2K", "8K"))
print("  " + "-" * 44)
for name, _, _, _ in models_config:
    r = all_results[name]
    row = "  {:>12}".format(name)
    for cl in ctx_lens:
        vals = r.get(cl, {})
        avg = sum(vals.values()) / max(len(vals), 1)
        row += "  {:>7.0%}".format(avg)
    print(row)

print("\n  Breakdown by depth:")
print("  {:>12}  {:>5}  {:>8}  {:>8}  {:>8}".format("Model", "Depth", "512", "2K", "8K"))
print("  " + "-" * 50)
for name, _, _, _ in models_config:
    r = all_results[name]
    for d in [0, 50, 100]:
        row = "  {:>12}  {:>4}%".format(name if d==0 else "", d)
        for cl in ctx_lens:
            acc = r.get(cl, {}).get(d, 0)
            row += "  {:>7.0%}".format(acc)
        print(row)
    print()

print("Done.")
