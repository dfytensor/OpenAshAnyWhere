#!/usr/bin/env python3
"""Inference quality/speed check: baseline vs fixed models"""
import os, sys, time, math, json, torch, torch.nn.functional as F
from collections import Counter

ROOT = r"F:\OpenASH2605"
BENCH = os.path.join(ROOT, "experiment_openash_vs_wdlm", "bench")
sys.path.insert(0, ROOT); sys.path.insert(0, BENCH); os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "wdlm_verification"))

from open_ash import OpenASH
from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from open_ash_infer import _sp, sample_next_token, build_user_prompt, format_response
from wdlm_neural import WaveDynamicsLanguageModel

DEV = "cuda"
voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1
sp = _sp(voc)
stop = {sp["im_end"], sp["pad"]}
SFT_DATA = os.path.join(ROOT, "minimind_data", "sft_t2t_mini.jsonl")
CHUNK = 64

oa58 = OpenASH(vs, hidden_size=640, num_heads=8, num_layers=10, model_flag="train")
oa58.load_state_dict(torch.load(os.path.join(BENCH, "openash60m_sft_final.pth"), map_location=DEV)["model"])
oa58.to(DEV).eval()

oa85 = OpenASH(vs, hidden_size=768, num_heads=8, num_layers=12, model_flag="train")
oa85.load_state_dict(torch.load(os.path.join(BENCH, "full_sft_768_12.pth"), map_location=DEV))
oa85.to(DEV).eval()

wm60 = WaveDynamicsLanguageModel(vs, hidden_dim=512, num_layers=10)
_ck = torch.load(os.path.join(BENCH, "wdlm60m_sft_final.pth"), map_location=DEV)
wm60.load_state_dict(_ck["model"] if "model" in _ck else _ck)
wm60.to(DEV).eval()

# Norm caps
WM_CAP = {i: 200 for i in range(10)}
OA58_CAP = {0: 96, 1: 84, 2: 107, 3: 158, 4: 341, 5: 150, 6: 145, 7: 96, 8: 66, 9: 103}
OA85_CAP = {i: 200 for i in range(12)}

# Warmup
for m in [oa58, oa85, wm60]:
    for _ in range(5):
        x = torch.randint(1, 100, (1, 128), device=DEV)
        with torch.no_grad(): m(x, state=None)
torch.cuda.synchronize()
print("Ready.\n")

GEN_CFG = dict(temperature=0.5, top_k=30, top_p=0.85, repetition_penalty=1.35)
PROMPTS = [
    "你好，请介绍一下你自己。",
    "什么是人工智能？",
    "请用Python写一个冒泡排序算法。",
    "解释量子计算的基本原理。",
    "中国的首都是哪里？",
    "写一首关于春天的诗。",
    "1+1等于几？",
    "请列举五种水果的名称。",
]


def gen_with_cap(model, pid, max_new=150, state_cap=None, **kw):
    """Generate with state norm cap intervention."""
    dev = next(model.parameters()).device
    x = torch.tensor([pid], dtype=torch.long).to(dev)
    if x.size(1) > 1024: x = x[:, -1024:]
    ids = []
    is_oa = hasattr(model, 'decoder_layers')
    layers = model.decoder_layers if is_oa else model.layers
    n_layers = len(layers)
    state_cap = state_cap or {}

    with torch.no_grad():
        states = [None] * n_layers
        # Process prompt
        for c0 in range(0, x.size(1), CHUNK):
            c = x[:, c0:c0+CHUNK]
            if is_oa:
                h = model.em(c)
                for i, layer in enumerate(layers):
                    h2, s = layer(h, states[i])
                    h = h2 + h
                    if i in state_cap and s is not None:
                        sn = s.norm()
                        if sn > state_cap[i]: s = s * (state_cap[i] / sn)
                    states[i] = s.detach() if s is not None else None
                logits = model.head_score(h)
            else:
                h = model.encoder(c)
                for i, layer in enumerate(layers):
                    h, s = layer(h, states[i])
                    if i in state_cap:
                        sn = s.norm()
                        if sn > state_cap[i]: s = s * (state_cap[i] / sn)
                    states[i] = s.detach()
                logits = model.head(h)

        # Generate tokens
        last_logits = logits[0, -1]
        for _ in range(max_new):
            nid = sample_next_token(last_logits, ids, **kw)
            if nid in stop: break
            ids.append(nid)

            c = torch.tensor([[nid]], dtype=torch.long, device=dev)
            if is_oa:
                h = model.em(c)
                for i, layer in enumerate(layers):
                    h2, s = layer(h, states[i])
                    h = h2 + h
                    if i in state_cap and s is not None:
                        sn = s.norm()
                        if sn > state_cap[i]: s = s * (state_cap[i] / sn)
                    states[i] = s.detach() if s is not None else None
                last_logits = model.head_score(h)[0, -1]
            else:
                h = model.encoder(c)
                for i, layer in enumerate(layers):
                    h, s = layer(h, states[i])
                    if i in state_cap:
                        sn = s.norm()
                        if sn > state_cap[i]: s = s * (state_cap[i] / sn)
                    states[i] = s.detach()
                last_logits = model.head(h)[0, -1]
    return ids


def quality(ids):
    if not ids: return {"unique": 0, "rep3": 1, "entropy": 0}
    u = len(set(ids)) / len(ids)
    tg = [(ids[i], ids[i+1], ids[i+2]) for i in range(len(ids)-2)]
    r3 = 1 - len(set(tg)) / max(len(tg), 1)
    c = Counter(ids)
    p = [v/len(ids) for v in c.values()]
    e = -sum(x * math.log(x+1e-10) for x in p)
    return {"unique": u, "rep3": r3, "entropy": e}


def ppl_short(model, path, seq_len=512, n=50, state_cap=None):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                convs = obj.get("conversations", [])
                ids = []
                for msg in convs:
                    r = msg.get("role",""); ct = msg.get("content","")
                    if r == "user": ids += [sp["im_start"], sp["user"]] + voc.encode(ct) + [sp["im_end"]]
                    elif r == "assistant": ids += [sp["im_start"], sp["agent"]] + voc.encode(ct) + [sp["im_end"]]
                if len(ids) >= seq_len: samples.append(torch.tensor(ids[:seq_len], dtype=torch.long))
            except: pass
            if len(samples) >= n: break
    if not samples: return float('inf')
    is_oa = hasattr(model, 'decoder_layers')
    layers = model.decoder_layers if is_oa else model.layers
    state_cap = state_cap or {}
    nll, ntok = 0, 0
    with torch.no_grad():
        for s in samples:
            x, t = s[:-1].unsqueeze(0).to(DEV), s[1:].unsqueeze(0).to(DEV)
            states = [None] * len(layers)
            cl = []
            for c0 in range(0, x.size(1), CHUNK):
                c = x[:, c0:c0+CHUNK]
                if is_oa:
                    h = model.em(c)
                    for i, layer in enumerate(layers):
                        h2, st = layer(h, states[i])
                        h = h2 + h
                        if i in state_cap and st is not None:
                            sn = st.norm()
                            if sn > state_cap[i]: st = st * (state_cap[i] / sn)
                        states[i] = st.detach() if st is not None else None
                    cl.append(model.head_score(h))
                else:
                    h = model.encoder(c)
                    for i, layer in enumerate(layers):
                        h, st = layer(h, states[i])
                        if i in state_cap:
                            sn = st.norm()
                            if sn > state_cap[i]: st = st * (state_cap[i] / sn)
                        states[i] = st.detach()
                    cl.append(model.head(h))
            clo = torch.cat(cl, dim=1)
            nll += F.cross_entropy(clo.reshape(-1, clo.size(-1)), t.reshape(-1), ignore_index=0, reduction="sum").item()
            ntok += max((t != 0).sum().item(), 1)
    return math.exp(nll / ntok)


configs = [
    ("WM-base",  wm60, None),
    ("WM-fix",   wm60, WM_CAP),
    ("OA58-base", oa58, None),
    ("OA58-fix",  oa58, OA58_CAP),
    ("OA85-base", oa85, None),
    ("OA85-fix",  oa85, OA85_CAP),
]

# ============================================================
# 1. Generation speed
# ============================================================
print("=" * 75)
print("  1. Generation Speed (state, 100 tok, median of 3)")
print("=" * 75)
for label, model, cap in configs:
    pid = build_user_prompt(voc, "你好")
    times = []
    for _ in range(3):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        ids = gen_with_cap(model, pid, 100, state_cap=cap, **GEN_CFG)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        times.append(len(ids) / dt if dt > 0 else 0)
    med = sorted(times)[1]
    print(f"  {label:>12}  {med:.1f} tok/s")

# ============================================================
# 2. PPL (short, seq=512, 50 samples)
# ============================================================
print()
print("=" * 75)
print("  2. PPL (SFT, seq=512, 50 samples)")
print("=" * 75)
for label, model, cap in configs:
    p = ppl_short(model, SFT_DATA, 512, 50, state_cap=cap)
    print(f"  {label:>12}  {p:.2f}")

# ============================================================
# 3. Generation Quality
# ============================================================
print()
print("=" * 75)
print("  3. Generation Quality (150 tok)")
print("=" * 75)
print(f"  {'Model':>12}  {'Prompt':>20}  {'Tok':>4}  {'Uni%':>6}  {'3g%':>5}  {'Ent':>5}")
print(f"  {'-'*58}")

for label, model, cap in configs:
    for text in ["请列举五种水果", "写春天的诗", "解释引力", "猫的特征"]:
        pid = build_user_prompt(voc, text)
        ids = gen_with_cap(model, pid, 150, state_cap=cap, **GEN_CFG)
        q = quality(ids)
        print(f"  {label:>12}  {text[:20]:>20}  {len(ids):>4}  {q['unique']*100:>5.1f}%  {q['rep3']*100:>4.1f}%  {q['entropy']:>5.2f}")

# ============================================================
# 4. Sample outputs
# ============================================================
print()
print("=" * 75)
print("  4. Sample Outputs")
print("=" * 75)
for text in ["你好，介绍一下你自己。", "什么是人工智能？", "冒泡排序算法。"]:
    pid = build_user_prompt(voc, text)
    for label, model, cap in configs:
        ids = gen_with_cap(model, pid, 150, state_cap=cap, **GEN_CFG)
        out = format_response(voc, ids).get('content', '')[:100].replace('\n', ' ')
        print(f"  [{label:>12}] {out}")
    print()

# ============================================================
# 5. Long context quality (seq=4096 top-k)
# ============================================================
print("=" * 75)
print("  5. Long-Context Quality (seq=4096 top-k predictions)")
print("=" * 75)
all_ids_long = []
with open(SFT_DATA, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            obj = json.loads(line)
            convs = obj.get("conversations", [])
            ids = []
            for msg in convs:
                r = msg.get("role",""); ct = msg.get("content","")
                if r == "user": ids += [sp["im_start"], sp["user"]] + voc.encode(ct) + [sp["im_end"]]
                elif r == "assistant": ids += [sp["im_start"], sp["agent"]] + voc.encode(ct) + [sp["im_end"]]
            if ids: all_ids_long.extend(ids)
        except: pass
        if len(all_ids_long) >= 4096: break

ctx_ids = all_ids_long[:4096]
# Take last 64 tokens as target
tgt_ids = ctx_ids[-64:]

for label, model, cap in configs:
    is_oa = hasattr(model, 'decoder_layers')
    layers = model.decoder_layers if is_oa else model.layers
    x = torch.tensor([ctx_ids[:-64]], dtype=torch.long).to(DEV)
    t = torch.tensor([tgt_ids], dtype=torch.long).to(DEV)

    with torch.no_grad():
        states = [None] * len(layers)
        for c0 in range(0, x.size(1), CHUNK):
            c = x[:, c0:c0+CHUNK]
            if is_oa:
                h = model.em(c)
                for i, layer in enumerate(layers):
                    h2, s = layer(h, states[i])
                    h = h2 + h
                    if i in cap and s is not None:
                        sn = s.norm()
                        if sn > cap[i]: s = s * (cap[i] / sn)
                    states[i] = s.detach() if s is not None else None
            else:
                h = model.encoder(c)
                for i, layer in enumerate(layers):
                    h, s = layer(h, states[i])
                    if i in cap:
                        sn = s.norm()
                        if sn > cap[i]: s = s * (cap[i] / sn)
                    states[i] = s.detach()

        # Now process last 64 tokens
        c = x2 = torch.tensor([tgt_ids[:-1]], dtype=torch.long).to(DEV)
        t2 = torch.tensor([tgt_ids[1:]], dtype=torch.long).to(DEV)
        if is_oa:
            h = model.em(x2)
            for i, layer in enumerate(layers):
                h2, s = layer(h, states[i])
                h = h2 + h
                if i in cap and s is not None:
                    sn = s.norm()
                    if sn > cap[i]: s = s * (cap[i] / sn)
                states[i] = s.detach() if s is not None else None
            logits = model.head_score(h)
        else:
            h = model.encoder(x2)
            for i, layer in enumerate(layers):
                h, s = layer(h, states[i])
                if i in cap:
                    sn = s.norm()
                    if sn > cap[i]: s = s * (cap[i] / sn)
                states[i] = s.detach()
            logits = model.head(h)

        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), t2.reshape(-1), ignore_index=0, reduction="sum").item()
        ntok = max((t2 != 0).sum().item(), 1)
        ppl = math.exp(nll / ntok)
        top5 = logits[0, -1].topk(5)
        toks = [voc.id_to_token.get(str(i.item()), '?') for i in top5.indices]
        print(f"  {label:>12}  PPL@4K={ppl:>8.1f}  top5={toks}")

print("\nDone.")
