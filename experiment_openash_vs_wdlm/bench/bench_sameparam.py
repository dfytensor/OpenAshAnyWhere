#!/usr/bin/env python3
"""
OpenASH 58M vs WDLM 60M — Full Benchmark (同参数量, 同数据, 同训练配置)
"""
import os, sys, time, math, json, torch, torch.nn.functional as F
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
_ORIG = r"F:\OpenASH2605"

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src_openash'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src_wdlm'))
os.chdir(_ORIG)

from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from wdlm_neural import WaveDynamicsLanguageModel
from open_ash import OpenASH
from open_ash_infer import sample_next_token, build_user_prompt, format_response, _sp

DEV = "cuda" if torch.cuda.is_available() else "cpu"
VOC_PATH = os.path.join(SCRIPT_DIR, "open_ash_voc_agent.json")
OA60_W = os.path.join(SCRIPT_DIR, "openash60m_sft_final.pth")
WDLM_W = os.path.join(SCRIPT_DIR, "wdlm60m_sft_final.pth")
SFT_DATA = os.path.join(_ORIG, "minimind_data", "sft_t2t_mini.jsonl")

GEN_CFG = dict(temperature=0.5, top_k=30, top_p=0.85, repetition_penalty=1.35)
TEST_PROMPTS = [
    "你好，请介绍一下你自己。",
    "什么是人工智能？",
    "请用Python写一个冒泡排序算法。",
    "解释量子计算的基本原理。",
    "中国的首都是哪里？",
    "写一首关于春天的诗。",
    "1+1等于几？",
    "请列举五种水果的名称。",
]


def _gen(model, pid, max_new=200, **kw):
    device = next(model.parameters()).device
    sp = _sp(voc)
    stop = {sp["im_end"], sp["pad"]}
    x = torch.tensor([pid], dtype=torch.long).to(device)
    if x.size(1) > 1024: x = x[:, -1024:]
    ids = []
    with torch.no_grad():
        state, chunk = None, x
        for _ in range(max_new):
            if chunk.size(1) > 1024: break
            r = model(chunk, state=state)
            logits = r[0] if isinstance(r, tuple) else r
            state = r[1] if isinstance(r, tuple) and len(r) > 1 else None
            nid = sample_next_token(logits[0, -1], ids, **kw)
            if nid in stop: break
            ids.append(nid)
            chunk = torch.tensor([[nid]], dtype=torch.long, device=device)
            if state is not None: state = [s.detach() for s in state]
    return ids


def _quality(ids):
    if not ids: return {"unique": 0, "rep3": 1, "entropy": 0}
    u = len(set(ids)) / len(ids)
    tg = [(ids[i], ids[i+1], ids[i+2]) for i in range(len(ids)-2)]
    r3 = 1 - len(set(tg)) / max(len(tg), 1)
    c = Counter(ids)
    p = [v/len(ids) for v in c.values()]
    e = -sum(x * math.log(x+1e-10) for x in p)
    return {"unique": u, "rep3": r3, "entropy": e}


def _ppl(model, path, seq_len=512, n=300):
    sp = _sp(voc)
    samples = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                convs = obj.get('conversations', [])
                ids = []
                for msg in convs:
                    r = msg.get('role', ''); ct = msg.get('content', '')
                    if r == 'user': ids += [sp["im_start"], sp["user"]] + voc.encode(ct) + [sp["im_end"]]
                    elif r == 'assistant':
                        ids += [sp["im_start"], sp["agent"]] + voc.encode(ct) + [sp["im_end"]]
                if len(ids) >= seq_len + 1:
                    samples.append(torch.tensor(ids[:seq_len+1], dtype=torch.long))
            except: pass
            if len(samples) >= n: break
    if not samples: return float('inf')
    nll, tok = 0, 0
    with torch.no_grad():
        for s in samples:
            x, t = s[:-1].unsqueeze(0).to(DEV), s[1:].unsqueeze(0).to(DEV)
            r = model(x, state=None)
            lo = r[0] if isinstance(r, tuple) else r
            nll += F.cross_entropy(lo.reshape(-1, lo.size(-1)), t.reshape(-1), ignore_index=0, reduction='sum').item()
            tok += max((t != 0).sum().item(), 1)
    return math.exp(nll / tok)


# ============================================================
print("=" * 70)
print("  OpenASH 58M vs WDLM 60M — Same-Param Full Benchmark")
print("=" * 70)

voc = OpenASHVoc(agent_voc_path=VOC_PATH)
vs = len(voc.token_to_id) + 1

print("\n[Load] OpenASH H=640 L=10 (58.2M)...")
oa = OpenASH(vs, hidden_size=640, num_heads=8, num_layers=10, model_flag="train")
_ckp = torch.load(OA60_W, map_location=DEV)
oa.load_state_dict(_ckp['model'] if 'model' in _ckp else _ckp)
oa.to(DEV).eval()
poa = sum(p.numel() for p in oa.parameters())
print(f"  Params: {poa:,}")

print("[Load] WDLM-Neural H=512 L=10 (60.3M)...")
wm = WaveDynamicsLanguageModel(vs, hidden_dim=512, num_layers=10)
ckp = torch.load(WDLM_W, map_location=DEV)
wm.load_state_dict(ckp['model'] if 'model' in ckp else ckp)
wm.to(DEV).eval()
pwm = sum(p.numel() for p in wm.parameters())
print(f"  Params: {pwm:,}")

# warmup
for _ in range(3):
    _gen(oa, build_user_prompt(voc, "warmup"), 30, **GEN_CFG)
    _gen(wm, build_user_prompt(voc, "warmup"), 30, **GEN_CFG)

# ============================================================
# 1. Generation Speed
# ============================================================
print(f"\n{'='*70}")
print("  1. Generation Speed (state, 3 runs median)")
print(f"{'='*70}")

sa, sw = [], []
for text in TEST_PROMPTS:
    pid = build_user_prompt(voc, text)
    to, tw = [], []
    for _ in range(3):
        ids_o, t0 = [], time.perf_counter(); torch.cuda.synchronize()
        ids_o = _gen(oa, pid, 200, **GEN_CFG)
        torch.cuda.synchronize(); to.append(len(ids_o) / (time.perf_counter() - t0))
        ids_w = _gen(wm, pid, 200, **GEN_CFG)
        torch.cuda.synchronize(); tw.append(len(ids_w) / (time.perf_counter() - t0))
    mo, mw = sorted(to)[1], sorted(tw)[1]
    sa.append(mo); sw.append(mw)
    print(f"  OA {mo:>5.0f} | WM {mw:>5.0f} tok/s | {text[:25]}")

aoa, awm = sum(sa)/len(sa), sum(sw)/len(sw)
print(f"\n  >>> OpenASH: {aoa:.1f} tok/s | WDLM: {awm:.1f} tok/s | ratio: {awm/aoa:.2f}x")

# ============================================================
# 2. TTFT
# ============================================================
print(f"\n{'='*70}")
print("  2. Time To First Token")
print(f"{'='*70}")
print(f"  {'Seq':>5}  {'OpenASH':>8}  {'WDLM':>8}")
print(f"  {'-'*26}")
for sl in [64, 128, 256, 512]:
    base = build_user_prompt(voc, "测试TTFT的提示词" * 10)
    ids = (base * (sl // len(base) + 1))[:sl]
    x = torch.tensor([ids], dtype=torch.long, device=DEV)
    tos, tws = [], []
    for _ in range(5):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad(): _ = oa(x, state=None)
        torch.cuda.synchronize(); tos.append(time.perf_counter() - t0)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad(): _ = wm(x, state=None)
        torch.cuda.synchronize(); tws.append(time.perf_counter() - t0)
    mo, mw = sorted(tos)[2]*1000, sorted(tws)[2]*1000
    print(f"  {sl:>5}  {mo:>6.1f}ms  {mw:>6.1f}ms")

# ============================================================
# 3. PPL
# ============================================================
print(f"\n{'='*70}")
print("  3. Perplexity (SFT, seq=512)")
print(f"{'='*70}")
if os.path.exists(SFT_DATA):
    ppl_oa = _ppl(oa, SFT_DATA, 512, 300)
    ppl_wm = _ppl(wm, SFT_DATA, 512, 300)
    print(f"  OpenASH: {ppl_oa:.2f}")
    print(f"  WDLM:    {ppl_wm:.2f}")
else:
    ppl_oa, ppl_wm = float('inf'), float('inf')
    print("  SFT data not found")

# PPL vs seq len
print(f"\n  PPL vs Context Length:")
print(f"  {'Seq':>5}  {'OpenASH':>8}  {'WDLM':>8}  {'Delta':>7}")
print(f"  {'-'*32}")
for sl in [64, 128, 256, 512]:
    po = _ppl(oa, SFT_DATA, sl, 50) if os.path.exists(SFT_DATA) else float('inf')
    pw = _ppl(wm, SFT_DATA, sl, 50) if os.path.exists(SFT_DATA) else float('inf')
    print(f"  {sl:>5}  {po:>8.2f}  {pw:>8.2f}  {pw-po:>+6.2f}")

# ============================================================
# 4. Generation Quality
# ============================================================
print(f"\n{'='*70}")
print("  4. Generation Quality")
print(f"{'='*70}")
print(f"  {'Prompt':>25}  {'M':>4}  {'Tok':>4}  {'Uni%':>6}  {'3g%':>5}  {'Ent':>5}")
print(f"  {'-'*58}")
for text in ["请列举五种水果", "写春天的诗", "解释引力", "猫的特征"]:
    pid = build_user_prompt(voc, text)
    for m, n in [(oa, "OA"), (wm, "WM")]:
        ids = _gen(m, pid, 150, **GEN_CFG)
        q = _quality(ids)
        print(f"  {text[:25]:>25}  {n:>4}  {len(ids):>4}  {q['unique']*100:>5.1f}%  {q['rep3']*100:>4.1f}%  {q['entropy']:>5.2f}")

# ============================================================
# 5. Sample Outputs
# ============================================================
print(f"\n{'='*70}")
print("  5. Sample Outputs")
print(f"{'='*70}")
for text in ["你好，介绍一下你自己。", "什么是人工智能？", "冒泡排序算法。"]:
    pid = build_user_prompt(voc, text)
    for m, n in [(oa, "OpenASH"), (wm, "WDLM")]:
        ids = _gen(m, pid, 150, **GEN_CFG)
        out = format_response(voc, ids).get('content', '')[:120].replace('\n', ' ')
        print(f"  [{n}] {text[:20]}: {out}")
    print()

# ============================================================
# 6. GPU Memory
# ============================================================
print(f"{'='*70}")
print("  6. GPU Memory")
print(f"{'='*70}")
for sl in [512, 1024]:
    for m, n in [(oa, "OpenASH"), (wm, "WDLM")]:
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
        with torch.no_grad():
            _ = m(torch.randint(1, 100, (1, sl), device=DEV), state=None)
        mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  {n} seq={sl}: {mem:.0f} MB")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*70}")
print("  SUMMARY — Same-Param Comparison")
print(f"{'='*70}")
print(f"  {'Metric':<30} {'OpenASH 58M':>14} {'WDLM 60M':>14}")
print(f"  {'-'*58}")
print(f"  {'Params':<30} {poa:>14,} {pwm:>14,}")
print(f"  {'Gen speed (tok/s)':<30} {aoa:>14.1f} {awm:>14.1f}")
print(f"  {'Speed ratio':<30} {'1.00x':>14} {awm/aoa:>13.2f}x")
if ppl_oa < float('inf'):
    print(f"  {'PPL (SFT, 512)':<30} {ppl_oa:>14.2f} {ppl_wm:>14.2f}")
print(f"  {'Param ratio':<30} {'100%':>14} {pwm/poa:>13.1%}")
print(f"{'='*70}")
