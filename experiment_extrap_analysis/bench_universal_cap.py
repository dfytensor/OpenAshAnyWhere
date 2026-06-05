#!/usr/bin/env python3
"""Test universal cap values across models — find cap that works without per-model tuning."""
import os, sys, math, json, torch, torch.nn.functional as F

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
voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1
sp = _sp(voc)
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

all_ids = []
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
            if ids: all_ids.extend(ids)
        except: pass
        if len(all_ids) >= 16384: break
print(f"Tokens: {len(all_ids)}")

for m in [oa58, oa85, wm60]:
    for _ in range(5):
        x = torch.randint(1, 100, (1, 128), device=DEV)
        with torch.no_grad(): m(x, state=None)
torch.cuda.synchronize()
print("Ready.\n")


def oa_ppl(model, ids, sl, chunk=64, state_norm_cap=None, state_decay=None):
    state_norm_cap = state_norm_cap or {}
    n_layers = len(model.decoder_layers)
    s = ids[:sl]
    x = torch.tensor([s[:-1]], dtype=torch.long).to(DEV)
    t = torch.tensor([s[1:]], dtype=torch.long).to(DEV)
    with torch.no_grad():
        states = [None] * n_layers
        cl = []
        for c0 in range(0, x.size(1), chunk):
            c = x[:, c0:c0+chunk]
            h = model.em(c)
            for i, layer in enumerate(model.decoder_layers):
                h2, s = layer(h, states[i])
                h = h2 + h
                if i in state_norm_cap and s is not None:
                    sn = s.norm()
                    if sn > state_norm_cap[i]:
                        s = s * (state_norm_cap[i] / sn)
                if state_decay is not None and s is not None:
                    s = s * state_decay
                states[i] = s.detach() if s is not None else None
            cl.append(model.head_score(h))
        clo = torch.cat(cl, dim=1)
        nll = F.cross_entropy(clo.reshape(-1, clo.size(-1)), t.reshape(-1), ignore_index=0, reduction="sum").item()
        ntok = max((t != 0).sum().item(), 1)
    return math.exp(nll / ntok)


def wdlm_ppl(model, ids, sl, chunk=64, state_norm_cap=None, state_decay=None):
    state_norm_cap = state_norm_cap or {}
    s = ids[:sl]
    x = torch.tensor([s[:-1]], dtype=torch.long).to(DEV)
    t = torch.tensor([s[1:]], dtype=torch.long).to(DEV)
    with torch.no_grad():
        state = None
        cl = []
        for c0 in range(0, x.size(1), chunk):
            c = x[:, c0:c0+chunk]
            logits, state_out = model(c, state=state)
            if isinstance(state_out, list):
                for i in range(len(state_out)):
                    if state_out[i] is not None:
                        if i in state_norm_cap:
                            sn = state_out[i].norm()
                            if sn > state_norm_cap[i]:
                                state_out[i] = state_out[i] * (state_norm_cap[i] / sn)
                        if state_decay is not None:
                            state_out[i] = state_out[i] * state_decay
            state = [s2.detach() if s2 is not None else None for s2 in state_out]
            cl.append(logits)
        clo = torch.cat(cl, dim=1)
        nll = F.cross_entropy(clo.reshape(-1, clo.size(-1)), t.reshape(-1), ignore_index=0, reduction="sum").item()
        ntok = max((t != 0).sum().item(), 1)
    return math.exp(nll / ntok)


def make_cap(n_layers, val):
    return {i: val for i in range(n_layers)}


CAPS = [50, 100, 150, 200, 300, 500]
DECAYS_COMBO = [0.99, 0.97]
EVAL_SEQS = [1024, 4096, 8192, 16384]
EVAL_SEQS = [s for s in EVAL_SEQS if s <= len(all_ids)]

models_info = [
    ("OA-58M", oa58, oa_ppl, 10),
    ("OA-85M", oa85, oa_ppl, 12),
    ("WM-60M", wm60, wdlm_ppl, 10),
]

# ============================================================
# Part 1: Universal cap sweep (no decay)
# ============================================================
print("=" * 120)
print("  Part 1: Universal cap sweep (no decay) — find best uniform cap")
print("=" * 120)
for name, model, fn, nl in models_info:
    print(f"\n  [{name}]")
    hdr = f"  {'Seq':>5}  {'base':>8}"
    for c in CAPS:
        hdr += f"  {'cap='+str(c):>8}"
    print(hdr)
    print(f"  {'-'*len(hdr)}")
    for sl in EVAL_SEQS:
        row = f"  {sl:>5}"
        r_base = fn(model, all_ids, sl, CHUNK)
        row += f"  {r_base:>8.1f}"
        for c in CAPS:
            r = fn(model, all_ids, sl, CHUNK, state_norm_cap=make_cap(nl, c))
            row += f"  {r:>8.1f}"
        print(row)
        sys.stdout.flush()

# ============================================================
# Part 2: decay-only sweep (no cap)
# ============================================================
print("\n" + "=" * 120)
print("  Part 2: Decay-only sweep (no cap) — model-agnostic?")
print("=" * 120)
DECAYS_SOLO = [0.995, 0.99, 0.97, 0.95, 0.90]
for name, model, fn, nl in models_info:
    print(f"\n  [{name}]")
    hdr = f"  {'Seq':>5}  {'base':>8}"
    for d in DECAYS_SOLO:
        hdr += f"  {'d='+str(d):>8}"
    print(hdr)
    print(f"  {'-'*len(hdr)}")
    for sl in EVAL_SEQS:
        row = f"  {sl:>5}"
        r_base = fn(model, all_ids, sl, CHUNK)
        row += f"  {r_base:>8.1f}"
        for d in DECAYS_SOLO:
            r = fn(model, all_ids, sl, CHUNK, state_decay=d)
            row += f"  {r:>8.1f}"
        print(row)
        sys.stdout.flush()

# ============================================================
# Part 3: Best cap + decay combo (cap=150 universal)
# ============================================================
print("\n" + "=" * 120)
print("  Part 3: Universal cap=150 + decay combo")
print("=" * 120)
for name, model, fn, nl in models_info:
    print(f"\n  [{name}]")
    hdr = f"  {'Seq':>5}  {'base':>8}  {'cap150':>8}  {'d0.97':>8}  {'d0.99':>8}  {'c+d97':>8}  {'c+d99':>8}"
    print(hdr)
    print(f"  {'-'*len(hdr)}")
    for sl in EVAL_SEQS:
        r = {}
        r["base"] = fn(model, all_ids, sl, CHUNK)
        r["cap"] = fn(model, all_ids, sl, CHUNK, state_norm_cap=make_cap(nl, 150))
        r["d97"] = fn(model, all_ids, sl, CHUNK, state_decay=0.97)
        r["d99"] = fn(model, all_ids, sl, CHUNK, state_decay=0.99)
        r["cd97"] = fn(model, all_ids, sl, CHUNK, state_norm_cap=make_cap(nl, 150), state_decay=0.97)
        r["cd99"] = fn(model, all_ids, sl, CHUNK, state_norm_cap=make_cap(nl, 150), state_decay=0.99)
        print(f"  {sl:>5}  {r['base']:>8.1f}  {r['cap']:>8.1f}  {r['d97']:>8.1f}  {r['d99']:>8.1f}  {r['cd97']:>8.1f}  {r['cd99']:>8.1f}")
        sys.stdout.flush()

# ============================================================
# Part 4: Ratio-based cap — cap = k * hidden_size
# ============================================================
print("\n" + "=" * 120)
print("  Part 4: Ratio-based cap — cap = k * sqrt(hidden_dim)")
print("=" * 120)
H_SIZES = {"OA-58M": 640, "OA-85M": 768, "WM-60M": 512}
RATIOS = [0.2, 0.3, 0.5, 0.8, 1.0]
for name, model, fn, nl in models_info:
    h = H_SIZES[name]
    print(f"\n  [{name}] hidden_dim={h}, sqrt={h**0.5:.1f}")
    hdr = f"  {'Seq':>5}  {'base':>8}"
    for k in RATIOS:
        cap_val = int(k * (h ** 0.5))
        hdr += f"  {'k='+str(k)+'('+str(cap_val)+')':>14}"
    print(hdr)
    print(f"  {'-'*len(hdr)}")
    for sl in EVAL_SEQS:
        row = f"  {sl:>5}"
        r_base = fn(model, all_ids, sl, CHUNK)
        row += f"  {r_base:>8.1f}"
        for k in RATIOS:
            cap_val = int(k * (h ** 0.5))
            r = fn(model, all_ids, sl, CHUNK, state_norm_cap=make_cap(nl, cap_val))
            row += f"  {r:>14.1f}"
        print(row)
        sys.stdout.flush()

print("\nDone.")
