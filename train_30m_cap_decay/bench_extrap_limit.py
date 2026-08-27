#!/usr/bin/env python3
"""
OpenASH-30M (cap+decay trained) extrapolation limit test
  Compare with OA-58M (no cap) baseline from bench/
  Test PPL at seq = 1K → 128K (up to 125x training length)
"""
import os, sys, math, json, torch, torch.nn.functional as F, time

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

SFT_DATA = os.path.join(ROOT, "minimind_data", "sft_t2t_mini.jsonl")

# ============================================================
# Load models
# ============================================================
print("Loading models...", flush=True)

# 30M cap+decay trained
m30 = OpenASH(vs, hidden_size=432, num_heads=8, num_layers=8, model_flag="train")
m30.load_state_dict(torch.load(os.path.join(ROOT, "train_30m_cap_decay", "openash30m_cd_sft_final.pth"), map_location=DEV)["model"])
m30.to(DEV).eval()

# 58M baseline (no cap)
m58 = OpenASH(vs, hidden_size=640, num_heads=8, num_layers=10, model_flag="train")
m58.load_state_dict(torch.load(os.path.join(BENCH, "openash60m_sft_final.pth"), map_location=DEV)["model"])
m58.to(DEV).eval()

# 85M baseline (no cap)
m85 = OpenASH(vs, hidden_size=768, num_heads=8, num_layers=12, model_flag="train")
m85.load_state_dict(torch.load(os.path.join(BENCH, "full_sft_768_12.pth"), map_location=DEV))
m85.to(DEV).eval()

print("Loaded: OA-30M-cd, OA-58M, OA-85M", flush=True)

# Warmup
for m in [m30, m58, m85]:
    for _ in range(3):
        x = torch.randint(1, 100, (1, 128), device=DEV)
        with torch.no_grad(): m(x, state=None)
torch.cuda.synchronize()

# ============================================================
# Prepare long sequence
# ============================================================
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
        if len(all_ids) >= 131072: break
print(f"Tokens: {len(all_ids)}", flush=True)


# ============================================================
# Chunked PPL with cap+decay
# ============================================================
def oa_ppl_chunked(model, ids, sl, chunk=64, use_cap_decay=False, n_layers=None):
    if n_layers is None:
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
                states[i] = s
                if use_cap_decay and s is not None:
                    sn = s.norm()
                    if sn > STATE_CAP:
                        s = s * (STATE_CAP / sn)
                    states[i] = s * STATE_DECAY
            cl.append(model.head_score(h))
        clo = torch.cat(cl, dim=1)
        nll = F.cross_entropy(clo.reshape(-1, clo.size(-1)), t.reshape(-1), ignore_index=0, reduction="sum").item()
        ntok = max((t != 0).sum().item(), 1)
    return math.exp(nll / ntok)


# ============================================================
# Test 1: Extrapolation sweep
# ============================================================
print(f"\n{'='*100}")
print("  Extrapolation PPL: OA-30M-cd (cap+decay trained) vs OA-58M vs OA-85M")
print(f"  30M cap={STATE_CAP}, decay={STATE_DECAY}")
print(f"{'='*100}")

seqs = [256, 512, 768, 1024, 1536, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
seqs = [s for s in seqs if s <= len(all_ids)]

print(f"  {'Seq':>7}  {'30M-cd':>10}  {'58M-base':>10}  {'85M-base':>10}  {'30M/58M':>8}  {'30M/85M':>8}")
print(f"  {'-'*65}")

for sl in seqs:
    p30 = oa_ppl_chunked(m30, all_ids, sl, CHUNK, use_cap_decay=True, n_layers=8)
    p58 = oa_ppl_chunked(m58, all_ids, sl, CHUNK, use_cap_decay=False)
    p85 = oa_ppl_chunked(m85, all_ids, sl, CHUNK, use_cap_decay=False)
    r58 = f"{p30/p58:.2f}x" if p58 > 0 else "-"
    r85 = f"{p30/p85:.2f}x" if p85 > 0 else "-"
    label = f"{sl//1024}K" if sl >= 1024 else str(sl)
    print(f"  {label:>7}  {p30:>10.1f}  {p58:>10.1f}  {p85:>10.1f}  {r58:>8}  {r85:>8}")
    sys.stdout.flush()

# ============================================================
# Test 2: 30M with/without cap+decay at inference
# ============================================================
print(f"\n{'='*100}")
print("  OA-30M-cd: with vs without cap+decay at inference")
print(f"{'='*100}")

print(f"  {'Seq':>7}  {'no-cap':>10}  {'cap+decay':>10}  {'delta':>8}")
print(f"  {'-'*45}")

for sl in seqs:
    p_nc = oa_ppl_chunked(m30, all_ids, sl, CHUNK, use_cap_decay=False, n_layers=8)
    p_cd = oa_ppl_chunked(m30, all_ids, sl, CHUNK, use_cap_decay=True, n_layers=8)
    delta = f"{p_cd-p_nc:+.1f}"
    label = f"{sl//1024}K" if sl >= 1024 else str(sl)
    print(f"  {label:>7}  {p_nc:>10.1f}  {p_cd:>10.1f}  {delta:>8}")
    sys.stdout.flush()

# ============================================================
# Test 3: 58M/85M also with cap+decay at inference (for fairness)
# ============================================================
print(f"\n{'='*100}")
print("  All models with cap+decay at inference (fair comparison)")
print(f"{'='*100}")

print(f"  {'Seq':>7}  {'30M-cd':>10}  {'58M-cd':>10}  {'85M-cd':>10}")
print(f"  {'-'*45}")

for sl in seqs:
    p30 = oa_ppl_chunked(m30, all_ids, sl, CHUNK, use_cap_decay=True, n_layers=8)
    p58 = oa_ppl_chunked(m58, all_ids, sl, CHUNK, use_cap_decay=True)
    p85 = oa_ppl_chunked(m85, all_ids, sl, CHUNK, use_cap_decay=True)
    label = f"{sl//1024}K" if sl >= 1024 else str(sl)
    print(f"  {label:>7}  {p30:>10.1f}  {p58:>10.1f}  {p85:>10.1f}")
    sys.stdout.flush()

# ============================================================
# Test 4: State norm tracking
# ============================================================
print(f"\n{'='*100}")
print("  State norm per layer at different seq lengths")
print(f"  OA-30M-cd (with cap+decay at inference)")
print(f"{'='*100}")

print(f"  {'Seq':>7}", end="")
for i in range(8): print(f"  {'L'+str(i):>8}", end="")
print()
print(f"  {'-'*80}")

for sl in [512, 1024, 4096, 16384, 65536, 131072]:
    if sl > len(all_ids): continue
    s = all_ids[:sl]
    x = torch.tensor([s[:-1]], dtype=torch.long).to(DEV)
    with torch.no_grad():
        states = [None] * 8
        for c0 in range(0, x.size(1), CHUNK):
            c = x[:, c0:c0+CHUNK]
            h = m30.em(c)
            for i, layer in enumerate(m30.decoder_layers):
                h2, s = layer(h, states[i])
                h = h2 + h
                states[i] = s
                if s is not None:
                    sn = s.norm()
                    if sn > STATE_CAP:
                        s = s * (STATE_CAP / sn)
                    states[i] = s * STATE_DECAY
    label = f"{sl//1024}K" if sl >= 1024 else str(sl)
    print(f"  {label:>7}", end="")
    for i in range(8):
        if states[i] is not None:
            print(f"  {states[i].norm().item():>8.1f}", end="")
        else:
            print(f"  {'?':>8}", end="")
    print()
    sys.stdout.flush()

print("\nDone.")
