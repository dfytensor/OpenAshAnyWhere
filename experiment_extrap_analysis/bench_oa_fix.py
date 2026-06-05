#!/usr/bin/env python3
"""Can OA's extrapolation be fixed? Test state interventions on OA-58M and OA-85M"""
import os, sys, math, json, torch, torch.nn.functional as F, time

ROOT = r"F:\OpenASH2605"
BENCH = os.path.join(ROOT, "experiment_openash_vs_wdlm", "bench")
sys.path.insert(0, ROOT); sys.path.insert(0, BENCH); os.chdir(ROOT)

from open_ash import OpenASH
from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from open_ash_infer import _sp

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

for m in [oa58, oa85]:
    for _ in range(5):
        x = torch.randint(1, 100, (1, 128), device=DEV)
        with torch.no_grad(): m(x, state=None)
torch.cuda.synchronize()
print("Ready.\n")


def oa_ppl_intervened(model, ids, sl, chunk=64,
                      state_norm_cap=None,
                      state_decay=None,
                      output_norm_cap=None):
    """
    OA chunked PPL with interventions:
    - state_norm_cap: {layer_idx: max_norm}
    - state_decay: float, multiply state by this each chunk (< 1.0 for decay)
    - output_norm_cap: float, cap hidden state norm per token
    """
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

                # State norm cap
                if i in state_norm_cap and s is not None:
                    sn = s.norm()
                    if sn > state_norm_cap[i]:
                        s = s * (state_norm_cap[i] / sn)

                # State decay
                if state_decay is not None and s is not None:
                    s = s * state_decay

                states[i] = s.detach() if s is not None else None

            # Output norm cap
            if output_norm_cap is not None:
                hn = h.norm(dim=-1, keepdim=True)
                mask = hn > output_norm_cap
                if mask.any():
                    h = h * (output_norm_cap / hn.clamp(min=1e-8))

            cl.append(model.head_score(h))
        clo = torch.cat(cl, dim=1)
        nll = F.cross_entropy(clo.reshape(-1, clo.size(-1)), t.reshape(-1), ignore_index=0, reduction="sum").item()
        ntok = max((t != 0).sum().item(), 1)
    return math.exp(nll / ntok)


# First: get OA-58M saturated state norms per layer at seq=1024 (training limit)
# These are from the analysis: L0=96, L1=84, L2=107, L3=158, L4=341, L5=150, L6=145, L7=96, L8=66, L9=103
OA58_SATURATED = {0: 96, 1: 84, 2: 107, 3: 158, 4: 341, 5: 150, 6: 145, 7: 96, 8: 66, 9: 103}
OA58_CAP_SAME = {i: v for i, v in OA58_SATURATED.items()}  # cap at saturated value
OA58_CAP_TIGHT = {i: int(v * 0.8) for i, v in OA58_SATURATED.items()}

seqs = [256, 512, 768, 1024, 1536, 2048, 4096, 8192, 12288, 16384]
seqs = [s for s in seqs if s <= len(all_ids)]

# ============================================================
# Test 1: OA-58M state norm cap (cap at training saturated value)
# ============================================================
print("=" * 100)
print("  OA-58M: baseline vs state-norm-cap vs decay vs output-cap")
print("=" * 100)
print(f"  {'Seq':>5}  {'base':>8}  {'cap-same':>10}  {'cap-0.8x':>10}  {'decay0.99':>10}  {'decay0.95':>10}  {'out-cap':>10}")
print(f"  {'-'*68}")

for sl in seqs:
    r = {}
    r["base"] = oa_ppl_intervened(oa58, all_ids, sl, CHUNK)
    r["cap"] = oa_ppl_intervened(oa58, all_ids, sl, CHUNK, state_norm_cap=OA58_CAP_SAME)
    r["tight"] = oa_ppl_intervened(oa58, all_ids, sl, CHUNK, state_norm_cap=OA58_CAP_TIGHT)
    r["d99"] = oa_ppl_intervened(oa58, all_ids, sl, CHUNK, state_decay=0.99)
    r["d95"] = oa_ppl_intervened(oa58, all_ids, sl, CHUNK, state_decay=0.95)
    r["out"] = oa_ppl_intervened(oa58, all_ids, sl, CHUNK, output_norm_cap=100.0)
    print(f"  {sl:>5}  {r['base']:>8.1f}  {r['cap']:>10.1f}  {r['tight']:>10.1f}  {r['d99']:>10.1f}  {r['d95']:>10.1f}  {r['out']:>10.1f}")
    sys.stdout.flush()

# ============================================================
# Test 2: OA-85M state norm cap
# ============================================================
print()
print("=" * 100)
print("  OA-85M: baseline vs state-norm-cap vs decay")
print("=" * 100)

# Get OA-85M saturated norms (estimate, not measured — use generous values)
# OA-85M has H=768 L=12, likely higher norms
OA85_CAP = {i: 200 for i in range(12)}
OA85_CAP_TIGHT = {i: 150 for i in range(12)}

print(f"  {'Seq':>5}  {'base':>8}  {'cap-200':>10}  {'cap-150':>10}  {'decay0.99':>10}  {'decay0.95':>10}")
print(f"  {'-'*55}")

for sl in seqs:
    r = {}
    r["base"] = oa_ppl_intervened(oa85, all_ids, sl, CHUNK)
    r["c200"] = oa_ppl_intervened(oa85, all_ids, sl, CHUNK, state_norm_cap=OA85_CAP)
    r["c150"] = oa_ppl_intervened(oa85, all_ids, sl, CHUNK, state_norm_cap=OA85_CAP_TIGHT)
    r["d99"] = oa_ppl_intervened(oa85, all_ids, sl, CHUNK, state_decay=0.99)
    r["d95"] = oa_ppl_intervened(oa85, all_ids, sl, CHUNK, state_decay=0.95)
    print(f"  {sl:>5}  {r['base']:>8.1f}  {r['c200']:>10.1f}  {r['c150']:>10.1f}  {r['d99']:>10.1f}  {r['d95']:>10.1f}")
    sys.stdout.flush()

# ============================================================
# Test 3: Get actual OA-85M state norms to calibrate
# ============================================================
print()
print("=" * 100)
print("  OA-85M actual state norms at different seq lengths")
print("=" * 100)

print(f"  {'Seq':>5}", end="")
for i in range(12): print(f"  {'L'+str(i):>8}", end="")
print()
print(f"  {'-'*105}")

for sl in [256, 1024, 4096, 16384]:
    if sl > len(all_ids): continue
    s = all_ids[:sl]
    x = torch.tensor([s[:-1]], dtype=torch.long).to(DEV)
    with torch.no_grad():
        states = [None] * 12
        for c0 in range(0, x.size(1), CHUNK):
            c = x[:, c0:c0+CHUNK]
            h = oa85.em(c)
            for i, layer in enumerate(oa85.decoder_layers):
                h2, s = layer(h, states[i])
                h = h2 + h
                states[i] = s.detach() if s is not None else None
    print(f"  {sl:>5}", end="")
    for i in range(12):
        if states[i] is not None:
            print(f"  {states[i].norm().item():>8.1f}", end="")
        else:
            print(f"  {'?':>8}", end="")
    print()

# ============================================================
# Test 4: OA-85M with calibrated cap
# ============================================================
print()
print("=" * 100)
print("  OA-85M: baseline vs calibrated-cap vs decay")
print("=" * 100)

print(f"  {'Seq':>5}  {'base':>8}  {'cap-cal':>10}  {'decay0.99':>10}  {'decay0.97':>10}  {'decay0.95':>10}")
print(f"  {'-'*55}")

for sl in seqs:
    r = {}
    r["base"] = oa_ppl_intervened(oa85, all_ids, sl, CHUNK)
    r["cal"] = oa_ppl_intervened(oa85, all_ids, sl, CHUNK, state_norm_cap=OA85_CAP)
    r["d99"] = oa_ppl_intervened(oa85, all_ids, sl, CHUNK, state_decay=0.99)
    r["d97"] = oa_ppl_intervened(oa85, all_ids, sl, CHUNK, state_decay=0.97)
    r["d95"] = oa_ppl_intervened(oa85, all_ids, sl, CHUNK, state_decay=0.95)
    print(f"  {sl:>5}  {r['base']:>8.1f}  {r['cal']:>10.1f}  {r['d99']:>10.1f}  {r['d97']:>10.1f}  {r['d95']:>10.1f}")
    sys.stdout.flush()

print("\nDone.")
