#!/usr/bin/env python3
"""Combine cap-same + decay on OA-58M and OA-85M, compare with single-method."""
import os, sys, math, json, torch, torch.nn.functional as F
import time

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
                      state_decay=None):
    state_norm_cap = state_norm_cap or {}
    n_layers = len(model.decoder_layers)
    s_data = ids[:sl]
    x = torch.tensor([s_data[:-1]], dtype=torch.long).to(DEV)
    t = torch.tensor([s_data[1:]], dtype=torch.long).to(DEV)
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


OA58_SATURATED = {0: 96, 1: 84, 2: 107, 3: 158, 4: 341, 5: 150, 6: 145, 7: 96, 8: 66, 9: 103}
OA58_CAP_SAME = {i: v for i, v in OA58_SATURATED.items()}
OA85_CAP = {i: 200 for i in range(12)}

seqs = [256, 512, 768, 1024, 1536, 2048, 4096, 8192, 12288, 16384]
seqs = [s for s in seqs if s <= len(all_ids)]

DECAYS = [0.99, 0.97, 0.95]

# ============================================================
# Test 1: OA-58M  baseline / cap-same / decay0.97 / cap-same+decay0.97 / cap-same+decay0.95
# ============================================================
print("=" * 110)
print("  OA-58M: cap-same + decay combo")
print("=" * 110)
hdr = f"  {'Seq':>5}  {'base':>8}  {'cap':>8}  {'d0.99':>8}  {'d0.97':>8}  {'d0.95':>8}  {'cap+d0.99':>10}  {'cap+d0.97':>10}  {'cap+d0.95':>10}"
print(hdr)
print(f"  {'-'*100}")

for sl in seqs:
    r = {}
    r["base"] = oa_ppl_intervened(oa58, all_ids, sl, CHUNK)
    r["cap"] = oa_ppl_intervened(oa58, all_ids, sl, CHUNK, state_norm_cap=OA58_CAP_SAME)
    r["d99"] = oa_ppl_intervened(oa58, all_ids, sl, CHUNK, state_decay=0.99)
    r["d97"] = oa_ppl_intervened(oa58, all_ids, sl, CHUNK, state_decay=0.97)
    r["d95"] = oa_ppl_intervened(oa58, all_ids, sl, CHUNK, state_decay=0.95)
    r["cd99"] = oa_ppl_intervened(oa58, all_ids, sl, CHUNK, state_norm_cap=OA58_CAP_SAME, state_decay=0.99)
    r["cd97"] = oa_ppl_intervened(oa58, all_ids, sl, CHUNK, state_norm_cap=OA58_CAP_SAME, state_decay=0.97)
    r["cd95"] = oa_ppl_intervened(oa58, all_ids, sl, CHUNK, state_norm_cap=OA58_CAP_SAME, state_decay=0.95)
    print(f"  {sl:>5}  {r['base']:>8.1f}  {r['cap']:>8.1f}  {r['d99']:>8.1f}  {r['d97']:>8.1f}  {r['d95']:>8.1f}  {r['cd99']:>10.1f}  {r['cd97']:>10.1f}  {r['cd95']:>10.1f}")
    sys.stdout.flush()

# ============================================================
# Test 2: OA-85M  baseline / cap-200 / decay0.97 / cap-200+decay0.97 / cap-200+decay0.95
# ============================================================
print()
print("=" * 110)
print("  OA-85M: cap-200 + decay combo")
print("=" * 110)
print(hdr)
print(f"  {'-'*100}")

for sl in seqs:
    r = {}
    r["base"] = oa_ppl_intervened(oa85, all_ids, sl, CHUNK)
    r["cap"] = oa_ppl_intervened(oa85, all_ids, sl, CHUNK, state_norm_cap=OA85_CAP)
    r["d99"] = oa_ppl_intervened(oa85, all_ids, sl, CHUNK, state_decay=0.99)
    r["d97"] = oa_ppl_intervened(oa85, all_ids, sl, CHUNK, state_decay=0.97)
    r["d95"] = oa_ppl_intervened(oa85, all_ids, sl, CHUNK, state_decay=0.95)
    r["cd99"] = oa_ppl_intervened(oa85, all_ids, sl, CHUNK, state_norm_cap=OA85_CAP, state_decay=0.99)
    r["cd97"] = oa_ppl_intervened(oa85, all_ids, sl, CHUNK, state_norm_cap=OA85_CAP, state_decay=0.97)
    r["cd95"] = oa_ppl_intervened(oa85, all_ids, sl, CHUNK, state_norm_cap=OA85_CAP, state_decay=0.95)
    print(f"  {sl:>5}  {r['base']:>8.1f}  {r['cap']:>8.1f}  {r['d99']:>8.1f}  {r['d97']:>8.1f}  {r['d95']:>8.1f}  {r['cd99']:>10.1f}  {r['cd97']:>10.1f}  {r['cd95']:>10.1f}")
    sys.stdout.flush()

# ============================================================
# Test 3: WDLM-60M combo (norm-200 + decay)
# ============================================================
sys.path.insert(0, os.path.join(ROOT, "wdlm_verification"))
from wdlm_neural import WaveDynamicsLanguageModel

wm60 = WaveDynamicsLanguageModel(vs, hidden_dim=512, num_layers=10)
_ck = torch.load(os.path.join(BENCH, "wdlm60m_sft_final.pth"), map_location=DEV)
wm60.load_state_dict(_ck["model"] if "model" in _ck else _ck)
wm60.to(DEV).eval()

for _ in range(5):
    x = torch.randint(1, 100, (1, 128), device=DEV)
    with torch.no_grad(): wm60(x, state=None)
torch.cuda.synchronize()

WM_CAP = {i: 200 for i in range(10)}


def wdlm_ppl_intervened(model, ids, sl, chunk=64,
                        state_norm_cap=None,
                        state_decay=None):
    state_norm_cap = state_norm_cap or {}
    s_data = ids[:sl]
    x = torch.tensor([s_data[:-1]], dtype=torch.long).to(DEV)
    t = torch.tensor([s_data[1:]], dtype=torch.long).to(DEV)
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
            state = [s.detach() if s is not None else None for s in state_out]
            cl.append(logits)
        clo = torch.cat(cl, dim=1)
        nll = F.cross_entropy(clo.reshape(-1, clo.size(-1)), t.reshape(-1), ignore_index=0, reduction="sum").item()
        ntok = max((t != 0).sum().item(), 1)
    return math.exp(nll / ntok)


print()
print("=" * 110)
print("  WDLM-60M: cap-200 + decay combo")
print("=" * 110)
print(hdr)
print(f"  {'-'*100}")

for sl in seqs:
    r = {}
    r["base"] = wdlm_ppl_intervened(wm60, all_ids, sl, CHUNK)
    r["cap"] = wdlm_ppl_intervened(wm60, all_ids, sl, CHUNK, state_norm_cap=WM_CAP)
    r["d99"] = wdlm_ppl_intervened(wm60, all_ids, sl, CHUNK, state_decay=0.99)
    r["d97"] = wdlm_ppl_intervened(wm60, all_ids, sl, CHUNK, state_decay=0.97)
    r["d95"] = wdlm_ppl_intervened(wm60, all_ids, sl, CHUNK, state_decay=0.95)
    r["cd99"] = wdlm_ppl_intervened(wm60, all_ids, sl, CHUNK, state_norm_cap=WM_CAP, state_decay=0.99)
    r["cd97"] = wdlm_ppl_intervened(wm60, all_ids, sl, CHUNK, state_norm_cap=WM_CAP, state_decay=0.97)
    r["cd95"] = wdlm_ppl_intervened(wm60, all_ids, sl, CHUNK, state_norm_cap=WM_CAP, state_decay=0.95)
    print(f"  {sl:>5}  {r['base']:>8.1f}  {r['cap']:>8.1f}  {r['d99']:>8.1f}  {r['d97']:>8.1f}  {r['d95']:>8.1f}  {r['cd99']:>10.1f}  {r['cd97']:>10.1f}  {r['cd95']:>10.1f}")
    sys.stdout.flush()

print("\nDone.")
