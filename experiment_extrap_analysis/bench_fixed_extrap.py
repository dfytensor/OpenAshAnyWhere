#!/usr/bin/env python3
"""Extrapolation limit: WDLM-fixed vs OA — using model.forward() directly"""
import os, sys, math, json, torch, torch.nn.functional as F, time

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


def model_ppl_chunked(model, ids, sl, chunk=64, state_norm_cap=None):
    """Generic chunked PPL using model.forward(). state_norm_cap: {layer_idx: max_norm} for WDLM only."""
    s = ids[:sl]
    x = torch.tensor([s[:-1]], dtype=torch.long).to(DEV)
    t = torch.tensor([s[1:]], dtype=torch.long).to(DEV)
    state_norm_cap = state_norm_cap or {}

    with torch.no_grad():
        state = None
        cl = []
        for c0 in range(0, x.size(1), chunk):
            c = x[:, c0:c0+chunk]
            logits, state_out = model(c, state=state)

            # Apply norm cap for WDLM (state is list of tensors)
            if state_norm_cap and isinstance(state_out, list):
                for i in range(len(state_out)):
                    if i in state_norm_cap and state_out[i] is not None:
                        sn = state_out[i].norm()
                        if sn > state_norm_cap[i]:
                            state_out[i] = state_out[i] * (state_norm_cap[i] / sn)
                        state_out[i] = state_out[i].detach()

            state = state_out
            cl.append(logits)

        clo = torch.cat(cl, dim=1)
        nll = F.cross_entropy(clo.reshape(-1, clo.size(-1)), t.reshape(-1), ignore_index=0, reduction="sum").item()
        ntok = max((t != 0).sum().item(), 1)
    return math.exp(nll / ntok)


# Norm caps from OA-58M saturated norms
CAP_LOOSE = {0: 160, 1: 150, 2: 170, 3: 250, 4: 520, 5: 230, 6: 220, 7: 150, 8: 100, 9: 160}
CAP_TIGHT = {0: 130, 1: 120, 2: 140, 3: 200, 4: 420, 5: 190, 6: 180, 7: 120, 8: 80, 9: 130}
CAP_ALL200 = {i: 200 for i in range(10)}

seqs = [256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384]
seqs = [s for s in seqs if s <= len(all_ids)]

print("=" * 100)
print("  Extrapolation: WM-base | WM-loose | WM-tight | WM-200 | OA-58M | OA-85M")
print("=" * 100)
print(f"  {'Seq':>5}  {'WM-base':>10}  {'WM-loose':>10}  {'WM-tight':>10}  {'WM-200':>10}  {'OA-58M':>10}  {'OA-85M':>10}")
print(f"  {'-'*68}")

results = {}
for sl in seqs:
    t0 = time.time()
    r = {}
    r["wb"] = model_ppl_chunked(wm60, all_ids, sl, CHUNK)
    r["wl"] = model_ppl_chunked(wm60, all_ids, sl, CHUNK, state_norm_cap=CAP_LOOSE)
    r["wt"] = model_ppl_chunked(wm60, all_ids, sl, CHUNK, state_norm_cap=CAP_TIGHT)
    r["w2"] = model_ppl_chunked(wm60, all_ids, sl, CHUNK, state_norm_cap=CAP_ALL200)
    dt_wm = time.time() - t0

    t0 = time.time()
    r["o58"] = model_ppl_chunked(oa58, all_ids, sl, CHUNK)
    dt_oa = time.time() - t0

    r["o85"] = model_ppl_chunked(oa85, all_ids, sl, CHUNK)

    results[sl] = r
    print(f"  {sl:>5}  {r['wb']:>10.1f}  {r['wl']:>10.1f}  {r['wt']:>10.1f}  {r['w2']:>10.1f}  {r['o58']:>10.1f}  {r['o85']:>10.1f}  ({dt_wm:.1f}+{dt_oa:.1f}s)")
    sys.stdout.flush()

# Growth ratio
print()
print("=" * 100)
print(f"  Growth ratio (PPL / PPL@1024)")
print("=" * 100)
base = results[1024]
print(f"  {'Seq':>5}  {'WM-base':>10}  {'WM-loose':>10}  {'WM-tight':>10}  {'WM-200':>10}  {'OA-58M':>10}  {'OA-85M':>10}")
print(f"  {'-'*68}")
print(f"  {1024:>5}  {'1.0x':>10}  {'1.0x':>10}  {'1.0x':>10}  {'1.0x':>10}  {'1.0x':>10}  {'1.0x':>10}")
for sl in seqs:
    if sl == 1024: continue
    r = results[sl]
    print(f"  {sl:>5}  {r['wb']/base['wb']:>9.1f}x  {r['wl']/base['wl']:>9.1f}x  {r['wt']/base['wt']:>9.1f}x  {r['w2']/base['w2']:>9.1f}x  {r['o58']/base['o58']:>9.1f}x  {r['o85']/base['o85']:>9.1f}x")

print("\nDone.")
