#!/usr/bin/env python3
"""
Layer-by-layer state analysis: directly instrument forward pass.
Track cummax state at each layer for increasing context lengths.
"""
import os, sys, math, json, torch, torch.nn.functional as F
import numpy as np

ROOT = r"F:\OpenASH2605"
BENCH = os.path.join(ROOT, "experiment_openash_vs_wdlm", "bench")
OUT = os.path.join(ROOT, "experiment_extrap_analysis")
sys.path.insert(0, ROOT); sys.path.insert(0, BENCH); os.chdir(ROOT)

from open_ash import OpenASH
from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from open_ash_infer import _sp
sys.path.insert(0, os.path.join(ROOT, "wdlm_verification"))
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

wm60 = WaveDynamicsLanguageModel(vs, hidden_dim=512, num_layers=10)
_ck = torch.load(os.path.join(BENCH, "wdlm60m_sft_final.pth"), map_location=DEV)
wm60.load_state_dict(_ck["model"] if "model" in _ck else _ck)
wm60.to(DEV).eval()

# Collect long data
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
        if len(all_ids) >= 8192: break
print(f"Tokens: {len(all_ids)}")

# Warmup
for m in [oa58, wm60]:
    for _ in range(5):
        x = torch.randint(1, 100, (1, 64), device=DEV)
        with torch.no_grad(): m(x, state=None)
torch.cuda.synchronize()
print("Warmup done.\n")


def wdlm_forward_instrumented(model, x, chunk=64):
    """Run WDLM chunked forward, collect state stats after each layer at each chunk."""
    n_layers = len(model.layers)
    # Results: {layer_idx: {"chunk_states": [state_norm_per_chunk], ...}}
    results = {i: {"state_norms": [], "state_maxs": [], "out_norms": [], "out_maxs": []} for i in range(n_layers)}

    with torch.no_grad():
        states = [None] * n_layers
        for c_start in range(0, x.size(1), chunk):
            c = x[:, c_start:c_start+chunk]
            h = model.encoder(c)
            for i, layer in enumerate(model.layers):
                h, s = layer(h, states[i])
                states[i] = s.detach() if s is not None else None

                results[i]["state_norms"].append(s.norm().item() if s is not None else 0)
                results[i]["state_maxs"].append(s.abs().max().item() if s is not None else 0)
                results[i]["out_norms"].append(h.norm(dim=-1).mean().item())
                results[i]["out_maxs"].append(h.abs().max().item())

    return results


def oa_forward_instrumented(model, x, chunk=64):
    """Run OA chunked forward, collect state stats."""
    n_layers = len(model.decoder_layers)
    results = {i: {"state_norms": [], "state_maxs": [], "out_norms": [], "out_maxs": []} for i in range(n_layers)}

    with torch.no_grad():
        states = [None] * n_layers
        for c_start in range(0, x.size(1), chunk):
            c = x[:, c_start:c_start+chunk]
            h = model.em(c)
            for i, layer in enumerate(model.decoder_layers):
                h, s = layer(h, states[i])
                states[i] = s.detach() if s is not None else None

                results[i]["state_norms"].append(s.norm().item() if s is not None else 0)
                results[i]["state_maxs"].append(s.abs().max().item() if s is not None else 0)
                results[i]["out_norms"].append(h.norm(dim=-1).mean().item())
                results[i]["out_maxs"].append(h.abs().max().item())

    return results


# ============================================================
# Test 1: State norm at different total sequence lengths
# ============================================================
print("=" * 80)
print("  Test 1: Final state NORM per layer at different seq lengths")
print("=" * 80)

seqs = [256, 512, 768, 1024, 1536, 2048, 4096, 8192]
seqs = [s for s in seqs if s <= len(all_ids)]

print(f"\n  --- WDLM-60M ---")
print(f"  {'Seq':>5}", end="")
for i in range(10): print(f"  {'L'+str(i):>8}", end="")
print(f"  {'TOTAL':>10}")
print(f"  {'-'*100}")

for sl in seqs:
    x = torch.tensor([all_ids[:sl]], dtype=torch.long).to(DEV)
    r = wdlm_forward_instrumented(wm60, x, CHUNK)
    print(f"  {sl:>5}", end="")
    for i in range(10):
        v = r[i]["state_norms"][-1]
        print(f"  {v:>8.1f}", end="")
    total = sum(r[i]["state_norms"][-1] for i in range(10))
    print(f"  {total:>10.1f}")

print(f"\n  --- OA-58M ---")
print(f"  {'Seq':>5}", end="")
for i in range(10): print(f"  {'L'+str(i):>8}", end="")
print(f"  {'TOTAL':>10}")
print(f"  {'-'*100}")

for sl in seqs:
    x = torch.tensor([all_ids[:sl]], dtype=torch.long).to(DEV)
    r = oa_forward_instrumented(oa58, x, CHUNK)
    print(f"  {sl:>5}", end="")
    for i in range(10):
        v = r[i]["state_norms"][-1]
        print(f"  {v:>8.1f}", end="")
    total = sum(r[i]["state_norms"][-1] for i in range(10))
    print(f"  {total:>10.1f}")


# ============================================================
# Test 2: State MAX value per layer (detect explosion)
# ============================================================
print()
print("=" * 80)
print("  Test 2: Final state MAX value per layer (detect explosion)")
print("=" * 80)

print(f"\n  --- WDLM-60M ---")
print(f"  {'Seq':>5}", end="")
for i in range(10): print(f"  {'L'+str(i):>8}", end="")
print()
print(f"  {'-'*90}")

for sl in seqs:
    x = torch.tensor([all_ids[:sl]], dtype=torch.long).to(DEV)
    r = wdlm_forward_instrumented(wm60, x, CHUNK)
    print(f"  {sl:>5}", end="")
    for i in range(10):
        v = r[i]["state_maxs"][-1]
        if v > 100: print(f"  {'!!!'+str(int(v)):>8}", end="")
        else: print(f"  {v:>8.2f}", end="")
    print()

print(f"\n  --- OA-58M ---")
print(f"  {'Seq':>5}", end="")
for i in range(10): print(f"  {'L'+str(i):>8}", end="")
print()
print(f"  {'-'*90}")

for sl in seqs:
    x = torch.tensor([all_ids[:sl]], dtype=torch.long).to(DEV)
    r = oa_forward_instrumented(oa58, x, CHUNK)
    print(f"  {sl:>5}", end="")
    for i in range(10):
        v = r[i]["state_maxs"][-1]
        if v > 100: print(f"  {'!!!'+str(int(v)):>8}", end="")
        else: print(f"  {v:>8.2f}", end="")
    print()


# ============================================================
# Test 3: Output norm per layer (layer-by-layer propagation)
# ============================================================
print()
print("=" * 80)
print("  Test 3: Output NORM per layer (layer-by-layer at seq=4096)")
print("=" * 80)

SL = 4096 if 4096 <= len(all_ids) else len(all_ids)
x = torch.tensor([all_ids[:SL]], dtype=torch.long).to(DEV)

print(f"\n  --- WDLM-60M (seq={SL}) ---")
r = wdlm_forward_instrumented(wm60, x, CHUNK)
print(f"  {'Layer':>6}  {'st_norm_final':>14}  {'st_max_final':>13}  {'out_norm_last':>14}  {'out_max_last':>13}")
print(f"  {'-'*65}")
for i in range(10):
    sn = r[i]["state_norms"][-1]
    sm = r[i]["state_maxs"][-1]
    on = r[i]["out_norms"][-1]
    om = r[i]["out_maxs"][-1]
    flag = " <<<" if sm > 50 else ""
    print(f"  {'L'+str(i):>6}  {sn:>14.2f}  {sm:>13.2f}  {on:>14.2f}  {om:>13.2f}{flag}")

print(f"\n  --- OA-58M (seq={SL}) ---")
r = oa_forward_instrumented(oa58, x, CHUNK)
print(f"  {'Layer':>6}  {'st_norm_final':>14}  {'st_max_final':>13}  {'out_norm_last':>14}  {'out_max_last':>13}")
print(f"  {'-'*65}")
for i in range(10):
    sn = r[i]["state_norms"][-1]
    sm = r[i]["state_maxs"][-1]
    on = r[i]["out_norms"][-1]
    om = r[i]["out_maxs"][-1]
    flag = " <<<" if sm > 50 else ""
    print(f"  {'L'+str(i):>6}  {sn:>14.2f}  {sm:>13.2f}  {on:>14.2f}  {om:>13.2f}{flag}")


# ============================================================
# Test 4: State evolution across chunks at seq=4096
# ============================================================
print()
print("=" * 80)
print(f"  Test 4: State NORM evolution across chunks (seq={SL})")
print("=" * 80)

x = torch.tensor([all_ids[:SL]], dtype=torch.long).to(DEV)

# Pick layers to display: first, middle, last
display_layers_wm = [0, 3, 6, 9]
display_layers_oa = [0, 3, 6, 9]

print(f"\n  --- WDLM-60M ---")
r = wdlm_forward_instrumented(wm60, x, CHUNK)
n_chunks = len(r[0]["state_norms"])
print(f"  {'Chunk':>5}  {'Pos':>8}", end="")
for i in display_layers_wm: print(f"  {'L'+str(i):>10}", end="")
print()
print(f"  {'-'*55}")
for ci in range(n_chunks):
    if ci % 8 == 0 or ci == n_chunks - 1:
        pos = ci * CHUNK
        print(f"  {ci:>5}  {pos:>4}-{pos+CHUNK:>4}", end="")
        for i in display_layers_wm:
            print(f"  {r[i]['state_norms'][ci]:>10.1f}", end="")
        print()

print(f"\n  --- OA-58M ---")
r = oa_forward_instrumented(oa58, x, CHUNK)
print(f"  {'Chunk':>5}  {'Pos':>8}", end="")
for i in display_layers_oa: print(f"  {'L'+str(i):>10}", end="")
print()
print(f"  {'-'*55}")
for ci in range(n_chunks):
    if ci % 8 == 0 or ci == n_chunks - 1:
        pos = ci * CHUNK
        print(f"  {ci:>5}  {pos:>4}-{pos+CHUNK:>4}", end="")
        for i in display_layers_oa:
            print(f"  {r[i]['state_norms'][ci]:>10.1f}", end="")
        print()


# ============================================================
# Test 5: Growth rate per layer (norm ratio seq=4096 / seq=256)
# ============================================================
print()
print("=" * 80)
print("  Test 5: State NORM growth rate (seq=4096 / seq=256)")
print("=" * 80)

base_sl = 256
long_sl = 4096 if 4096 <= len(all_ids) else len(all_ids)

x_base = torch.tensor([all_ids[:base_sl]], dtype=torch.long).to(DEV)
x_long = torch.tensor([all_ids[:long_sl]], dtype=torch.long).to(DEV)

r_wm_base = wdlm_forward_instrumented(wm60, x_base, CHUNK)
r_wm_long = wdlm_forward_instrumented(wm60, x_long, CHUNK)
r_oa_base = oa_forward_instrumented(oa58, x_base, CHUNK)
r_oa_long = oa_forward_instrumented(oa58, x_long, CHUNK)

print(f"\n  {'Layer':>6}  {'WM base':>10}  {'WM long':>10}  {'WM ratio':>10}  {'OA base':>10}  {'OA long':>10}  {'OA ratio':>10}")
print(f"  {'-'*70}")
for i in range(10):
    wb = r_wm_base[i]["state_norms"][-1]
    wl = r_wm_long[i]["state_norms"][-1]
    wr = wl / wb if wb > 0 else 0
    ob = r_oa_base[i]["state_norms"][-1]
    ol = r_oa_long[i]["state_norms"][-1]
    orr = ol / ob if ob > 0 else 0
    flag = " <<<" if wr > 5 else ""
    print(f"  {'L'+str(i):>6}  {wb:>10.1f}  {wl:>10.1f}  {wr:>9.1f}x  {ob:>10.1f}  {ol:>10.1f}  {orr:>9.1f}x{flag}")

print("\nDone.")
