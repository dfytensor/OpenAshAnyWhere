#!/usr/bin/env python3
"""
Ablation: clamp/normalize WDLM cummax state to prevent explosion.
Test if clamping state fixes extrapolation.
"""
import os, sys, math, json, torch, torch.nn.functional as F, copy

ROOT = r"F:\OpenASH2605"
BENCH = os.path.join(ROOT, "experiment_openash_vs_wdlm", "bench")
sys.path.insert(0, ROOT); sys.path.insert(0, BENCH); os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "wdlm_verification"))

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

# Load WDLM
wm60 = WaveDynamicsLanguageModel(vs, hidden_dim=512, num_layers=10)
_ck = torch.load(os.path.join(BENCH, "wdlm60m_sft_final.pth"), map_location=DEV)
wm60.load_state_dict(_ck["model"] if "model" in _ck else _ck)
wm60.to(DEV).eval()

# Collect data
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
for _ in range(5):
    x = torch.randint(1, 100, (1, 64), device=DEV)
    with torch.no_grad(): wm60(x, state=None)
torch.cuda.synchronize()
print("Warmup done.\n")


def chunk_ppl_custom(model, ids, sl, chunk=64, state_clamp=None, state_norm_max=None,
                     skip_layers=None, freeze_layers=None):
    """
    Custom forward with state interventions.
    state_clamp: dict {layer_idx: max_val} - clamp state to [-max, max]
    state_norm_max: dict {layer_idx: max_norm} - normalize state norm
    skip_layers: set of layer indices to skip (pass-through)
    freeze_layers: set of layer indices to freeze state (don't update)
    """
    s = ids[:sl]
    x = torch.tensor([s[:-1]], dtype=torch.long).to(DEV)
    t = torch.tensor([s[1:]], dtype=torch.long).to(DEV)

    state_clamp = state_clamp or {}
    state_norm_max = state_norm_max or {}
    skip_layers = skip_layers or set()
    freeze_layers = freeze_layers or set()

    with torch.no_grad():
        states = [None] * len(model.layers)
        chunk_logits = []

        for c_start in range(0, x.size(1), chunk):
            c = x[:, c_start:c_start+chunk]
            h = model.encoder(c)

            for i, layer in enumerate(model.layers):
                if i in skip_layers:
                    continue

                old_state = states[i]
                h, s = layer(h, states[i])

                # Interventions
                if i in state_clamp:
                    s = s.clamp(-state_clamp[i], state_clamp[i])
                if i in state_norm_max:
                    sn = s.norm()
                    if sn > state_norm_max[i]:
                        s = s * (state_norm_max[i] / sn)
                if i in freeze_layers:
                    s = old_state if old_state is not None else s

                states[i] = s.detach()

            chunk_logits.append(model.head(h))

        clo = torch.cat(chunk_logits, dim=1)
        nll = F.cross_entropy(clo.reshape(-1, clo.size(-1)), t.reshape(-1), ignore_index=0, reduction="sum").item()
        ntok = max((t != 0).sum().item(), 1)
    return math.exp(nll / ntok)


# ============================================================
# Test 1: Clamp state at different layers
# ============================================================
print("=" * 80)
print("  Test 1: State clamping ablation (clamp all layers to max=50)")
print("  WDLM-60M at seq=4096")
print("=" * 80)

seqs = [512, 1024, 2048, 4096]
seqs = [s for s in seqs if s <= len(all_ids)]

# Baseline (no intervention)
print(f"\n  {'Seq':>5}  {'Baseline':>10}", end="")

# Interventions
interventions = [
    ("clamp=50 all", {"state_clamp": {i: 50 for i in range(10)}}),
    ("clamp=30 all", {"state_clamp": {i: 30 for i in range(10)}}),
    ("clamp=20 L1-8", {"state_clamp": {i: 20 for i in range(1, 9)}}),
    ("clamp=50 L7+L8", {"state_clamp": {7: 50, 8: 50}}),
    ("clamp=30 L7+L8", {"state_clamp": {7: 30, 8: 30}}),
    ("norm=300 L7+L8", {"state_norm_max": {7: 300, 8: 300}}),
    ("norm=500 L7+L8", {"state_norm_max": {7: 500, 8: 500}}),
    ("skip L8", {"skip_layers": {8}}),
    ("freeze L8", {"freeze_layers": {8}}),
]

for name, _ in interventions:
    print(f"  {name:>14}", end="")
print()
print(f"  {'-'*(16 + 14*len(interventions))}")

for sl in seqs:
    x_data = all_ids[:sl]
    # Baseline
    base_ppl = chunk_ppl_custom(wm60, x_data, sl, CHUNK)
    print(f"  {sl:>5}  {base_ppl:>10.1f}", end="")

    for name, kwargs in interventions:
        ppl = chunk_ppl_custom(wm60, x_data, sl, CHUNK, **kwargs)
        print(f"  {ppl:>14.1f}", end="")
    print()

# ============================================================
# Test 2: Layer-by-layer clamping (only clamp one layer at a time)
# ============================================================
print()
print("=" * 80)
print("  Test 2: Single-layer clamping (clamp=50, one layer at a time)")
print("  WDLM-60M at seq=4096")
print("=" * 80)

SL = 4096 if 4096 <= len(all_ids) else len(all_ids)
print(f"\n  {'Layer clamped':>15}  {'PPL':>10}  {'vs baseline':>12}")
print(f"  {'-'*42}")

base_ppl = chunk_ppl_custom(wm60, all_ids, SL, CHUNK)
print(f"  {'(baseline)':>15}  {base_ppl:>10.1f}  {'1.0x':>12}")

for i in range(10):
    ppl = chunk_ppl_custom(wm60, all_ids, SL, CHUNK, state_clamp={i: 50})
    ratio = ppl / base_ppl
    print(f"  {'clamp L'+str(i):>15}  {ppl:>10.1f}  {ratio:>11.2f}x")

# Also test clamping L7+L8 together
ppl = chunk_ppl_custom(wm60, all_ids, SL, CHUNK, state_clamp={7: 50, 8: 50})
ratio = ppl / base_ppl
print(f"  {'clamp L7+L8':>15}  {ppl:>10.1f}  {ratio:>11.2f}x")

# Clamp all
ppl = chunk_ppl_custom(wm60, all_ids, SL, CHUNK, state_clamp={i: 50 for i in range(10)})
ratio = ppl / base_ppl
print(f"  {'clamp ALL':>15}  {ppl:>10.1f}  {ratio:>11.2f}x")

# ============================================================
# Test 3: State normalization (normalize to fixed norm)
# ============================================================
print()
print("=" * 80)
print("  Test 3: Single-layer norm capping (max_norm=200)")
print("  WDLM-60M at seq=4096")
print("=" * 80)

print(f"\n  {'Layer':>15}  {'PPL':>10}  {'vs baseline':>12}")
print(f"  {'-'*42}")

print(f"  {'(baseline)':>15}  {base_ppl:>10.1f}  {'1.0x':>12}")

for i in range(10):
    ppl = chunk_ppl_custom(wm60, all_ids, SL, CHUNK, state_norm_max={i: 200})
    ratio = ppl / base_ppl
    print(f"  {'norm L'+str(i):>15}  {ppl:>10.1f}  {ratio:>11.2f}x")

ppl = chunk_ppl_custom(wm60, all_ids, SL, CHUNK, state_norm_max={7: 200, 8: 200})
ratio = ppl / base_ppl
print(f"  {'norm L7+L8':>15}  {ppl:>10.1f}  {ratio:>11.2f}x")

ppl = chunk_ppl_custom(wm60, all_ids, SL, CHUNK, state_norm_max={i: 200 for i in range(10)})
ratio = ppl / base_ppl
print(f"  {'norm ALL':>15}  {ppl:>10.1f}  {ratio:>11.2f}x")

print("\nDone.")
