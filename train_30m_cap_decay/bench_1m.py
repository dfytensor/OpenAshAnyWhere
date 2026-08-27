#!/usr/bin/env python3
"""
OA-30M-cd extrapolation to 1M tokens
"""
import os, sys, math, torch, torch.nn.functional as F, time

ROOT = r"F:\OpenASH2605"
BENCH = os.path.join(ROOT, "experiment_openash_vs_wdlm", "bench")
sys.path.insert(0, ROOT); sys.path.insert(0, BENCH); os.chdir(ROOT)

from open_ash import OpenASH
from open_ash_voc import OpenASHVoc
from config import agent_voc_path

DEV = "cuda"
CHUNK = 64
STATE_CAP = 150
STATE_DECAY = 0.97

voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1

print("Loading data...", flush=True)
t0 = time.time()
all_ids = torch.load(os.path.join(ROOT, "train_30m_cap_decay", "ids_1m.pt"), weights_only=False)
print(f"Tokens: {len(all_ids):,} in {time.time()-t0:.1f}s", flush=True)

print("Loading models...", flush=True)

m30 = OpenASH(vs, hidden_size=432, num_heads=8, num_layers=8, model_flag="train")
m30.load_state_dict(torch.load(os.path.join(ROOT, "train_30m_cap_decay", "openash30m_cd_sft_final.pth"), map_location=DEV)["model"])
m30.to(DEV).eval()

m58 = OpenASH(vs, hidden_size=640, num_heads=8, num_layers=10, model_flag="train")
m58.load_state_dict(torch.load(os.path.join(BENCH, "openash60m_sft_final.pth"), map_location=DEV)["model"])
m58.to(DEV).eval()

m85 = OpenASH(vs, hidden_size=768, num_heads=8, num_layers=12, model_flag="train")
m85.load_state_dict(torch.load(os.path.join(BENCH, "full_sft_768_12.pth"), map_location=DEV))
m85.to(DEV).eval()

print("Ready.", flush=True)

for m in [m30, m58, m85]:
    for _ in range(3):
        x = torch.randint(1, 100, (1, 128), device=DEV)
        with torch.no_grad(): m(x, state=None)
torch.cuda.synchronize()


def oa_ppl(model, ids, sl, use_cd=False):
    n_layers = len(model.decoder_layers)
    s = ids[:sl]
    x = torch.tensor([s[:-1]], dtype=torch.long).to(DEV)
    t = torch.tensor([s[1:]], dtype=torch.long).to(DEV)
    with torch.no_grad():
        states = [None] * n_layers
        nll = 0.0
        ntok = 0
        for c0 in range(0, x.size(1), CHUNK):
            c = x[:, c0:c0+CHUNK]
            tc = t[:, c0:c0+CHUNK]
            h = model.em(c)
            for i, layer in enumerate(model.decoder_layers):
                h2, s = layer(h, states[i])
                h = h2 + h
                states[i] = s
                if use_cd and s is not None:
                    sn = s.norm()
                    if sn > STATE_CAP:
                        s = s * (STATE_CAP / sn)
                    states[i] = s * STATE_DECAY
            logits = model.head_score(h)
            nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)), tc.reshape(-1), ignore_index=0, reduction="sum").item()
            ntok += (tc != 0).sum().item()
    return math.exp(nll / max(ntok, 1))


# ============================================================
# Main test: all models with cap+decay to 1M
# ============================================================
print(f"\n{'='*100}")
print("  Extrapolation to 1M tokens (cap+decay at inference for all)")
print(f"  30M trained with cap+decay | 58M/85M: NOT trained with cap+decay")
print(f"  cap={STATE_CAP}, decay={STATE_DECAY}")
print(f"{'='*100}")

seqs = [512, 1024, 4096, 16384, 65536, 131072, 262144, 524288, 1048576]
seqs = [s for s in seqs if s <= len(all_ids)]

print(f"  {'Seq':>7}  {'30M-cd':>10}  {'58M+cd':>10}  {'85M+cd':>10}  {'30M退化':>8}  {'58M退化':>8}  {'85M退化':>8}")
print(f"  {'-'*80}")

base_ppl = {}

for sl in seqs:
    t0 = time.time()
    p30 = oa_ppl(m30, all_ids, sl, use_cd=True)
    p58 = oa_ppl(m58, all_ids, sl, use_cd=True)
    p85 = oa_ppl(m85, all_ids, sl, use_cd=True)

    if sl == 1024:
        base_ppl = {30: p30, 58: p58, 85: p85}

    d30 = f"{p30/base_ppl[30]:.2f}x" if base_ppl else "-"
    d58 = f"{p58/base_ppl[58]:.2f}x" if base_ppl else "-"
    d85 = f"{p85/base_ppl[85]:.2f}x" if base_ppl else "-"

    label = f"{sl//1024}K" if sl >= 1024 else str(sl)
    elapsed = time.time() - t0
    print(f"  {label:>7}  {p30:>10.1f}  {p58:>10.1f}  {p85:>10.1f}  {d30:>8}  {d58:>8}  {d85:>8}  ({elapsed:.0f}s)")
    sys.stdout.flush()

# ============================================================
# 30M no-cap at long seq
# ============================================================
print(f"\n{'='*100}")
print("  OA-30M-cd: with vs without cap+decay")
print(f"{'='*100}")

print(f"  {'Seq':>7}  {'no-cap':>10}  {'cap+decay':>10}")
print(f"  {'-'*35}")

for sl in [1024, 16384, 131072, 1048576]:
    if sl > len(all_ids): continue
    p_nc = oa_ppl(m30, all_ids, sl, use_cd=False)
    p_cd = oa_ppl(m30, all_ids, sl, use_cd=True)
    label = f"{sl//1024}K" if sl >= 1024 else str(sl)
    print(f"  {label:>7}  {p_nc:>10.1f}  {p_cd:>10.1f}")
    sys.stdout.flush()

print("\nDone.")
