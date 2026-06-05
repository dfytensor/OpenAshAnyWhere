#!/usr/bin/env python3
"""Extrapolation to 8K+ tokens"""
import os, sys, math, json, torch, torch.nn.functional as F, time

ROOT = r"F:\OpenASH2605"
BENCH = os.path.join(ROOT, "experiment_openash_vs_wdlm", "bench")
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
PRETRAIN_DATA = os.path.join(ROOT, "minimind_data", "pretrain_data_mini.jsonl")
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

print("Models loaded.")

# Warmup
print("Warmup...", flush=True)
for m in [oa58, oa85, wm60]:
    for _ in range(5):
        x = torch.randint(1, 100, (1, 128), device=DEV)
        with torch.no_grad():
            r = m(x, state=None)
            if isinstance(r, tuple) and len(r) > 1:
                st = [s.detach() for s in r[1]]
                _ = m(torch.randint(1, 100, (1, CHUNK), device=DEV), state=st)
torch.cuda.synchronize()
print("Done.\n", flush=True)

# Collect as many tokens as possible from both SFT and pretrain
all_ids = []

def load_tokens(path, target):
    ids = []
    if not os.path.exists(path):
        return ids
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                convs = obj.get("conversations", [])
                if convs:
                    for msg in convs:
                        r = msg.get("role",""); ct = msg.get("content","")
                        if r == "user": ids += [sp["im_start"], sp["user"]] + voc.encode(ct) + [sp["im_end"]]
                        elif r == "assistant": ids += [sp["im_start"], sp["agent"]] + voc.encode(ct) + [sp["im_end"]]
                else:
                    text = obj.get("text", "")
                    if text: ids += voc.encode(text) + [sp["im_end"]]
            except: pass
            if len(ids) >= target: break
    return ids

TARGET = 16384
print("Collecting tokens...", flush=True)
all_ids = load_tokens(SFT_DATA, TARGET)
print(f"  SFT: {len(all_ids)} tokens")
if len(all_ids) < TARGET:
    extra = load_tokens(PRETRAIN_DATA, TARGET - len(all_ids))
    all_ids += extra
    print(f"  + Pretrain: {len(all_ids)} tokens total")

if len(all_ids) < 8192:
    # duplicate to reach target
    reps = (TARGET // len(all_ids)) + 1
    all_ids = (all_ids * reps)[:TARGET]
    print(f"  Duplicated to: {len(all_ids)} tokens")

print(f"  Final: {len(all_ids)} tokens\n", flush=True)


def chunk_ppl(model, ids, sl, chunk=64):
    s = ids[:sl]
    x = torch.tensor([s[:-1]], dtype=torch.long).to(DEV)
    t = torch.tensor([s[1:]], dtype=torch.long).to(DEV)
    with torch.no_grad():
        state = None; cl = []
        for i in range(0, x.size(1), chunk):
            c = x[:, i:i+chunk]
            r = model(c, state=state)
            lo = r[0] if isinstance(r, tuple) else r
            state = r[1] if isinstance(r, tuple) and len(r) > 1 else None
            if state is not None: state = [s2.detach() for s2 in state]
            cl.append(lo)
        clo = torch.cat(cl, dim=1)
        nll = F.cross_entropy(clo.reshape(-1, clo.size(-1)), t.reshape(-1), ignore_index=0, reduction="sum").item()
        ntok = max((t != 0).sum().item(), 1)
    return math.exp(nll / ntok)


models = [
    ("OA-58M", oa58, 768),
    ("WDLM-60M", wm60, 1024),
    ("OA-85M", oa85, 1024),
]
seqs = [256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384]
seqs = [s for s in seqs if s <= len(all_ids)]

print("=" * 80)
print("  Extrapolation: chunked PPL (chunk=64) — Extended to 16K")
print("  Train limits: OA-58M=768, WDLM-60M=1024, OA-85M=1024")
print("=" * 80)
hdr = f"  {'Seq':>5}  {'OA-58M':>10}  {'WDLM-60M':>10}  {'OA-85M':>10}"
print(hdr)
print(f"  {'-'*50}")

for sl in seqs:
    row = f"  {sl:>5}"
    for name, m, lim in models:
        t0 = time.time()
        p = chunk_ppl(m, all_ids, sl, CHUNK)
        dt = time.time() - t0
        mark = " *" if sl > lim else "  "
        row += f"  {p:>8.1f}{mark}"
    print(row)
    sys.stdout.flush()

print()
print("  * = beyond training length")
print("Done.")
