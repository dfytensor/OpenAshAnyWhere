#!/usr/bin/env python3
"""
Push extrapolation to the limit: 4K → 128K+
"""
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
CHUNK = 64

oa58 = OpenASH(vs, hidden_size=640, num_heads=8, num_layers=10, model_flag="train")
oa58.load_state_dict(torch.load(os.path.join(BENCH, "openash60m_sft_final.pth"), map_location=DEV)["model"])
oa58.to(DEV).eval()

wm60 = WaveDynamicsLanguageModel(vs, hidden_dim=512, num_layers=10)
_ck = torch.load(os.path.join(BENCH, "wdlm60m_sft_final.pth"), map_location=DEV)
wm60.load_state_dict(_ck["model"] if "model" in _ck else _ck)
wm60.to(DEV).eval()

# Collect ALL available tokens
print("Collecting tokens...", flush=True)
all_ids = []

def load_jsonl(path):
    ids = []
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
    return ids

TARGET = 200000  # 200K tokens
all_ids = load_jsonl(os.path.join(ROOT, "minimind_data", "sft_t2t_mini.jsonl"))
print(f"  SFT: {len(all_ids)} tokens", flush=True)
if len(all_ids) < TARGET:
    all_ids += load_jsonl(os.path.join(ROOT, "minimind_data", "pretrain_t2t_mini.jsonl"))
    print(f"  + Pretrain: {len(all_ids)} tokens", flush=True)

print(f"  Total: {len(all_ids)} tokens ({len(all_ids)//1024}K)\n", flush=True)

# Warmup
for m in [oa58, wm60]:
    for _ in range(5):
        x = torch.randint(1, 100, (1, 128), device=DEV)
        with torch.no_grad(): m(x, state=None)
torch.cuda.synchronize()
print("Ready.\n", flush=True)


WM_CAP = {i: 200 for i in range(10)}
OA58_CAP = {0: 96, 1: 84, 2: 107, 3: 158, 4: 341, 5: 150, 6: 145, 7: 96, 8: 66, 9: 103}


def wm_ppl_fixed(ids, sl, chunk=64, cap=None):
    cap = cap or {}
    x = torch.tensor([ids[:sl-1]], dtype=torch.long).to(DEV)
    t = torch.tensor([ids[1:sl]], dtype=torch.long).to(DEV)
    with torch.no_grad():
        states = [None] * len(wm60.layers)
        cl = []
        for c0 in range(0, x.size(1), chunk):
            c = x[:, c0:c0+chunk]
            h = wm60.encoder(c)
            for i, layer in enumerate(wm60.layers):
                h, s = layer(h, states[i])
                if i in cap:
                    sn = s.norm()
                    if sn > cap[i]: s = s * (cap[i] / sn)
                states[i] = s.detach()
            cl.append(wm60.head(h))
        clo = torch.cat(cl, dim=1)
        nll = F.cross_entropy(clo.reshape(-1, clo.size(-1)), t.reshape(-1), ignore_index=0, reduction="sum").item()
        ntok = max((t != 0).sum().item(), 1)
    return math.exp(nll / ntok)


def oa_ppl_fixed(ids, sl, chunk=64, cap=None):
    cap = cap or {}
    x = torch.tensor([ids[:sl-1]], dtype=torch.long).to(DEV)
    t = torch.tensor([ids[1:sl]], dtype=torch.long).to(DEV)
    with torch.no_grad():
        states = [None] * len(oa58.decoder_layers)
        cl = []
        for c0 in range(0, x.size(1), chunk):
            c = x[:, c0:c0+chunk]
            h = oa58.em(c)
            for i, layer in enumerate(oa58.decoder_layers):
                h2, s = layer(h, states[i])
                h = h2 + h
                if i in cap and s is not None:
                    sn = s.norm()
                    if sn > cap[i]: s = s * (cap[i] / sn)
                states[i] = s.detach() if s is not None else None
            cl.append(oa58.head_score(h))
        clo = torch.cat(cl, dim=1)
        nll = F.cross_entropy(clo.reshape(-1, clo.size(-1)), t.reshape(-1), ignore_index=0, reduction="sum").item()
        ntok = max((t != 0).sum().item(), 1)
    return math.exp(nll / ntok)


seqs = [1024, 4096, 8192, 16384, 32768, 65536, 131072]
if len(all_ids) >= 200000:
    seqs.append(200000)
seqs = [s for s in seqs if s <= len(all_ids)]

print("=" * 90)
print("  Extrapolation Limit Push (4K → max)")
print("  WM-base | WM-fix | OA58-base | OA58-fix")
print("=" * 90)
print(f"  {'Seq':>7}  {'WM-base':>10}  {'WM-fix':>10}  {'OA58-base':>10}  {'OA58-fix':>10}  {'Time':>8}")
print(f"  {'-'*62}")

for sl in seqs:
    t0 = time.time()

    # WM-base (skip if too long — will be inf anyway)
    if sl <= 16384:
        wb = wm_ppl_fixed(all_ids, sl, CHUNK, cap={})
    else:
        wb = float('inf')

    wf = wm_ppl_fixed(all_ids, sl, CHUNK, cap=WM_CAP)

    ob = oa_ppl_fixed(all_ids, sl, CHUNK, cap={})
    of = oa_ppl_fixed(all_ids, sl, CHUNK, cap=OA58_CAP)

    dt = time.time() - t0
    wb_s = f"{wb:>10.1f}" if wb < 1e6 else f"{'inf':>10}"
    print(f"  {sl//1024:>4}K  {wb_s}  {wf:>10.1f}  {ob:>10.1f}  {of:>10.1f}  {dt:>6.0f}s")
    sys.stdout.flush()

# Growth ratio
print()
print("=" * 90)
print(f"  Growth ratio (PPL / PPL@1K)")
print("=" * 90)
# Rerun 1K for baseline
wb1k = wm_ppl_fixed(all_ids, 1024, CHUNK, cap={})
wf1k = wm_ppl_fixed(all_ids, 1024, CHUNK, cap=WM_CAP)
ob1k = oa_ppl_fixed(all_ids, 1024, CHUNK, cap={})
of1k = oa_ppl_fixed(all_ids, 1024, CHUNK, cap=OA58_CAP)

print(f"  {'Seq':>7}  {'WM-base':>10}  {'WM-fix':>10}  {'OA58-base':>10}  {'OA58-fix':>10}")
print(f"  {'-'*52}")
print(f"  {'1K':>7}  {'1.0x':>10}  {'1.0x':>10}  {'1.0x':>10}  {'1.0x':>10}")

for sl in seqs:
    if sl == 1024: continue
    r = {}
    if sl <= 16384:
        wb = wm_ppl_fixed(all_ids, sl, CHUNK, cap={})
        r["wb"] = wb / wb1k
    else:
        r["wb"] = float('inf')
    wf = wm_ppl_fixed(all_ids, sl, CHUNK, cap=WM_CAP)
    ob = oa_ppl_fixed(all_ids, sl, CHUNK, cap={})
    of = oa_ppl_fixed(all_ids, sl, CHUNK, cap=OA58_CAP)
    r["wf"] = wf / wf1k
    r["ob"] = ob / ob1k
    r["of"] = of / of1k

    wb_s = f"{r['wb']:>9.1f}x" if r['wb'] < 1e6 else f"{'inf':>10}"
    print(f"  {sl//1024:>4}K  {wb_s}  {r['wf']:>9.1f}x  {r['ob']:>9.1f}x  {r['of']:>9.1f}x")
    sys.stdout.flush()

print("\nDone.")
