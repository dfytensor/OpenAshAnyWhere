#!/usr/bin/env python3
"""State + Extrapolation benchmark — v3, fast"""
import os, sys, time, math, json, torch, torch.nn.functional as F

ROOT = r"F:\OpenASH2605"
BENCH = os.path.join(ROOT, "experiment_openash_vs_wdlm", "bench")
sys.path.insert(0, ROOT)
sys.path.insert(0, BENCH)
os.chdir(ROOT)

from open_ash import OpenASH
from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from open_ash_infer import _sp, sample_next_token, build_user_prompt
sys.path.insert(0, os.path.join(ROOT, "wdlm_verification"))
from wdlm_neural import WaveDynamicsLanguageModel

DEV = "cuda"
voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1
sp = _sp(voc)
stop = {sp["im_end"], sp["pad"]}
CFG = dict(temperature=0.5, top_k=30, top_p=0.85, repetition_penalty=1.35)
SFT_DATA = os.path.join(ROOT, "minimind_data", "sft_t2t_mini.jsonl")
CHUNK = 64

oa = OpenASH(vs, hidden_size=640, num_heads=8, num_layers=10, model_flag="train")
_ckp = torch.load(os.path.join(BENCH, "openash60m_sft_final.pth"), map_location=DEV)
oa.load_state_dict(_ckp["model"])
oa.to(DEV).eval()

wm = WaveDynamicsLanguageModel(vs, hidden_dim=512, num_layers=10)
_ckp2 = torch.load(os.path.join(BENCH, "wdlm60m_sft_final.pth"), map_location=DEV)
wm.load_state_dict(_ckp2["model"] if "model" in _ckp2 else _ckp2)
wm.to(DEV).eval()
print("Loaded.\n")


def collect_samples(seq_len, n=10):
    out = []
    with open(SFT_DATA, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                convs = obj.get("conversations", [])
                ids = []
                for msg in convs:
                    r = msg.get("role", ""); ct = msg.get("content", "")
                    if r == "user": ids += [sp["im_start"], sp["user"]] + voc.encode(ct) + [sp["im_end"]]
                    elif r == "assistant": ids += [sp["im_start"], sp["agent"]] + voc.encode(ct) + [sp["im_end"]]
                if len(ids) >= seq_len: out.append(torch.tensor(ids[:seq_len], dtype=torch.long))
            except: pass
            if len(out) >= n: break
    return out


def gen_state(model, pid, max_new=100, **kw):
    dev = next(model.parameters()).device
    x = torch.tensor([pid], dtype=torch.long).to(dev)
    if x.size(1) > 1024: x = x[:, -1024:]
    ids = []
    with torch.no_grad():
        state, chunk = None, x
        for _ in range(max_new):
            if chunk.size(1) > 1024: break
            r = model(chunk, state=state)
            logits = r[0] if isinstance(r, tuple) else r
            state = r[1] if isinstance(r, tuple) and len(r) > 1 else None
            nid = sample_next_token(logits[0, -1], ids, **kw)
            if nid in stop: break
            ids.append(nid)
            chunk = torch.tensor([[nid]], dtype=torch.long, device=dev)
            if state is not None: state = [s.detach() for s in state]
    return ids


# ============================================================
# 1. State Speed (state mode only — O(1) per step)
# ============================================================
print("=" * 60)
print("  1. State Generation Speed (100 tok, state mode)")
print("=" * 60)
pid = build_user_prompt(voc, "你好，请介绍一下你自己。")

for label, m in [("OpenASH", oa), ("WDLM", wm)]:
    gen_state(m, pid, 30, **CFG)  # warmup
    times = []
    for _ in range(3):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        ids = gen_state(m, pid, 100, **CFG)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        times.append(len(ids) / dt if dt > 0 else 0)
    med = sorted(times)[1]
    print(f"  {label:8s} {med:.1f} tok/s (median of 3, ~{len(ids)} tok)")

# ============================================================
# 2. State Accuracy: chunked vs full PPL (seq=512)
# ============================================================
print()
print("=" * 60)
print("  2. State Accuracy: chunk=64 vs full (seq=512)")
print("=" * 60)
s512 = collect_samples(512, 10)

for label, m in [("OpenASH", oa), ("WDLM", wm)]:
    full_nll, chunk_nll, ntok = 0, 0, 0
    with torch.no_grad():
        for s in s512:
            x, t = s[:-1].unsqueeze(0).to(DEV), s[1:].unsqueeze(0).to(DEV)
            r = m(x, state=None); lo = r[0] if isinstance(r, tuple) else r
            full_nll += F.cross_entropy(lo.reshape(-1,lo.size(-1)), t.reshape(-1), ignore_index=0, reduction="sum").item()

            state = None; cl = []
            for i in range(0, x.size(1), CHUNK):
                c = x[:, i:i+CHUNK]
                r2 = m(c, state=state); lo2 = r2[0] if isinstance(r2, tuple) else r2
                state = r2[1] if isinstance(r2, tuple) and len(r2) > 1 else None
                if state is not None: state = [s2.detach() for s2 in state]
                cl.append(lo2)
            clo = torch.cat(cl, dim=1)
            chunk_nll += F.cross_entropy(clo.reshape(-1,clo.size(-1)), t.reshape(-1), ignore_index=0, reduction="sum").item()
            ntok += max((t != 0).sum().item(), 1)

    fp = math.exp(full_nll/ntok); cp = math.exp(chunk_nll/ntok)
    print(f"  {label:8s} full={fp:.2f}  chunk={cp:.2f}  ratio={cp/fp:.4f}x")

# ============================================================
# 3. Extrapolation: PPL vs seq_len
# ============================================================
print()
print("=" * 60)
print("  3. Extrapolation: PPL vs Context Length")
print("=" * 60)
print(f"  {'Seq':>5}  {'OpenASH':>10}  {'WDLM':>10}  {'Delta':>8}")
print(f"  {'-'*38}")
sx = collect_samples(1024, 10)

for sl in [64, 128, 256, 512, 768, 1024]:
    ppls = {}
    for label, m in [("OA", oa), ("WM", wm)]:
        nll, ntok = 0, 0
        with torch.no_grad():
            for s in sx:
                if len(s) < sl+1: continue
                x, t = s[:sl].unsqueeze(0).to(DEV), s[1:sl+1].unsqueeze(0).to(DEV)
                r = m(x, state=None); lo = r[0] if isinstance(r, tuple) else r
                nll += F.cross_entropy(lo.reshape(-1,lo.size(-1)), t.reshape(-1), ignore_index=0, reduction="sum").item()
                ntok += max((t != 0).sum().item(), 1)
        ppls[label] = math.exp(nll/ntok)
    o, w = ppls["OA"], ppls["WM"]
    print(f"  {sl:>5}  {o:>10.2f}  {w:>10.2f}  {w-o:>+7.2f}")

# ============================================================
# 4. State extrapolation: chunked PPL at long seq
# ============================================================
print()
print("=" * 60)
print("  4. State Extrapolation: chunk=64 at long seq")
print("=" * 60)
print(f"  {'Seq':>5}  {'OA-full':>8} {'OA-chunk':>10} {'WM-full':>8} {'WM-chunk':>10}")
print(f"  {'-'*45}")

for sl in [512, 768, 1024]:
    row = {}
    for label, m in [("OA", oa), ("WM", wm)]:
        full_nll, chunk_nll, ntok = 0, 0, 0
        with torch.no_grad():
            for s in sx:
                if len(s) < sl+1: continue
                x, t = s[:sl].unsqueeze(0).to(DEV), s[1:sl+1].unsqueeze(0).to(DEV)
                r = m(x, state=None); lo = r[0] if isinstance(r, tuple) else r
                full_nll += F.cross_entropy(lo.reshape(-1,lo.size(-1)), t.reshape(-1), ignore_index=0, reduction="sum").item()

                state = None; cl = []
                for i in range(0, x.size(1), CHUNK):
                    c = x[:, i:i+CHUNK]
                    r2 = m(c, state=state); lo2 = r2[0] if isinstance(r2, tuple) else r2
                    state = r2[1] if isinstance(r2, tuple) and len(r2) > 1 else None
                    if state is not None: state = [s2.detach() for s2 in state]
                    cl.append(lo2)
                clo = torch.cat(cl, dim=1)
                chunk_nll += F.cross_entropy(clo.reshape(-1,clo.size(-1)), t.reshape(-1), ignore_index=0, reduction="sum").item()
                ntok += max((t != 0).sum().item(), 1)
        row[label+"_f"] = math.exp(full_nll/ntok)
        row[label+"_c"] = math.exp(chunk_nll/ntok)
    print(f"  {sl:>5}  {row['OA_f']:>8.2f} {row['OA_c']:>9.2f}x  {row['WM_f']:>8.2f} {row['WM_c']:>9.2f}x")

# ============================================================
# 5. Long-context top-k
# ============================================================
print()
print("=" * 60)
print("  5. Long-Context Top-k Predictions")
print("=" * 60)
for ctx_len in [256, 512, 1024]:
    padding_ids = []
    while len(padding_ids) < ctx_len:
        padding_ids += [sp["im_start"], sp["system"], sp["im_end"]]
    long_ctx = padding_ids[:ctx_len] + [sp["im_start"], sp["user"]] + voc.encode("请说一句话") + [sp["im_end"], sp["im_start"], sp["agent"]]
    x = torch.tensor([long_ctx], dtype=torch.long).to(DEV)
    for label, m in [("OA", oa), ("WM", wm)]:
        with torch.no_grad():
            r = m(x, state=None); lo = r[0] if isinstance(r, tuple) else r
            top5 = lo[0, -1].topk(5)
            toks = [voc.id_to_token.get(str(i.item()), "?") for i in top5.indices]
            probs = ["%.3f" % p for p in top5.values.tolist()]
            print(f"  ctx={ctx_len:>4} {label}: {toks} p={probs}")
    print()

print("Done.")
