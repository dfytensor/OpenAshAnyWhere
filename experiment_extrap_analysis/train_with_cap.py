#!/usr/bin/env python3
"""
Training with state norm cap — verify loss converges normally.
Quick training: WDLM-60M config, 200 steps, compare baseline vs capped.
"""
import os, sys, time, math, json, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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
STATE_CAP = {i: 200 for i in range(10)}

# Collect training data
print("Loading data...", flush=True)
samples = []
with open(SFT_DATA, encoding="utf-8") as f:
    for i, line in enumerate(f):
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
            if len(ids) >= 512: samples.append(torch.tensor(ids[:512], dtype=torch.long))
        except: pass
        if len(samples) >= 2000: break
print(f"  {len(samples)} samples\n", flush=True)


class SeqDataset(Dataset):
    def __init__(self, samples):
        self.data = samples
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        s = self.data[idx]
        return s[:-1], s[1:]


def train_run(label, use_state_cap=False, steps=200, lr=3e-4, bs=8):
    """Train WDLM from scratch for N steps, report loss."""
    print(f"\n{'='*70}")
    print(f"  Training: {label}")
    print(f"  steps={steps}, lr={lr}, bs={bs}, state_cap={use_state_cap}")
    print(f"{'='*70}")

    model = WaveDynamicsLanguageModel(vs, hidden_dim=512, num_layers=10).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda')

    dl = DataLoader(SeqDataset(samples), batch_size=bs, shuffle=True, num_workers=0, drop_last=True)

    model.train()
    step = 0
    losses = []
    t0 = time.time()

    for epoch in range(100):
        for x, t in dl:
            x, t = x.to(DEV), t.to(DEV)
            opt.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Chunked forward with optional state cap
                states = [None] * len(model.layers)
                chunk_logits = []
                for c0 in range(0, x.size(1), CHUNK):
                    c = x[:, c0:c0+CHUNK].clamp(0, vs-1)
                    h = model.encoder(c)
                    for i, layer in enumerate(model.layers):
                        h, s = layer(h, states[i])
                        if use_state_cap and i in STATE_CAP:
                            with torch.no_grad():
                                sn = s.norm(dim=-1, keepdim=True)
                                mask = sn > STATE_CAP[i]
                                if mask.any():
                                    s = s * torch.where(mask, STATE_CAP[i] / sn.clamp(min=1e-8), torch.ones_like(sn))
                        states[i] = s
                    chunk_logits.append(model.head(h))
                logits = torch.cat(chunk_logits, dim=1)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), t.reshape(-1), ignore_index=0)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            step += 1
            losses.append(loss.item())

            if step % 20 == 0:
                avg = sum(losses[-20:]) / 20
                elapsed = time.time() - t0
                print(f"  step {step:>4}  loss={avg:.4f}  ({elapsed:.1f}s)")
                sys.stdout.flush()

            if step >= steps:
                break
        if step >= steps:
            break

    final_avg = sum(losses[-50:]) / min(50, len(losses))
    print(f"  Final avg loss (last 50): {final_avg:.4f}")
    return model, losses


# Run both
model_base, losses_base = train_run("Baseline (no state cap)", use_state_cap=False)
model_cap, losses_cap = train_run("State cap (norm<=200)", use_state_cap=True)

# ============================================================
# Compare: short-sequence PPL
# ============================================================
print(f"\n{'='*70}")
print("  Short-sequence PPL comparison (seq=512, 50 samples)")
print(f"{'='*70}")

def eval_ppl(model, use_cap=False):
    model.eval()
    nll, ntok, cnt = 0, 0, 0
    with torch.no_grad():
        for s in samples[:50]:
            x, t = s[:-1].unsqueeze(0).to(DEV), s[1:].unsqueeze(0).to(DEV)
            states = [None] * len(model.layers)
            cl = []
            for c0 in range(0, x.size(1), CHUNK):
                c = x[:, c0:c0+CHUNK]
                h = model.encoder(c)
                for i, layer in enumerate(model.layers):
                    h, s = layer(h, states[i])
                    if use_cap and i in STATE_CAP:
                        sn = s.norm()
                        if sn > STATE_CAP[i]: s = s * (STATE_CAP[i] / sn)
                    states[i] = s.detach()
                cl.append(model.head(h))
            clo = torch.cat(cl, dim=1)
            nll += F.cross_entropy(clo.reshape(-1, clo.size(-1)), t.reshape(-1), ignore_index=0, reduction="sum").item()
            ntok += max((t != 0).sum().item(), 1)
            cnt += 1
    return math.exp(nll / ntok)

ppl_base = eval_ppl(model_base, use_cap=False)
ppl_base_cap = eval_ppl(model_base, use_cap=True)
ppl_cap = eval_ppl(model_cap, use_cap=False)
ppl_cap_cap = eval_ppl(model_cap, use_cap=True)

print(f"  Model                | Eval: no cap  | Eval: with cap")
print(f"  {'-'*55}")
print(f"  Trained: no cap      | {ppl_base:>12.2f}  | {ppl_base_cap:>13.2f}")
print(f"  Trained: with cap    | {ppl_cap:>12.2f}  | {ppl_cap_cap:>13.2f}")

# ============================================================
# Extrapolation test
# ============================================================
print(f"\n{'='*70}")
print("  Extrapolation (trained 200 steps only)")
print(f"{'='*70}")

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
        if len(all_ids) >= 65536: break

def extrap_ppl(model, ids, sl, use_cap=False):
    model.eval()
    x = torch.tensor([ids[:sl-1]], dtype=torch.long).to(DEV)
    t = torch.tensor([ids[1:sl]], dtype=torch.long).to(DEV)
    with torch.no_grad():
        states = [None] * len(model.layers)
        cl = []
        for c0 in range(0, x.size(1), CHUNK):
            c = x[:, c0:c0+CHUNK]
            h = model.encoder(c)
            for i, layer in enumerate(model.layers):
                h, s = layer(h, states[i])
                if use_cap and i in STATE_CAP:
                    sn = s.norm()
                    if sn > STATE_CAP[i]: s = s * (STATE_CAP[i] / sn)
                states[i] = s.detach()
            cl.append(model.head(h))
        clo = torch.cat(cl, dim=1)
        nll = F.cross_entropy(clo.reshape(-1, clo.size(-1)), t.reshape(-1), ignore_index=0, reduction="sum").item()
        ntok = max((t != 0).sum().item(), 1)
    return math.exp(nll / ntok)

print(f"  {'Seq':>7}  {'base/no':>10}  {'base/cap':>10}  {'cap/no':>10}  {'cap/cap':>10}")
print(f"  {'-'*55}")
for sl in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
    if sl > len(all_ids): continue
    r = {}
    r["bn"] = extrap_ppl(model_base, all_ids, sl, use_cap=False)
    r["bc"] = extrap_ppl(model_base, all_ids, sl, use_cap=True)
    r["cn"] = extrap_ppl(model_cap, all_ids, sl, use_cap=False)
    r["cc"] = extrap_ppl(model_cap, all_ids, sl, use_cap=True)
    print(f"  {sl//1024:>4}K  {r['bn']:>10.1f}  {r['bc']:>10.1f}  {r['cn']:>10.1f}  {r['cc']:>10.1f}")
    sys.stdout.flush()

# ============================================================
# Loss curve summary
# ============================================================
print(f"\n{'='*70}")
print("  Loss Curve")
print(f"{'='*70}")
print(f"  {'Step':>6}  {'Baseline':>10}  {'With Cap':>10}")
print(f"  {'-'*30}")
for s in [10, 20, 50, 100, 150, 200]:
    if s <= len(losses_base):
        b = sum(losses_base[max(0,s-10):s]) / min(10, s)
        c = sum(losses_cap[max(0,s-10):s]) / min(10, s)
        print(f"  {s:>6}  {b:>10.4f}  {c:>10.4f}")

print("\nDone.")
