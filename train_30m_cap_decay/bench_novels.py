#!/usr/bin/env python3
"""
Test OA-30M-cd on real novels — PPL at different context lengths
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

m30 = OpenASH(vs, hidden_size=432, num_heads=8, num_layers=8, model_flag="train")
m30.load_state_dict(torch.load(os.path.join(ROOT, "train_30m_cap_decay", "openash30m_cd_sft_final.pth"), map_location=DEV)["model"])
m30.to(DEV).eval()

m85 = OpenASH(vs, hidden_size=768, num_heads=8, num_layers=12, model_flag="train")
m85.load_state_dict(torch.load(os.path.join(BENCH, "full_sft_768_12.pth"), map_location=DEV))
m85.to(DEV).eval()

for m in [m30, m85]:
    for _ in range(3):
        with torch.no_grad(): m(torch.randint(1, 100, (1, 128), device=DEV), state=None)
torch.cuda.synchronize()
print("Models ready.\n")


def load_novel(path, max_chars=200000):
    with open(path, encoding='utf-8', errors='ignore') as f:
        text = f.read(max_chars)
    return text


def novel_ppl(model, text, seq_len, use_cd=False):
    ids = voc.encode(text)
    if len(ids) < seq_len + 10:
        return None
    ids = ids[:seq_len]
    n_layers = len(model.decoder_layers)
    x = torch.tensor([ids[:-1]], dtype=torch.long).to(DEV).clamp(0, vs - 1)
    t = torch.tensor([ids[1:]], dtype=torch.long).to(DEV).clamp(0, vs - 1)
    with torch.no_grad():
        states = [None] * n_layers
        nll = 0.0; ntok = 0
        for c0 in range(0, x.size(1), CHUNK):
            c = x[:, c0:c0 + CHUNK]
            tc = t[:, c0:c0 + CHUNK]
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
            nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)), tc.reshape(-1),
                                   ignore_index=0, reduction="sum").item()
            ntok += (tc != 0).sum().item()
    return math.exp(nll / max(ntok, 1))


NOVEL_DIR = r"F:\小说\女生小说"
novels = [
    "傲世九重天-风凌天下.txt",
    "奥术神座-爱潜水的乌贼.txt",
    "百炼成仙-幻雨.txt",
    "八零小富婆-风夜晚晚.txt",
    "霸道总裁宠鲜妻-衣林夕.txt",
]

seqs = [512, 1024, 4096, 16384, 65536]

print(f"{'='*110}")
print(f"  Novel PPL Test: OA-30M-cd (cap+decay trained) vs OA-85M (baseline)")
print(f"  cap={STATE_CAP}, decay={STATE_DECAY}")
print(f"{'='*110}")

for novel_name in novels:
    path = os.path.join(NOVEL_DIR, novel_name)
    if not os.path.exists(path):
        print(f"  SKIP: {novel_name} not found")
        continue

    print(f"\n  [{novel_name[:50]}]")
    text = load_novel(path)
    n_tokens = len(voc.encode(text))
    print(f"  Text: {len(text):,} chars -> ~{n_tokens:,} tokens", flush=True)

    print(f"  {'Seq':>7}  {'30M-cd':>10}  {'85M-base':>10}  {'30M退化':>8}")
    print(f"  {'-'*45}")

    base = None
    for sl in seqs:
        if n_tokens < sl:
            continue
        t0 = time.time()
        p30 = novel_ppl(m30, text, sl, use_cd=True)
        p85 = novel_ppl(m85, text, sl, use_cd=False)
        if p30 is None or p85 is None:
            continue
        if base is None:
            base = p30
        deg = f"{p30 / base:.2f}x"
        label = f"{sl // 1024}K" if sl >= 1024 else str(sl)
        elapsed = time.time() - t0
        print(f"  {label:>7}  {p30:>10.1f}  {p85:>10.1f}  {deg:>8}  ({elapsed:.0f}s)")
        sys.stdout.flush()

print("\nDone.")
