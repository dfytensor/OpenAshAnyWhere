#!/usr/bin/env python3
"""Long-Range Dependency Test only"""
import os, sys, time, math, json, torch, torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
_ORIG_ROOT = r"F:\OpenASH2605"

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src_openash'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src_wdlm'))
os.chdir(_ORIG_ROOT)

from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from wdlm_neural import WaveDynamicsLanguageModel
from open_ash import OpenASH
from open_ash_infer import sample_next_token, build_user_prompt, format_response, _sp

DEV = "cuda" if torch.cuda.is_available() else "cpu"
VOC_PATH = os.path.join(SCRIPT_DIR, "open_ash_voc_agent.json")
OPENASH_WEIGHT = os.path.join(SCRIPT_DIR, "full_sft_768_12.pth")
WDLM_WEIGHT = os.path.join(SCRIPT_DIR, "wdlm60m_sft_final.pth")
SFT_DATA = os.path.join(_ORIG_ROOT, "minimind_data", "sft_t2t_mini.jsonl")

voc = OpenASHVoc(agent_voc_path=VOC_PATH)
vs = len(voc.token_to_id) + 1

openash = OpenASH(voc_size=vs, hidden_size=768, num_heads=8, num_layers=12, model_flag="infer")
openash.load_state_dict(torch.load(OPENASH_WEIGHT, map_location=DEV), strict=False)
openash.to(DEV).eval()

wdlm = WaveDynamicsLanguageModel(vs, hidden_dim=512, num_layers=10)
ckp = torch.load(WDLM_WEIGHT, map_location=DEV)
wdlm.load_state_dict(ckp['model'] if 'model' in ckp else ckp)
wdlm.to(DEV).eval()

def _gen(model, prompt_ids, max_new=80, **kw):
    device = next(model.parameters()).device
    sp = _sp(voc)
    stop_ids = {sp["im_end"], sp["pad"]}
    x = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    if x.size(1) > 1024: x = x[:, -1024:]
    new_ids = []
    with torch.no_grad():
        state = None
        chunk = x
        for _ in range(max_new):
            if chunk.size(1) > 1024: break
            out_raw = model(chunk, state=state)
            outputs = out_raw[0] if isinstance(out_raw, tuple) else out_raw
            raw_state = out_raw[1] if isinstance(out_raw, tuple) and len(out_raw) > 1 else None
            logits = outputs[0, -1, :]
            next_id = sample_next_token(logits, new_ids, **kw)
            if next_id in stop_ids: break
            new_ids.append(next_id)
            chunk = torch.tensor([[next_id]], dtype=torch.long, device=device)
            if raw_state is not None:
                state = [s.detach() for s in raw_state]
    return new_ids

print("=" * 70)
print("  TEST 8: Long-Range Dependency")
print("=" * 70)

# --- 8a: Key-Value Retrieval ---
print("\n  --- 8a: Key-Value Retrieval (needle in haystack) ---")
filler_text = "这是一段用于填充的文本，目的是增加序列长度来测试模型的长期依赖能力。模型需要从长文本开头记住关键信息然后在末尾回答问题。"

tests = [
    ("XYZ", "blue123", 50, "XYZ"),
    ("ABC", "mars777", 100, "ABC"),
    ("QR", "quantum42", 200, "QR"),
    ("MN", "answer42", 400, "MN"),
]

print(f"  {'Dist':>5}  {'Model':>8}  {'SeqLen':>6}  {'Hit':>4}  Answer")
print(f"  {'-'*65}")

for key, val, dist, question in tests:
    kv = f"记住：{key}={val}。"
    filler = (filler_text * 10)[:dist * 3]
    full = f"{kv}{filler}问题：{question}等于什么？直接回答值。"
    pid = build_user_prompt(voc, full)

    for model, name in [(openash, "OpenASH"), (wdlm, "WDLM")]:
        ids = _gen(model, pid, 40, temperature=0.1, top_k=5, repetition_penalty=2.0)
        ans = voc.decode(ids).strip()[:50]
        hit = val in ans
        print(f"  {dist:>5}  {name:>8}  {len(pid)+len(ids):>6}  {'Y' if hit else 'N':>4}  {ans[:50]}")

# --- 8b: PPL vs Context Length ---
print("\n  --- 8b: PPL vs Context Length ---")

sft_lines = []
with open(SFT_DATA, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            obj = json.loads(line)
            convs = obj.get('conversations', [])
            ids = []
            sp_map = _sp(voc)
            for msg in convs:
                r = msg.get('role', ''); ct = msg.get('content', '')
                if r == 'user':
                    ids += [sp_map["im_start"], sp_map["user"]] + voc.encode(ct) + [sp_map["im_end"]]
                elif r == 'assistant':
                    ids += [sp_map["im_start"], sp_map["agent"]] + voc.encode(ct) + [sp_map["im_end"]]
            if len(ids) > 600: sft_lines.append(ids)
        except: pass
        if len(sft_lines) >= 100: break

print(f"  {'SeqLen':>6}  {'OpenASH':>10}  {'WDLM':>10}  {'Delta':>8}")
print(f"  {'-'*38}")

for seq_l in [64, 128, 256, 512]:
    nlls_o, nlls_w, toks = 0, 0, 0
    for ids in sft_lines[:50]:
        chunk = ids[:seq_l + 1]
        x = torch.tensor([chunk[:-1]], dtype=torch.long, device=DEV)
        t = torch.tensor([chunk[1:]], dtype=torch.long, device=DEV)
        with torch.no_grad():
            lo = openash(x, state=None)[0]
            lw = wdlm(x, state=None)[0]
        nlls_o += F.cross_entropy(lo.reshape(-1, lo.size(-1)), t.reshape(-1), ignore_index=0, reduction='sum').item()
        nlls_w += F.cross_entropy(lw.reshape(-1, lw.size(-1)), t.reshape(-1), ignore_index=0, reduction='sum').item()
        toks += max((t != 0).sum().item(), 1)
    po = math.exp(nlls_o / toks)
    pw = math.exp(nlls_w / toks)
    print(f"  {seq_l:>6}  {po:>10.2f}  {pw:>10.2f}  {pw-po:>+8.2f}")

# --- 8c: Multi-turn dependency ---
print("\n  --- 8c: Multi-Turn State Dependency ---")
print("  (Simulate multi-turn chat: PPL of turn N with state from turns 1..N-1)")

dialogue_ids = sft_lines[0][:512]
turn_sizes = [64, 128, 256, 512]

print(f"  {'Turn':>6}  {'WDLM PPL (state)':>18}  {'WDLM PPL (full)':>18}  {'Speedup':>8}")
print(f"  {'-'*56}")

for turn_end in turn_sizes:
    chunk = dialogue_ids[:turn_end]
    if len(chunk) < 4: continue
    x = torch.tensor([chunk[:-1]], dtype=torch.long, device=DEV)
    t = torch.tensor([chunk[1:]], dtype=torch.long, device=DEV)

    # Full reprocess
    with torch.no_grad():
        lw = wdlm(x, state=None)[0]
    nll_full = F.cross_entropy(lw.reshape(-1, lw.size(-1)), t.reshape(-1), ignore_index=0, reduction='sum').item()
    valid = max((t != 0).sum().item(), 1)
    ppl_full = math.exp(nll_full / valid)

    # Stateful incremental (split into chunks of 64)
    state = None
    nll_state = 0
    cs = 64
    with torch.no_grad():
        for start in range(0, len(chunk) - 1, cs):
            end = min(start + cs, len(chunk) - 1)
            c = chunk[start:end + 1]
            if len(c) < 2: break
            cx = torch.tensor([c[:-1]], dtype=torch.long, device=DEV)
            ct = torch.tensor([c[1:]], dtype=torch.long, device=DEV)
            out = wdlm(cx, state=state)
            logits = out[0]
            state = out[1]
            nll_state += F.cross_entropy(logits.reshape(-1, logits.size(-1)), ct.reshape(-1), ignore_index=0, reduction='sum').item()
            if state is not None:
                state = [s.detach() for s in state]
    ppl_state = math.exp(nll_state / valid)

    print(f"  {turn_end:>6}  {ppl_state:>18.2f}  {ppl_full:>18.2f}  {ppl_state/ppl_full:>7.2f}x")

print(f"\n{'=' * 70}")
print("  Long-Range Dependency Test Complete")
print(f"{'=' * 70}")
