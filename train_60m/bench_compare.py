#!/usr/bin/env python3
import os, sys, time, torch

sys.path.insert(0, 'F:/OpenASH2605')
sys.path.insert(0, 'F:/OpenASH2605/wdlm_verification')
os.chdir('F:/OpenASH2605')

from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from wdlm_neural import WaveDynamicsLanguageModel
from open_ash import OpenASH
from open_ash_infer import (
    sample_next_token, build_user_prompt, format_response, _sp
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1

TEST_PROMPTS = [
    ("你好，请介绍一下你自己。", None),
    ("什么是人工智能？", None),
    ("请用Python写一个冒泡排序算法。", "你是一个有用的编程助手。"),
    ("解释量子计算的基本原理。", None),
    ("中国的首都是哪里？", None),
]

MAX_NEW = 200
GEN_CFG = dict(temperature=0.5, top_k=30, top_p=0.85, repetition_penalty=1.35)


def _generate(model, tokenizer, prompt_ids, max_new, stateful, **cfg):
    device = next(model.parameters()).device
    sp = _sp(tokenizer)
    stop_ids = {sp["im_end"], sp["pad"]}

    input_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    max_seq = getattr(model, '_max_seq', 1024) if stateful else 8192
    if input_tensor.size(1) > max_seq:
        input_tensor = input_tensor[:, -max_seq:]

    new_ids = []
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        state = None
        chunk = input_tensor
        for _ in range(max_new):
            if chunk.size(1) > max_seq:
                break
            if stateful:
                outputs, state = model(chunk, state=state)
            else:
                out_raw = model(chunk, state=None)
                outputs = out_raw[0] if isinstance(out_raw, tuple) else out_raw
                state = out_raw[1] if isinstance(out_raw, tuple) and len(out_raw) > 1 else None
            logits = outputs[0, -1, :]
            next_id = sample_next_token(logits, new_ids, **cfg)
            if next_id in stop_ids:
                break
            new_ids.append(next_id)
            chunk = torch.tensor([[next_id]], dtype=torch.long, device=device)
            if state is not None:
                state = [s.detach() for s in state]
    elapsed = time.perf_counter() - t0
    return new_ids, elapsed


def _count_params(m):
    return sum(p.numel() for p in m.parameters())


print("=" * 70)
print("  WDLM-Neural 60M  vs  OpenASH 768  —  Inference Benchmark")
print("=" * 70)

# --- Load OpenASH ---
print("\n[1] Loading OpenASH (H=768, L=12)...")
t0 = time.perf_counter()
openash = OpenASH(voc_size=vs, hidden_size=768, num_heads=8, num_layers=12, model_flag="infer")
wpath = os.path.join(r"F:\OpenASH\out", "full_sft_768_12.pth")
openash.load_state_dict(torch.load(wpath, map_location=DEV), strict=False)
openash.to(DEV).eval()
t_openash_load = time.perf_counter() - t0
p_openash = _count_params(openash)
print(f"  Params: {p_openash:,}  |  Load: {t_openash_load:.1f}s")

# --- Load WDLM ---
print("\n[2] Loading WDLM-Neural (H=512, L=10)...")
t0 = time.perf_counter()
wdlm = WaveDynamicsLanguageModel(vs, hidden_dim=512, num_layers=10)
wpath2 = r"F:\OpenASH2605\train_60m\wdlm60m_sft_final.pth"
ckp = torch.load(wpath2, map_location=DEV)
wdlm.load_state_dict(ckp['model'] if 'model' in ckp else ckp)
wdlm.to(DEV).eval()
t_wdlm_load = time.perf_counter() - t0
p_wdlm = _count_params(wdlm)
print(f"  Params: {p_wdlm:,}  |  Load: {t_wdlm_load:.1f}s")

# --- GPU Memory ---
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
mem_base = torch.cuda.memory_allocated() / 1024**2
_ = openash(torch.tensor([[1]], device=DEV))
mem_openash = torch.cuda.max_memory_allocated() / 1024**2

torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
_ = wdlm(torch.tensor([[1]], device=DEV))
mem_wdlm = torch.cuda.max_memory_allocated() / 1024**2

print(f"\n  GPU Memory — OpenASH: {mem_openash:.0f} MB  |  WDLM: {mem_wdlm:.0f} MB")

# --- Warmup ---
print("\n[Warmup]...")
_ = _generate(openash, voc, build_user_prompt(voc, "test"), 20, stateful=False, **GEN_CFG)
_ = _generate(wdlm, voc, build_user_prompt(voc, "test"), 20, stateful=True, **GEN_CFG)

# --- Bench ---
print(f"\n{'=' * 70}")
print(f"  Prompts: {len(TEST_PROMPTS)}  |  max_new_tokens: {MAX_NEW}")
print(f"  cfg: T={GEN_CFG['temperature']} top_k={GEN_CFG['top_k']} top_p={GEN_CFG['top_p']}")
print(f"{'=' * 70}")

speeds_openash = []
speeds_wdlm = []

for i, (text, sys_) in enumerate(TEST_PROMPTS):
    prompt_ids = build_user_prompt(voc, text, system_text=sys_)
    print(f"\n--- Prompt {i+1}: {text[:40]}... ---")

    # OpenASH
    ids_o, t_o = _generate(openash, voc, prompt_ids, MAX_NEW, stateful=False, **GEN_CFG)
    res_o = format_response(voc, ids_o)
    tok_s_o = len(ids_o) / t_o if t_o > 0 else 0
    speeds_openash.append(tok_s_o)
    out_o = res_o.get("content", "")[:200].replace("\n", " ")

    # WDLM
    ids_w, t_w = _generate(wdlm, voc, prompt_ids, MAX_NEW, stateful=True, **GEN_CFG)
    res_w = format_response(voc, ids_w)
    tok_s_w = len(ids_w) / t_w if t_w > 0 else 0
    speeds_wdlm.append(tok_s_w)
    out_w = res_w.get("content", "")[:200].replace("\n", " ")

    print(f"  [OpenASH] {len(ids_o):>3d} tok | {t_o:.2f}s | {tok_s_o:>7.1f} tok/s")
    print(f"    {out_o}")
    print(f"  [WDLM   ] {len(ids_w):>3d} tok | {t_w:.2f}s | {tok_s_w:>7.1f} tok/s")
    print(f"    {out_w}")

# --- Summary ---
avg_o = sum(speeds_openash) / len(speeds_openash)
avg_w = sum(speeds_wdlm) / len(speeds_wdlm)
ratio = avg_w / avg_o if avg_o > 0 else 0

print(f"\n{'=' * 70}")
print(f"  SUMMARY")
print(f"{'=' * 70}")
print(f"  {'Model':<20} {'Params':>12} {'GPU Mem':>10} {'Avg tok/s':>12}")
print(f"  {'-'*54}")
print(f"  {'OpenASH 768/12':<20} {p_openash:>12,} {mem_openash:>8.0f}MB {avg_o:>10.1f}")
print(f"  {'WDLM-Neural 512/10':<20} {p_wdlm:>12,} {mem_wdlm:>8.0f}MB {avg_w:>10.1f}")
print(f"  {'-'*54}")
print(f"  Speed ratio (WDLM/OpenASH): {ratio:.2f}x")
print(f"  Param ratio (WDLM/OpenASH): {p_wdlm/p_openash:.2%}")
print(f"{'=' * 70}")
