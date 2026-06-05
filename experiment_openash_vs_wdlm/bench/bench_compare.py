#!/usr/bin/env python3
"""
WDLM-Neural 60M  vs  OpenASH 84M  —  Full Benchmark Suite
自包含: 所有路径指向本目录, 从 experiment_openash_vs_wdlm/bench/ 运行

测试项:
  1. 生成速度 (state 模式, 3次中位数)
  2. Time-To-First-Token (TTFT)
  3. PPL (困惑度, 在 SFT 数据上)
  4. 序列长度扩展性 (128→1024)
  5. 吞吐量 (batch=1/4/8)
  6. 生成质量 (多样性/重复率)
"""
import os, sys, time, math, json, torch, torch.nn.functional as F
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src_openash'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src_wdlm'))
os.chdir(ROOT_DIR)

from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from wdlm_neural import WaveDynamicsLanguageModel
from open_ash import OpenASH
from open_ash_infer import (
    sample_next_token, build_user_prompt, format_response, _sp
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"

VOC_PATH = os.path.join(SCRIPT_DIR, "open_ash_voc_agent.json")
OPENASH_WEIGHT = os.path.join(SCRIPT_DIR, "full_sft_768_12.pth")
WDLM_WEIGHT = os.path.join(SCRIPT_DIR, "wdlm60m_sft_final.pth")
SFT_DATA_PATH = os.path.join(ROOT_DIR, "minimind_data", "sft_t2t_mini.jsonl")
PRETRAIN_DATA_PATH = os.path.join(ROOT_DIR, "minimind_data", "pretrain_t2t_mini.jsonl")

_ORIG_ROOT = r"F:\OpenASH2605"
if not os.path.exists(SFT_DATA_PATH):
    SFT_DATA_PATH = os.path.join(_ORIG_ROOT, "minimind_data", "sft_t2t_mini.jsonl")
if not os.path.exists(PRETRAIN_DATA_PATH):
    PRETRAIN_DATA_PATH = os.path.join(_ORIG_ROOT, "minimind_data", "pretrain_t2t_mini.jsonl")

TEST_PROMPTS = [
    ("你好，请介绍一下你自己。", None),
    ("什么是人工智能？", None),
    ("请用Python写一个冒泡排序算法。", "你是一个有用的编程助手。"),
    ("解释量子计算的基本原理。", None),
    ("中国的首都是哪里？", None),
    ("写一首关于春天的诗。", None),
    ("1+1等于几？", None),
    ("请列举五种水果的名称。", None),
]

MAX_NEW = 200
GEN_CFG = dict(temperature=0.5, top_k=30, top_p=0.85, repetition_penalty=1.35)


# ============================================================
# Helpers
# ============================================================
def _generate(model, tokenizer, prompt_ids, max_new, **cfg):
    device = next(model.parameters()).device
    sp = _sp(tokenizer)
    stop_ids = {sp["im_end"], sp["pad"]}

    input_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    if input_tensor.size(1) > 1024:
        input_tensor = input_tensor[:, -1024:]

    new_ids = []
    model.eval()
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    with torch.no_grad():
        state = None
        chunk = input_tensor
        for _ in range(max_new):
            if chunk.size(1) > 1024:
                break
            out_raw = model(chunk, state=state)
            outputs = out_raw[0] if isinstance(out_raw, tuple) else out_raw
            raw_state = out_raw[1] if isinstance(out_raw, tuple) and len(out_raw) > 1 else None
            logits = outputs[0, -1, :]
            next_id = sample_next_token(logits, new_ids, **cfg)
            if next_id in stop_ids:
                break
            new_ids.append(next_id)
            chunk = torch.tensor([[next_id]], dtype=torch.long, device=device)
            if raw_state is not None:
                state = [s.detach() for s in raw_state]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return new_ids, elapsed


def _ttft(model, tokenizer, prompt_ids):
    device = next(model.parameters()).device
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    if input_tensor.size(1) > 1024:
        input_tensor = input_tensor[:, -1024:]
    model.eval()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_raw = model(input_tensor, state=None)
        outputs = out_raw[0] if isinstance(out_raw, tuple) else out_raw
        logits = outputs[0, -1, :]
        _ = logits.argmax().item()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _ppl(model, tokenizer, data_path, n_samples=500, seq_len=512):
    device = next(model.parameters()).device
    sp = _sp(tokenizer)
    im_s = sp["im_start"]; im_e = sp["im_end"]
    uid_ = sp["user"]; aid_ = sp["agent"]

    samples = []
    with open(data_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                convs = obj.get('conversations', [])
                ids = []
                for msg in convs:
                    r = msg.get('role', ''); ct = msg.get('content', '')
                    if r == 'user': ids += [im_s, uid_] + tokenizer.encode(ct) + [im_e]
                    elif r == 'assistant':
                        ids += [im_s, aid_]
                        if msg.get('reasoning_content'):
                            ids += [sp["think_s"]] + tokenizer.encode(msg['reasoning_content']) + [sp["think_e"]]
                        ids += tokenizer.encode(ct) + [im_e]
                if len(ids) >= seq_len + 1:
                    samples.append(torch.tensor(ids[:seq_len+1], dtype=torch.long))
            except: pass
            if len(samples) >= n_samples: break

    if not samples:
        return float('inf'), 0

    total_nll = 0.0
    total_tok = 0
    model.eval()
    with torch.no_grad():
        for s in samples:
            x = s[:-1].unsqueeze(0).to(device)
            t = s[1:].unsqueeze(0).to(device)
            out_raw = model(x, state=None)
            logits = out_raw[0] if isinstance(out_raw, tuple) else out_raw
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), t.reshape(-1),
                                  ignore_index=0, reduction='sum')
            valid = (t != 0).sum().item()
            total_nll += nll.item()
            total_tok += max(valid, 1)

    ppl = math.exp(total_nll / total_tok)
    return ppl, len(samples)


def _throughput(model, seq_tensor, n_steps=50):
    device = next(model.parameters()).device
    model.eval()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_steps):
            out_raw = model(seq_tensor, state=None)
            _ = out_raw[0] if isinstance(out_raw, tuple) else out_raw
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    total_tok = seq_tensor.numel() * n_steps
    return total_tok / elapsed


def _seq_scaling(model, tokenizer, seq_lens):
    device = next(model.parameters()).device
    base_ids = build_user_prompt(tokenizer, "测试序列长度扩展性能。") * 5
    results = {}
    model.eval()
    for sl in seq_lens:
        ids = (base_ids * (sl // len(base_ids) + 1))[:sl]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(10):
                out_raw = model(x, state=None)
                _ = out_raw[0] if isinstance(out_raw, tuple) else out_raw
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tok_per_s = x.numel() * 10 / elapsed
        results[sl] = tok_per_s
    return results


def _quality_metrics(token_ids):
    if not token_ids: return {"unique_ratio": 0, "repeat_3gram": 1.0, "entropy": 0}
    total = len(token_ids)
    unique = len(set(token_ids))
    unique_ratio = unique / total
    trigrams = []
    for i in range(len(token_ids) - 2):
        trigrams.append((token_ids[i], token_ids[i+1], token_ids[i+2]))
    repeat_3gram = 1 - len(set(trigrams)) / max(len(trigrams), 1)
    counts = Counter(token_ids)
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * math.log(p + 1e-10) for p in probs)
    return {"unique_ratio": unique_ratio, "repeat_3gram": repeat_3gram, "entropy": entropy}


def _count_params(m):
    return sum(p.numel() for p in m.parameters())


def _model_size_mb(path):
    return os.path.getsize(path) / 1024 / 1024


def _gpu_mem_peak(model, seq_len=256):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    ids = torch.randint(1, 100, (1, seq_len), device=DEV)
    with torch.no_grad():
        _ = model(ids, state=None)
    return torch.cuda.max_memory_allocated() / 1024**2


# ============================================================
# Main
# ============================================================
print("=" * 70)
print("  WDLM-Neural 60M  vs  OpenASH 84M  —  Full Benchmark Suite")
print("=" * 70)

# --- Load ---
print("\n[0] Loading vocabulary...")
voc = OpenASHVoc(agent_voc_path=VOC_PATH)
vs = len(voc.token_to_id) + 1
print(f"  Vocab size: {vs}")

print("\n[1] Loading OpenASH (H=768, L=12)...")
openash = OpenASH(voc_size=vs, hidden_size=768, num_heads=8, num_layers=12, model_flag="infer")
openash.load_state_dict(torch.load(OPENASH_WEIGHT, map_location=DEV), strict=False)
openash.to(DEV).eval()
p_openash = _count_params(openash)
sz_openash = _model_size_mb(OPENASH_WEIGHT)
print(f"  Params: {p_openash:,}  |  File: {sz_openash:.0f} MB")

print("\n[2] Loading WDLM-Neural (H=512, L=10)...")
wdlm = WaveDynamicsLanguageModel(vs, hidden_dim=512, num_layers=10)
ckp = torch.load(WDLM_WEIGHT, map_location=DEV)
wdlm.load_state_dict(ckp['model'] if 'model' in ckp else ckp)
wdlm.to(DEV).eval()
p_wdlm = _count_params(wdlm)
sz_wdlm = _model_size_mb(WDLM_WEIGHT)
print(f"  Params: {p_wdlm:,}  |  File: {sz_wdlm:.0f} MB")

# ============================================================
# Test 1: Generation Speed (state)
# ============================================================
print("\n" + "=" * 70)
print("  TEST 1: Generation Speed (state mode, 3 runs median)")
print("=" * 70)

for _ in range(3):
    _ = _generate(openash, voc, build_user_prompt(voc, "warmup"), 30, **GEN_CFG)
    _ = _generate(wdlm, voc, build_user_prompt(voc, "warmup"), 30, **GEN_CFG)

speeds_o, speeds_w = [], []
for i, (text, sys_) in enumerate(TEST_PROMPTS):
    prompt_ids = build_user_prompt(voc, text, system_text=sys_)
    so, sw = [], []
    for _ in range(3):
        ids_o, t_o = _generate(openash, voc, prompt_ids, MAX_NEW, **GEN_CFG)
        ids_w, t_w = _generate(wdlm, voc, prompt_ids, MAX_NEW, **GEN_CFG)
        so.append(len(ids_o) / t_o if t_o > 0 else 0)
        sw.append(len(ids_w) / t_w if t_w > 0 else 0)
    med_o = sorted(so)[1]; med_w = sorted(sw)[1]
    speeds_o.append(med_o); speeds_w.append(med_w)
    print(f"  P{i+1} OpenASH {med_o:>6.1f} | WDLM {med_w:>6.1f} tok/s | {text[:25]}...")

avg_spd_o = sum(speeds_o) / len(speeds_o)
avg_spd_w = sum(speeds_w) / len(speeds_w)
print(f"\n  >>> OpenASH: {avg_spd_o:.1f} tok/s  |  WDLM: {avg_spd_w:.1f} tok/s  |  ratio: {avg_spd_w/avg_spd_o:.2f}x")

# ============================================================
# Test 2: TTFT (Time To First Token)
# ============================================================
print("\n" + "=" * 70)
print("  TEST 2: Time To First Token (TTFT)")
print("=" * 70)

ttft_lens = [32, 64, 128, 256, 512]
print(f"  {'SeqLen':>6}  {'OpenASH':>10}  {'WDLM':>10}  {'ratio':>8}")
print(f"  {'-'*40}")
for sl in ttft_lens:
    base = build_user_prompt(voc, "这是一个用于测试TTFT的提示词。" * 10, system_text="你是助手。")
    ids_sl = (base * (sl // len(base) + 1))[:sl]
    tos, tws = [], []
    for _ in range(5):
        tos.append(_ttft(openash, voc, ids_sl))
        tws.append(_ttft(wdlm, voc, ids_sl))
    med_to = sorted(tos)[2]; med_tw = sorted(tws)[2]
    print(f"  {sl:>6}  {med_to*1000:>8.1f}ms  {med_tw*1000:>8.1f}ms  {med_tw/med_to:.2f}x")

# ============================================================
# Test 3: PPL (Perplexity)
# ============================================================
print("\n" + "=" * 70)
print("  TEST 3: Perplexity on SFT data")
print("=" * 70)

if os.path.exists(SFT_DATA_PATH):
    ppl_o, n_o = _ppl(openash, voc, SFT_DATA_PATH, n_samples=300, seq_len=512)
    ppl_w, n_w = _ppl(wdlm, voc, SFT_DATA_PATH, n_samples=300, seq_len=512)
    print(f"  OpenASH:  PPL={ppl_o:.2f}  ({n_o} samples, seq=512)")
    print(f"  WDLM:     PPL={ppl_w:.2f}  ({n_w} samples, seq=512)")
else:
    ppl_o, ppl_w = float('inf'), float('inf')
    print(f"  SFT data not found at {SFT_DATA_PATH}, skipping PPL")

# ============================================================
# Test 4: Sequence Length Scaling
# ============================================================
print("\n" + "=" * 70)
print("  TEST 4: Sequence Length Scaling (forward pass tok/s)")
print("=" * 70)

seq_lens = [64, 128, 256, 512, 768, 1024]
print(f"  {'SeqLen':>6}  {'OpenASH':>10}  {'WDLM':>10}  {'ratio':>8}")
print(f"  {'-'*40}")
base_prompt = build_user_prompt(voc, "序列长度扩展性能测试基准文本。" * 20)
for sl in seq_lens:
    ids = (base_prompt * (sl // len(base_prompt) + 1))[:sl]
    x = torch.tensor([ids], dtype=torch.long, device=DEV)
    so, sw = [], []
    for _ in range(3):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(20):
                r = openash(x, state=None)
        torch.cuda.synchronize(); so.append(x.numel() * 20 / (time.perf_counter() - t0))

        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(20):
                r = wdlm(x, state=None)
        torch.cuda.synchronize(); sw.append(x.numel() * 20 / (time.perf_counter() - t0))

    mo = sorted(so)[1]; mw = sorted(sw)[1]
    print(f"  {sl:>6}  {mo:>10.0f}  {mw:>10.0f}  {mw/mo:.2f}x")

# ============================================================
# Test 5: Batch Throughput
# ============================================================
print("\n" + "=" * 70)
print("  TEST 5: Batch Throughput (forward pass, seq=64)")
print("=" * 70)

prompt_short = build_user_prompt(voc, "测试吞吐量。")[:64]
for bs in [1, 2, 4]:
    seq_t = torch.tensor([prompt_short], dtype=torch.long, device=DEV).repeat(bs, 1)
    tos, tws = [], []
    for _ in range(3):
        tos.append(_throughput(openash, seq_t))
        tws.append(_throughput(wdlm, seq_t))
    mo = sorted(tos)[1]; mw = sorted(tws)[1]
    print(f"  batch={bs}:  OpenASH {mo:>10.0f} tok/s  |  WDLM {mw:>10.0f} tok/s  |  {mw/mo:.2f}x")

# ============================================================
# Test 6: Generation Quality
# ============================================================
print("\n" + "=" * 70)
print("  TEST 6: Generation Quality (diversity & repetition)")
print("=" * 70)

quality_prompts = [
    "请列举五种水果的名称。",
    "写一首关于春天的诗。",
    "解释什么是引力。",
    "描述一下猫的外形特征。",
]

print(f"  {'Prompt':>30}  {'Model':>8}  {'Tokens':>6}  {'Unique%':>8}  {'3gram_rep':>10}  {'Entropy':>8}")
print(f"  {'-'*80}")
for text in quality_prompts:
    pid = build_user_prompt(voc, text)
    for model, name in [(openash, "OpenASH"), (wdlm, "WDLM")]:
        ids, _ = _generate(model, voc, pid, 150, **GEN_CFG)
        qm = _quality_metrics(ids)
        print(f"  {text[:30]:>30}  {name:>8}  {len(ids):>6}  {qm['unique_ratio']*100:>7.1f}%  {qm['repeat_3gram']*100:>9.1f}%  {qm['entropy']:>8.2f}")

# ============================================================
# Test 7: GPU Memory Scaling
# ============================================================
print("\n" + "=" * 70)
print("  TEST 7: GPU Memory by Sequence Length")
print("=" * 70)

print(f"  {'SeqLen':>6}  {'OpenASH':>10}  {'WDLM':>10}")
print(f"  {'-'*30}")
for sl in [128, 256, 512, 1024]:
    mo = _gpu_mem_peak(openash, sl)
    mw = _gpu_mem_peak(wdlm, sl)
    print(f"  {sl:>6}  {mo:>8.0f}MB  {mw:>8.0f}MB")

# ============================================================
# Test 8: Long-Range Dependency
# ============================================================
print("\n" + "=" * 70)
print("  TEST 8: Long-Range Dependency")
print("=" * 70)

import json as _json

# --- 8a: Key-Value Retrieval (needle in haystack) ---
print("\n  --- 8a: Key-Value Retrieval ---")
retrieval_tests = [
    {
        "key": "XYZ",
        "value": "蓝鲸",
        "filler_len": 50,
        "question": "XYZ的值是什么？",
    },
    {
        "key": "ABC",
        "value": "火星",
        "filler_len": 100,
        "question": "ABC的值是什么？",
    },
    {
        "key": "QR",
        "value": "量子计算",
        "filler_len": 200,
        "question": "QR代表什么？",
    },
    {
        "key": "MN",
        "value": "四十二",
        "filler_len": 400,
        "question": "MN的值是多少？",
    },
]

filler_text = "这是一段用于填充的文本，目的是增加序列长度来测试模型的长期依赖能力。模型需要从长文本的开头记住关键信息，然后在末尾回答问题。"

print(f"  {'Filler':>6} {'TotalSeq':>9}  {'Model':>8}  {'Answer':>30}  {'Hit':>5}")
print(f"  {'-'*70}")

for test in retrieval_tests:
    kv_text = f"{test['key']}等于{test['value']}。"
    filler = (filler_text * (test['filler_len'] // len(filler_text) + 1))[:test['filler_len'] * 3]
    full_text = f"{kv_text}{filler}现在请回答：{test['question']}"
    prompt_ids = build_user_prompt(voc, full_text)

    for model, name in [(openash, "OpenASH"), (wdlm, "WDLM")]:
        ids, _ = _generate(model, voc, prompt_ids, 80, temperature=0.3, top_k=10, top_p=0.9, repetition_penalty=1.5)
        answer = voc.decode(ids).strip()[:60]
        hit = test['value'] in answer
        total_seq = len(prompt_ids) + len(ids)
        print(f"  {test['filler_len']:>6} {total_seq:>9}  {name:>8}  {answer[:30]:>30}  {'Y' if hit else 'N':>5}")

# --- 8b: PPL vs Context Length ---
print("\n  --- 8b: PPL vs Context Length ---")

if os.path.exists(SFT_DATA_PATH):
    sft_lines = []
    with open(SFT_DATA_PATH, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = _json.loads(line)
                convs = obj.get('conversations', [])
                ids = []
                sp_map = _sp(voc)
                for msg in convs:
                    r = msg.get('role', '')
                    ct = msg.get('content', '')
                    if r == 'user':
                        ids += [sp_map["im_start"], sp_map["user"]] + voc.encode(ct) + [sp_map["im_end"]]
                    elif r == 'assistant':
                        ids += [sp_map["im_start"], sp_map["agent"]]
                        if msg.get('reasoning_content'):
                            ids += [sp_map["think_s"]] + voc.encode(msg['reasoning_content']) + [sp_map["think_e"]]
                        ids += voc.encode(ct) + [sp_map["im_end"]]
                if len(ids) > 600:
                    sft_lines.append(ids)
            except: pass
            if len(sft_lines) >= 100: break

    print(f"  {'SeqLen':>6}  {'OpenASH PPL':>12}  {'WDLM PPL':>12}")
    print(f"  {'-'*36}")

    for seq_l in [64, 128, 256, 512]:
        nlls_o, nlls_w, toks_o, toks_w = 0, 0, 0, 0
        for ids in sft_lines[:50]:
            chunk = ids[:seq_l + 1]
            x = torch.tensor([chunk[:-1]], dtype=torch.long, device=DEV)
            t = torch.tensor([chunk[1:]], dtype=torch.long, device=DEV)

            with torch.no_grad():
                out_o = openash(x, state=None)
                log_o = out_o[0] if isinstance(out_o, tuple) else out_o
                nll_o = F.cross_entropy(log_o.reshape(-1, log_o.size(-1)), t.reshape(-1), ignore_index=0, reduction='sum')
                valid = (t != 0).sum().item()
                nlls_o += nll_o.item(); toks_o += max(valid, 1)

                out_w = wdlm(x, state=None)
                log_w = out_w[0] if isinstance(out_w, tuple) else out_w
                nll_w = F.cross_entropy(log_w.reshape(-1, log_w.size(-1)), t.reshape(-1), ignore_index=0, reduction='sum')
                nlls_w += nll_w.item(); toks_w += max(valid, 1)

        ppl_o = math.exp(nlls_o / toks_o) if toks_o > 0 else float('inf')
        ppl_w = math.exp(nlls_w / toks_w) if toks_w > 0 else float('inf')
        print(f"  {seq_l:>6}  {ppl_o:>12.2f}  {ppl_w:>12.2f}")
else:
    print("  SFT data not found, skipping PPL vs context length")

# --- 8c: State Accumulation (incremental PPL) ---
print("\n  --- 8c: State Accumulation (incremental PPL) ---")
print("  (WDLM uses stateful incremental, OpenASH uses full-sequence reprocess)")

if sft_lines:
    test_ids = sft_lines[0][:512]
    chunk_sizes = [512, 256, 128, 64, 32]

    print(f"  {'ChunkSize':>10}  {'WDLM PPL (state)':>18}  {'OpenASH PPL (full)':>20}")
    print(f"  {'-'*54}")

    for cs in chunk_sizes:
        ppls = []
        for model in [openash, wdlm]:
            total_nll, total_tok = 0, 0
            with torch.no_grad():
                for start in range(0, len(test_ids) - 1, cs):
                    end = min(start + cs, len(test_ids) - 1)
                    chunk = test_ids[start:end + 1]
                    if len(chunk) < 2: break
                    x = torch.tensor([chunk[:-1]], dtype=torch.long, device=DEV)
                    t = torch.tensor([chunk[1:]], dtype=torch.long, device=DEV)
                    # WDLM: use state accumulation; OpenASH: full reprocess (no state)
                    if model is wdlm and start > 0:
                        prev = test_ids[:start + 1]
                        full_x = torch.tensor([prev], dtype=torch.long, device=DEV)
                        out = model(full_x, state=None)
                        logits = out[0] if isinstance(out, tuple) else out
                        logits = logits[:, -(end - start):, :]
                    else:
                        out = model(x, state=None)
                        logits = out[0] if isinstance(out, tuple) else out
                    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), t.reshape(-1), ignore_index=0, reduction='sum')
                    valid = (t != 0).sum().item()
                    total_nll += nll.item()
                    total_tok += max(valid, 1)
            ppls.append(math.exp(total_nll / total_tok) if total_tok > 0 else float('inf'))

        print(f"  {cs:>10}  {ppls[1]:>18.2f}  {ppls[0]:>20.2f}")

# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
print(f"  {'Metric':<30} {'OpenASH 768/12':>18} {'WDLM 512/10':>18}")
print(f"  {'-'*66}")
print(f"  {'Parameters':<30} {p_openash:>18,} {p_wdlm:>18,}")
print(f"  {'Model File Size':<30} {sz_openash:>15.0f} MB {sz_wdlm:>15.0f} MB")
print(f"  {'Gen Speed (state, tok/s)':<30} {avg_spd_o:>18.1f} {avg_spd_w:>18.1f}")
print(f"  {'Speed Ratio':<30} {'1.00x':>18} {avg_spd_w/avg_spd_o:>17.2f}x")
if ppl_o < float('inf'):
    print(f"  {'PPL (SFT, seq=512)':<30} {ppl_o:>18.2f} {ppl_w:>18.2f}")
print(f"  {'Param Ratio':<30} {'100%':>18} {p_wdlm/p_openash:>17.1%}")
print(f"{'=' * 70}")
