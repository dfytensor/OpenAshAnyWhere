"""FRSM 百万级上下文 (1M tokens) 极限测试 - 分块处理版"""
import os, sys, math, torch, json, time
import torch.nn.functional as F

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, 'F:/OpenASH2605')
from config import agent_voc_path
from open_ash_voc import OpenASHVoc
from frsm.model import FractalRecursiveStateMachine

print("=== FRSM 1M Context Stress Test (Chunked) ===", flush=True)
device = torch.device("cuda")
voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1

ckpt = torch.load("frsm_checkpoints/frsm_pretrain_final.pt", map_location='cpu')
model = FractalRecursiveStateMachine(
    vocab_size=vs, d_model=ckpt.get('config_d_model', 256),
    num_scales=ckpt.get('config_num_scales', 4),
)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model = model.to(device).eval()
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

# 构建 1M token 序列
print("Building 1M token sequence...", flush=True)
TARGET = 1_000_000
all_seqs = []
with open('minimind_data/pretrain_t2t_mini.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 50000: break
        try: text = json.loads(line).get('text', '')
        except: continue
        ids = voc.encode(text)
        if len(ids) >= 32: all_seqs.append(ids)

giant = []
seq_idx = 0
while len(giant) < TARGET:
    giant.extend(all_seqs[seq_idx % len(all_seqs)])
    seq_idx += 1
giant = giant[:TARGET]
tokens_t = torch.tensor(giant, dtype=torch.long, device='cpu')  # 留CPU
print(f"Built: {len(giant):,} tokens", flush=True)

def chunked_forward(model, tokens, chunk_size=4096, return_final_state=True):
    """分块前向传播，不存储中间 logits，只保留最终状态"""
    h = [torch.zeros(1, model.d_model, device=model.embed.weight.device)
         for _ in range(model.num_scales)]
    
    total_chunks = (len(tokens) + chunk_size - 1) // chunk_size
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i+chunk_size].unsqueeze(0).to(device)
        with torch.no_grad():
            _, h, _ = model(chunk, h_prev=h, return_state=True, compute_critical_loss=False)
    return h

def chunked_forward_with_logit(model, tokens, target_pos, chunk_size=4096):
    """分块前向传播到指定位置，返回该位置的 logit"""
    h = [torch.zeros(1, model.d_model, device=model.embed.weight.device)
         for _ in range(model.num_scales)]
    processed = 0
    
    while processed + chunk_size < target_pos:
        chunk = tokens[processed:processed+chunk_size].unsqueeze(0).to(device)
        with torch.no_grad():
            _, h, _ = model(chunk, h_prev=h, return_state=True, compute_critical_loss=False)
        processed += chunk_size
    
    # 处理最后一段，取指定位置的 logit
    remaining = tokens[processed:target_pos+1].unsqueeze(0).to(device)
    with torch.no_grad():
        logits, h, _ = model(remaining, h_prev=h, return_state=True, compute_critical_loss=False)
    return logits[:, -1, :], h

# === Phase 1: Full 1M Forward Pass (chunked) ===
print(f"\n--- Phase 1: Full 1M Forward Pass (chunked, {4096} tok/chunk) ---", flush=True)

torch.cuda.synchronize()
t0 = time.time()
h_final = chunked_forward(model, tokens_t, chunk_size=4096)
torch.cuda.synchronize()
elapsed = time.time() - t0

tok_per_s = TARGET / elapsed
print(f"  1M tokens in {elapsed:.1f}s  ({tok_per_s:.0f} tok/s)", flush=True)

for s in range(model.num_scales):
    h = h_final[s]
    norm = h.norm(dim=-1).mean().item()
    std = h.std().item()
    has_nan = torch.isnan(h).any().item()
    has_inf = torch.isinf(h).any().item()
    print(f"  Scale {s} (period={2**s}): norm={norm:.4f} std={std:.4f} NaN={has_nan} Inf={has_inf}", flush=True)

# === Phase 2: PPL Spot Checks ===
print(f"\n--- Phase 2: PPL Spot Checks ---", flush=True)
checkpoints = [64, 128, 256, 512,
               1024, 2048, 4096, 8192,
               16384, 32768, 65536, 131072,
               262144, 524288, 999936]

eval_len = 64
print(f"  {'Position':>8} | {'PPL':>9} | {'Loss':>8}", flush=True)
print(f"  " + "-" * 35, flush=True)

ppl_results = []
for pos in checkpoints:
    if pos + eval_len > TARGET: continue
    
    with torch.no_grad():
        # 获取该位置的 logit
        logit, h = chunked_forward_with_logit(model, tokens_t, pos - 1, chunk_size=4096)
        tgt = tokens_t[pos:pos+eval_len].to(device)
        
        total_loss = 0.0
        for i in range(len(tgt)):
            if i == 0:
                pred = logit
            else:
                prev = tgt[i-1:i]
                pred, h = model.generate_step(prev.unsqueeze(0), h)
            total_loss += F.cross_entropy(pred, tgt[i:i+1], reduction='sum').item()
    
    avg_loss = total_loss / eval_len
    ppl = math.exp(avg_loss) if avg_loss < 20 else 99999
    ppl_results.append((pos, ppl))
    
    if ppl < 99999:
        print(f"  {pos:8,} | {ppl:9.1f} | {avg_loss:8.4f}", flush=True)
    else:
        print(f"  {pos:8,} | {'inf':>9} | {avg_loss:8.4f}", flush=True)

print(f"  " + "-" * 35, flush=True)

# === Phase 3: Speed Linearity ===
print(f"\n--- Phase 3: Speed Linearity ---", flush=True)
speed_checks = [64, 1024, 8192, 65536, 131072, 262144, 524288, 1000000]

print(f"  {'Context':>8} | {'Time(s)':>8} | {'tok/s':>8}", flush=True)
print(f"  " + "-" * 32, flush=True)

speed_results = []
for ctx_len in speed_checks:
    if ctx_len > TARGET: continue
    subtokens = tokens_t[:ctx_len]
    
    torch.cuda.synchronize()
    t0 = time.time()
    _ = chunked_forward(model, subtokens, chunk_size=4096)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    
    tps = ctx_len / elapsed if elapsed > 0 else 0
    speed_results.append((ctx_len, elapsed, tps))
    print(f"  {ctx_len:8,} | {elapsed:8.2f} | {tps:8.0f}", flush=True)

print(f"  " + "-" * 32, flush=True)

if len(speed_results) >= 2:
    speeds = [s[2] for s in speed_results]
    first_speed = speeds[0]
    last_speed = speeds[-1]
    ratio = last_speed / first_speed if first_speed > 0 else 0
    min_speed = min(speeds)
    max_speed = max(speeds)
    variation = (max_speed - min_speed) / ((max_speed + min_speed) / 2) * 100 if (max_speed + min_speed) > 0 else 0
    
    print(f"    速度范围: {min_speed:.0f} - {max_speed:.0f} tok/s (波动 {variation:.1f}%)", flush=True)
    print(f"    首尾速度比: {ratio:.2f}x", flush=True)
    
    if variation < 20 and 0.8 < ratio < 1.2:
        print(f"    => O(n) 线性度确认，无性能衰减", flush=True)

# === Phase 4: State Stability ===
print(f"\n--- Phase 4: State Stability Across Context Lengths ---", flush=True)
state_checks = [64, 1024, 8192, 65536, 262144, 524288, 999999]

print(f"  {'Position':>8} | " + " | ".join([f"S{s}_norm" for s in range(model.num_scales)]) + " |", flush=True)
print(f"  " + "-" * 70, flush=True)

for pos in state_checks:
    h = chunked_forward(model, tokens_t[:pos], chunk_size=4096)
    norms = [f"{h[s].norm(dim=-1).mean().item():.4f}" for s in range(model.num_scales)]
    print(f"  {pos:8,} | " + " | ".join(norms) + " |", flush=True)

# === Conclusion ===
print(f"\n{'='*70}", flush=True)
print(f"  FRSM 1M Context Test - RESULTS", flush=True)
print(f"{'='*70}", flush=True)
print(f"  Tokens processed: {TARGET:,}", flush=True)
print(f"  Total time: {elapsed:.1f}s ({tok_per_s:.0f} tok/s)", flush=True)
print(f"  Memory: ~4KB (fixed state, {model.d_model * model.num_scales * 4} bytes)", flush=True)

stable = all(
    not torch.isnan(h_final[s]).any() and not torch.isinf(h_final[s]).any()
    for s in range(model.num_scales)
)
print(f"  State stability: {'STABLE (no NaN/Inf)' if stable else 'UNSTABLE'}", flush=True)

if ppl_results:
    p_first = ppl_results[0][1]
    p_last = ppl_results[-1][1]
    delta = p_last - p_first
    print(f"  PPL({ppl_results[0][0]:,}) = {p_first:.1f}", flush=True)
    print(f"  PPL({ppl_results[-1][0]:,}) = {p_last:.1f}", flush=True)
    print(f"  PPL delta: {delta:+.1f}", flush=True)

print(f"  Speed linear: {'YES' if (len(speed_results) >= 2 and 0.8 < ratio < 1.2) else 'DEGRADED'}", flush=True)
print(f"{'='*70}", flush=True)
print(f"Done.", flush=True)
