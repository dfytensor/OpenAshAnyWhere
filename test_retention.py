"""FRSM 信息留存率分析 V2 — 修正采样 + 强化扰动"""
import os, sys, math, torch, json
import torch.nn.functional as F
import numpy as np

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, 'F:/OpenASH2605')
from config import agent_voc_path
from open_ash_voc import OpenASHVoc
from frsm.model import FractalRecursiveStateMachine

print("=== FRSM Information Retention Analysis V2 ===", flush=True)
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
print(f"Model: d_model={model.d_model}, scales={model.num_scales}", flush=True)

# 构建序列
TARGET = 32768
all_seqs = []
with open('minimind_data/pretrain_t2t_mini.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 8000: break
        try: text = json.loads(line).get('text', '')
        except: continue
        ids = voc.encode(text)
        if len(ids) >= 64: all_seqs.append(ids)

giant = []; si = 0
while len(giant) < TARGET:
    giant.extend(all_seqs[si % len(all_seqs)]); si += 1
giant = giant[:TARGET]
tokens_t = torch.tensor(giant, dtype=torch.long)
print(f"Built: {len(giant):,} tokens", flush=True)

# ============================================================
# 分析1: 状态自相关衰减 — 密集短距 + 稀疏长距 采样
# ============================================================
print(f"\n{'='*65}", flush=True)
print(f"  Analysis 1: State Autocorrelation Decay Curve", flush=True)
print(f"{'='*65}", flush=True)

# 密集采样短窗口 (0~1024), 稀疏采样长尾 (1024~32768)
short_positions = list(range(64, 1024, 64))
long_positions = list(range(1024, TARGET, 1024))
all_positions = short_positions + long_positions

# 精简版: 跳过已测试过的相近位置
test_distances = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 
                  1024, 2048, 4096, 8192, 16384, 32768]

# 在序列中采样固定数量的快照 (均匀间隔约 256 token)
snapshot_stride = 256
snapshot_positions = list(range(0, TARGET, snapshot_stride))
snapshot_states = []

print(f"  Collecting {len(snapshot_positions)} state snapshots (every {snapshot_stride} tokens)...", flush=True)

with torch.no_grad():
    h = [torch.zeros(1, model.d_model, device=device)
         for _ in range(model.num_scales)]
    current_pos = 0
    chunk_size = 2048
    
    for target_pos in snapshot_positions:
        while current_pos < target_pos:
            end = min(current_pos + chunk_size, target_pos)
            chunk = tokens_t[current_pos:end].unsqueeze(0).to(device)
            _, h, _ = model(chunk, h_prev=h, return_state=True, compute_critical_loss=False)
            current_pos = end
        snapshot_states.append((target_pos, [hs.clone().cpu() for hs in h]))

print(f"  Collected {len(snapshot_states)} states", flush=True)

# 计算每个距离的相似度
print(f"\n  {'Dist':>6} | {'S0(p=1)':>9} | {'S1(p=2)':>9} | {'S2(p=4)':>9} | {'S3(p=8)':>9} | {'Retention':>10}", flush=True)
print("  " + "-" * 65, flush=True)

retention_data = {}
for delta in test_distances:
    if delta >= len(giant): continue
    
    sims = {s: [] for s in range(model.num_scales)}
    step = delta // snapshot_stride
    
    for i in range(len(snapshot_states)):
        j = i + step
        if j >= len(snapshot_states): break
        if abs(snapshot_states[j][0] - snapshot_states[i][0] - delta) > snapshot_stride * 1.5:
            continue
        
        for s in range(model.num_scales):
            v1 = snapshot_states[i][1][s].squeeze(0)
            v2 = snapshot_states[j][1][s].squeeze(0)
            cos_sim = F.cosine_similarity(v1, v2, dim=0).item()
            sims[s].append(cos_sim)
    
    valid_delta = delta >= snapshot_stride  # 小于采样间隔的距离无意义
    if valid_delta:
        retention_data[delta] = {s: (sum(sims[s])/len(sims[s]) if sims[s] else 0) for s in range(model.num_scales)}
        avg_all = sum(retention_data[delta][s] for s in range(model.num_scales)) / model.num_scales
    else:
        # 这些距离太小，无有效对，标记占位
        retention_data[delta] = {s: -1 for s in range(model.num_scales)}
        avg_all = -1
    
    row = f"  {delta:6d}"
    for s in range(model.num_scales):
        v = retention_data[delta][s]
        if v >= 0:
            row += f" | {v:9.4f}"
        else:
            row += f" | {'<stride':>9}"
    row += f" | {avg_all:10.4f}" if avg_all >= 0 else f" | {'<stride':>10}"
    print(row, flush=True)

print("  " + "-" * 65, flush=True)

# 关键指标 (仅用有效距离)
print(f"\n  Key metrics (distance >= {snapshot_stride}, the min resolvable gap):", flush=True)
for s in range(model.num_scales):
    period = 2**s
    vals = [(d, retention_data[d][s]) for d in test_distances if d in retention_data and retention_data[d][s] > 0]
    if len(vals) >= 2:
        near_sim = vals[0][1]
        far_sim = vals[-1][1]
        half_idx = None
        for d, sim in vals:
            if sim < near_sim * 0.5 and half_idx is None:
                half_idx = d
        hl = f"{half_idx:,}" if half_idx else f">{vals[-1][0]:,}+"
        print(f"    Scale {s} (p={period}): near({vals[0][0]:,})={near_sim:.4f}, far({vals[-1][0]:,})={far_sim:.4f}, "
              f"half-life={hl} tokens", flush=True)

# ============================================================
# 分析2: 强化扰动 — 注入大噪声测试抗干扰性
# ============================================================
print(f"\n{'='*65}", flush=True)
print(f"  Analysis 2: Strong Perturbation Resilience (noise σ=0.5)", flush=True)
print(f"{'='*65}", flush=True)

ctx_len = 2048
ctx_t = tokens_t[:ctx_len].unsqueeze(0).to(device)

# 基准
with torch.no_grad():
    _, h_base, _ = model(ctx_t, return_state=True, compute_critical_loss=False)

perturb_positions = [32, 128, 512, 1024]
noise_level = 0.5  # 大噪声

print(f"  {'Inject@':>9} | {'Recover@':>9} | " + 
      " | ".join([f"S{s}_diff" for s in range(model.num_scales)]), flush=True)
print("  " + "-" * 65, flush=True)

for perturb_pos in perturb_positions:
    # 重新编码 + 扰动
    h = [torch.zeros(1, model.d_model, device=device) for _ in range(model.num_scales)]
    with torch.no_grad():
        _, h, _ = model(ctx_t[:, :perturb_pos], h_prev=h, return_state=True, compute_critical_loss=False)
        
        # 记录扰动前状态
        h_before = [hs.clone() for hs in h]
        
        # 注入强噪声
        h_perturbed = [hs.clone() for hs in h]
        for s in range(model.num_scales):
            h_perturbed[s] = h_perturbed[s] + torch.randn_like(h_perturbed[s]) * noise_level
        
        # 继续编码，观察是否能恢复
        remainder = ctx_t[:, perturb_pos:]
        _, h_recovered, _ = model(remainder, h_prev=h_perturbed, return_state=True, compute_critical_loss=False)
    
    # 差异: 扰动后终态 vs 基准终态
    diffs = []
    for s in range(model.num_scales):
        d = (h_recovered[s] - h_base[s]).norm(dim=-1).mean().item()
        diffs.append(d)
    
    remaining_tokens = ctx_len - perturb_pos
    row = f"  {perturb_pos:9d} | {remaining_tokens:9d}"
    for d in diffs:
        row += f" | {d:8.4f}"
    print(row, flush=True)

# 扰动前 vs 基准 (对照组)
print(f"\n  Baseline (noise=0, for reference):", flush=True)
h = [torch.zeros(1, model.d_model, device=device) for _ in range(model.num_scales)]
with torch.no_grad():
    _, h_clean, _ = model(ctx_t, h_prev=h, return_state=True, compute_critical_loss=False)
base_diffs = [(h_clean[s] - h_base[s]).norm(dim=-1).mean().item() for s in range(model.num_scales)]
print(f"  Clean rerun diff: " + " | ".join([f"S{s}={d:.6f}" for s, d in enumerate(base_diffs)]), flush=True)

# ============================================================
# 分析3: 遗忘曲线 — 输入 token 在状态中的留存
# ============================================================
print(f"\n{'='*65}", flush=True)
print(f"  Analysis 3: Input Token Memory Decay", flush=True)
print(f"{'='*65}", flush=True)

# 测量: 在第 0 步插入一个 token，在后续步骤测量状态对该 token 的敏感度
# 通过对比"有该token"和"无该token"的状态差异来量化留存

ctx_len = 128
probe_pos = 0  # 探测位置
test_deltas = [1, 2, 4, 8, 16, 32, 64]

ctx_t = tokens_t[:ctx_len].unsqueeze(0).to(device)
# 替换探测位置的 token 为另一个
alt_token = (tokens_t[probe_pos].item() + 100) % vs
ctx_alt = ctx_t.clone()
ctx_alt[0, probe_pos] = alt_token

with torch.no_grad():
    _, h_orig, _ = model(ctx_t, return_state=True, compute_critical_loss=False)
    _, h_alt, _ = model(ctx_alt, return_state=True, compute_critical_loss=False)

print(f"  Single-token perturbation at pos {probe_pos}:", flush=True)
print(f"  {'Delta':>6} | " + " | ".join([f"S{s}_diff" for s in range(model.num_scales)]), flush=True)
print("  " + "-" * 55, flush=True)

for delta in test_deltas:
    if delta >= ctx_len: continue
    # 只编码到 probe_pos + delta
    with torch.no_grad():
        _, h_o, _ = model(ctx_t[:, :probe_pos + delta], return_state=True, compute_critical_loss=False)
        _, h_a, _ = model(ctx_alt[:, :probe_pos + delta], return_state=True, compute_critical_loss=False)
    
    diffs = [(h_o[s] - h_a[s]).norm(dim=-1).mean().item() for s in range(model.num_scales)]
    row = f"  {delta:6d}"
    for d in diffs:
        row += f" | {d:8.6f}"
    print(row, flush=True)

print(f"\nDone.", flush=True)
