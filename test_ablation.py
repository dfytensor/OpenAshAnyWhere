"""
Ablation Test: FRSM 是否真的使用了远距离上下文？
对比: 完整上下文 vs 截断上下文(只看最近128) 的 PPL
"""
import os, sys, math, torch, json, time
import torch.nn.functional as F

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, 'F:/OpenASH2605')
from config import agent_voc_path
from open_ash_voc import OpenASHVoc
from frsm.model import FractalRecursiveStateMachine

device = torch.device("cuda")
voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1

ckpt = torch.load("frsm_checkpoints/frsm_pretrain_final.pt", map_location='cpu')
model = FractalRecursiveStateMachine(vocab_size=vs, d_model=256, num_scales=4)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model = model.to(device).eval()
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

# 收集长序列
all_seqs = []
with open('minimind_data/pretrain_t2t_mini.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 50000: break
        try: text = json.loads(line).get('text', '')
        except: continue
        ids = voc.encode(text)
        if len(ids) >= 512: all_seqs.append(ids)

# 截断距离
truncation_len = 128

# 测试: 不同上下文长度下，截断vs完整的 PPL 差距
context_levels = [128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
eval_len = 64  # 预测后续64个token

print(f"\n{'='*70}", flush=True)
print(f"  Ablation: Full Context vs Truncated (last {truncation_len})", flush=True)
print(f"  If model uses long-range info → Full PPL < Truncated PPL", flush=True)
print(f"{'='*70}", flush=True)

# 对每个上下文长度，在多条序列上取平均
all_results = {ctx: {'full_loss': [], 'trunc_loss': [], 'full_ppl': [], 'trunc_ppl': []}
               for ctx in context_levels}

for seq_idx, token_ids in enumerate(all_seqs[:20]):  # 20条序列
    seq_len = min(len(token_ids), 12288)  # 最大12K
    token_ids = token_ids[:seq_len]
    
    for ctx_len in context_levels:
        if ctx_len + eval_len > seq_len:
            continue
        
        # 完整上下文
        ctx_full = token_ids[:ctx_len]
        tgt = token_ids[ctx_len:ctx_len + eval_len]
        
        ctx_t = torch.tensor([ctx_full], dtype=torch.long, device=device)
        tgt_t = torch.tensor(tgt, dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits, h, _ = model(ctx_t, return_state=True, compute_critical_loss=False)
            total_loss = 0.0
            for i in range(len(tgt_t)):
                if i == 0:
                    pred = logits[:, -1, :]
                else:
                    prev = torch.tensor([[tgt_t[i-1].item()]], device=device)
                    pred, h = model.generate_step(prev, h)
                total_loss += F.cross_entropy(pred, tgt_t[i:i+1], reduction='sum').item()
        full_loss = total_loss / eval_len
        full_ppl = math.exp(full_loss) if full_loss < 20 else 99999
        
        # 截断上下文 (只看最近128)
        ctx_trunc = ctx_full[-truncation_len:]
        
        ctx_t2 = torch.tensor([ctx_trunc], dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits2, h2, _ = model(ctx_t2, return_state=True, compute_critical_loss=False)
            total_loss2 = 0.0
            for i in range(len(tgt_t)):
                if i == 0:
                    pred2 = logits2[:, -1, :]
                else:
                    prev2 = torch.tensor([[tgt_t[i-1].item()]], device=device)
                    pred2, h2 = model.generate_step(prev2, h2)
                total_loss2 += F.cross_entropy(pred2, tgt_t[i:i+1], reduction='sum').item()
        trunc_loss = total_loss2 / eval_len
        trunc_ppl = math.exp(trunc_loss) if trunc_loss < 20 else 99999
        
        all_results[ctx_len]['full_loss'].append(full_loss)
        all_results[ctx_len]['trunc_loss'].append(trunc_loss)
        all_results[ctx_len]['full_ppl'].append(full_ppl)
        all_results[ctx_len]['trunc_ppl'].append(trunc_ppl)

# 汇总
print(f"\n  {'Ctx':>6} | {'Full PPL':>9} | {'Trunc PPL':>9} | {'Δ PPL':>8} | {'Δ%':>7} | {'N':>4} | {'Verdict':>10}", flush=True)
print(f"  " + "-" * 68, flush=True)

for ctx_len in context_levels:
    r = all_results[ctx_len]
    n = len(r['full_ppl'])
    if n == 0: continue
    
    avg_full = sum(r['full_ppl']) / n
    avg_trunc = sum(r['trunc_ppl']) / n
    delta = avg_trunc - avg_full
    delta_pct = (delta / avg_full) * 100 if avg_full > 0 else 0
    
    # 判据
    if avg_full < avg_trunc - 5:
        verdict = "✓ USES LONG"
    elif abs(avg_full - avg_trunc) <= 5:
        verdict = "→ NO DIFF"
    else:
        verdict = "✗ REVERSED"
    
    print(f"  {ctx_len:6d} | {avg_full:9.1f} | {avg_trunc:9.1f} | {delta:+8.1f} | {delta_pct:+6.1f}% | {n:4d} | {verdict:>10}", flush=True)

print(f"  " + "-" * 68, flush=True)

# 关键趋势: 完整上下文优势是否随距离扩大
print(f"\n  Trend Analysis (Full Context Advantage = Trunc PPL - Full PPL):", flush=True)
deltas = []
for ctx_len in context_levels:
    r = all_results[ctx_len]
    if len(r['full_ppl']) > 0:
        avg_delta = sum(r['trunc_ppl']) / len(r['trunc_ppl']) - sum(r['full_ppl']) / len(r['full_ppl'])
        deltas.append((ctx_len, avg_delta))

if len(deltas) >= 3:
    # 线性趋势
    xs = [d[0] for d in deltas]
    ys = [d[1] for d in deltas]
    n_pts = len(xs)
    slope = (n_pts * sum(x*y for x,y in zip(xs,ys)) - sum(xs)*sum(ys)) / (n_pts * sum(x*x for x in xs) - sum(xs)**2)
    
    print(f"    Delta PPL slope: {slope:.4f}/token", flush=True)
    print(f"    At 128: {ys[0]:+.1f}  →  At {xs[-1]}: {ys[-1]:+.1f}", flush=True)
    
    if slope > 0.01:
        print(f"    => Model increasingly USES long-range info as context grows", flush=True)
        print(f"    => LONG-RANGE DEPENDENCY: CONFIRMED", flush=True)
    elif slope > -0.01:
        print(f"    => Context length has no effect on prediction quality", flush=True)
        print(f"    => Model does NOT use info beyond {truncation_len} tokens", flush=True)
    else:
        print(f"    => Unexpected: longer context HURTS prediction", flush=True)

# 对照组: 随机模型 (无训练的)
print(f"\n  Control: Random-Init Model (no training)", flush=True)
rand_model = FractalRecursiveStateMachine(vocab_size=vs, d_model=256, num_scales=4)
rand_model = rand_model.to(device).eval()

ctx_len = 1024
if min(len(t) for t in all_seqs) < ctx_len: ctx_len = 512

# 找一条序列
test_ids = all_seqs[0][:ctx_len+eval_len]
ctx_r = test_ids[:ctx_len]
tgt_r = test_ids[ctx_len:ctx_len+eval_len]

ctx_rt = torch.tensor([ctx_r], dtype=torch.long, device=device)
tgt_rt = torch.tensor(tgt_r, dtype=torch.long, device=device)

with torch.no_grad():
    logits_r, h_r, _ = rand_model(ctx_rt, return_state=True, compute_critical_loss=False)
    loss_r = 0.0
    for i in range(len(tgt_rt)):
        if i == 0: pred = logits_r[:, -1, :]
        else: pred, h_r = rand_model.generate_step(torch.tensor([[tgt_rt[i-1].item()]], device=device), h_r)
        loss_r += F.cross_entropy(pred, tgt_rt[i:i+1], reduction='sum').item()
rand_ppl = math.exp(loss_r / eval_len) if loss_r / eval_len < 20 else 99999

with torch.no_grad():
    # 训练的模型 full
    logits_f, h_f, _ = model(ctx_rt, return_state=True, compute_critical_loss=False)
    loss_f = 0.0
    for i in range(len(tgt_rt)):
        if i == 0: pred = logits_f[:, -1, :]
        else: pred, h_f = model.generate_step(torch.tensor([[tgt_rt[i-1].item()]], device=device), h_f)
        loss_f += F.cross_entropy(pred, tgt_rt[i:i+1], reduction='sum').item()
train_full_ppl = math.exp(loss_f / eval_len)

    # 训练的模型 truncated
ctx_t2 = torch.tensor([ctx_r[-truncation_len:]], dtype=torch.long, device=device)
logits_t2, h_t2, _ = model(ctx_t2, return_state=True, compute_critical_loss=False)
loss_t2 = 0.0
for i in range(len(tgt_rt)):
    if i == 0: pred2 = logits_t2[:, -1, :]
    else: pred2, h_t2 = model.generate_step(torch.tensor([[tgt_rt[i-1].item()]], device=device), h_t2)
    loss_t2 += F.cross_entropy(pred2, tgt_rt[i:i+1], reduction='sum').item()
train_trunc_ppl = math.exp(loss_t2 / eval_len)

print(f"    Random init (full context): PPL={rand_ppl:.1f}", flush=True)
print(f"    Trained (full context):     PPL={train_full_ppl:.1f}  Δ={rand_ppl-train_full_ppl:.0f}", flush=True)
print(f"    Trained (truncated):        PPL={train_trunc_ppl:.1f}  Δ={train_trunc_ppl-train_full_ppl:.0f}", flush=True)

print(f"\nDone.", flush=True)
