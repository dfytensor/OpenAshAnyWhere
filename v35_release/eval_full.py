"""
FRSM 14.7M 全量训练后完整评估
1. PPL (预训练+SFT)
2. 长期依赖 PPL (12K)
3. 消融实验: 完整 vs 截断128
4. 生成质量
5. 1M 状态稳定性
"""
import os, sys, math, json, time, random
import torch, torch.nn.functional as F

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, 'F:/OpenASH2605')
from config import agent_voc_path
from open_ash_voc import OpenASHVoc
from frsm.model import FractalRecursiveStateMachine
from frsm.dataset import create_dataloaders
from frsm.config import FRSMConfig

device = torch.device("cuda")
voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1
print(f"Vocab: {vs}", flush=True)

def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    m = FractalRecursiveStateMachine(
        vocab_size=vs, d_model=ckpt.get('config_d_model', 256),
        num_scales=ckpt.get('config_num_scales', 4),
    )
    m.load_state_dict(ckpt['model_state_dict'], strict=False)
    return m.to(device).eval()

@torch.no_grad()
def calc_ppl(model, loader, max_batches=50):
    model.eval()
    total_loss = 0.0; total_tokens = 0
    for i, (x, t) in enumerate(loader):
        if i >= max_batches: break
        x, t = x.to(device), t.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vs), t.reshape(-1), ignore_index=0, reduction='sum')
        total_loss += loss.item()
        total_tokens += (t != 0).sum().item()
    avg = total_loss / max(1, total_tokens)
    return avg, math.exp(avg) if avg < 20 else float('inf')

@torch.no_grad()
def generate(model, prompt_ids, max_new=100, temp=0.8, top_k=40):
    model.eval()
    ids = list(prompt_ids)
    h = None
    input_t = torch.tensor([ids], dtype=torch.long, device=device)
    logits_seq, h, _ = model(input_t, return_state=True, compute_critical_loss=False)
    logits = logits_seq[:, -1, :] / temp
    for _ in range(max_new):
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
        probs = F.softmax(logits, dim=-1)
        nt = torch.multinomial(probs, 1)
        nid = nt.item()
        im_end = voc.token_to_id.get('<|im_end|>')
        if nid == im_end or nid == 0: break
        ids.append(nid)
        logits, h = model.generate_step(nt, h)
        logits = logits / temp
    return ids

# ============================================================
print("=" * 70, flush=True)
print("  FRSM 14.7M Full Evaluation", flush=True)
print("=" * 70, flush=True)

# 加载模型
pt_model = load_model("frsm_checkpoints/frsm_pretrain_final.pt")
sft_model = load_model("frsm_checkpoints/frsm_sft_final.pt")
n = sum(p.numel() for p in pt_model.parameters())
print(f"  Params: {n:,}", flush=True)

# ============================================================
# 1. PPL
# ============================================================
print(f"\n--- 1. Perplexity ---", flush=True)
cfg = FRSMConfig(d_model=256, num_scales=4, max_seq_len=384, batch_size=8, max_pretrain_lines=5000)
pt_loader = create_dataloaders(voc, mode='pretrain', config=cfg)
pt_loss, pt_ppl = calc_ppl(pt_model, pt_loader, max_batches=50)
print(f"  Pretrain model on Pretrain data: PPL={pt_ppl:.2f} loss={pt_loss:.4f}", flush=True)

cfg2 = FRSMConfig(d_model=256, num_scales=4, max_seq_len=512, batch_size=8, max_sft_lines=5000)
sft_loader = create_dataloaders(voc, mode='sft', config=cfg2)
sft_loss, sft_ppl = calc_ppl(sft_model, sft_loader, max_batches=50)
print(f"  SFT model on SFT data:           PPL={sft_ppl:.2f} loss={sft_loss:.4f}", flush=True)

pt_on_sft_loss, pt_on_sft_ppl = calc_ppl(pt_model, sft_loader, max_batches=50)
print(f"  Pretrain model on SFT data:      PPL={pt_on_sft_ppl:.2f} loss={pt_on_sft_loss:.4f}", flush=True)

# ============================================================
# 2. 长期依赖 PPL (12K)
# ============================================================
print(f"\n--- 2. Long-Range PPL (12K sequence) ---", flush=True)
all_seqs = []
with open('minimind_data/pretrain_t2t_mini.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 50000: break
        try: text = json.loads(line).get('text', '')
        except: continue
        ids = voc.encode(text)
        if len(ids) >= 128: all_seqs.append(ids)

giant = []
for s in all_seqs:
    giant.extend(s)
    if len(giant) >= 12288: break
giant = giant[:12288]
print(f"  Sequence: {len(giant)} tokens", flush=True)

eval_len = 64
print(f"  {'Pos':>6} | {'PPL':>9} | {'Loss':>8}", flush=True)
print(f"  " + "-" * 35, flush=True)
ppl_results = []
ctx = 64
while ctx + eval_len <= len(giant):
    ctx_t = torch.tensor([giant[:ctx]], dtype=torch.long, device=device)
    tgt_t = torch.tensor(giant[ctx:ctx+eval_len], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, h, _ = sft_model(ctx_t, return_state=True, compute_critical_loss=False)
        total_loss = 0.0
        for i in range(len(tgt_t)):
            if i == 0: pred = logits[:, -1, :]
            else: pred, h = sft_model.generate_step(torch.tensor([[tgt_t[i-1].item()]], device=device), h)
            total_loss += F.cross_entropy(pred, tgt_t[i:i+1], reduction='sum').item()
    ppl = math.exp(total_loss / eval_len) if total_loss / eval_len < 20 else 99999
    ppl_results.append((ctx, ppl, total_loss / eval_len))
    print(f"  {ctx:6d} | {ppl:9.1f} | {total_loss/eval_len:8.4f}", flush=True)
    ctx += 512

if ppl_results:
    first_ppl = ppl_results[0][1]
    last_ppl = ppl_results[-1][1]
    q1 = sum(p for _,p,_ in ppl_results[:len(ppl_results)//4]) / (len(ppl_results)//4)
    q4 = sum(p for _,p,_ in ppl_results[-len(ppl_results)//4:]) / (len(ppl_results)//4)
    print(f"\n  PPL({ppl_results[0][0]})={first_ppl:.1f} → PPL({ppl_results[-1][0]})={last_ppl:.1f}", flush=True)
    print(f"  Q1 avg={q1:.1f}, Q4 avg={q4:.1f}, delta={q4-q1:+.1f}", flush=True)

# ============================================================
# 3. 消融实验: 完整 vs 截断128
# ============================================================
print(f"\n--- 3. Ablation: Full vs Truncated (last 128) ---", flush=True)
truncation_len = 128
context_levels = [128, 256, 384, 512, 768, 1024]
eval_len = 64

print(f"  {'Ctx':>6} | {'Full PPL':>9} | {'Trunc PPL':>9} | {'Delta':>8} | {'Verdict':>10}", flush=True)
print(f"  " + "-" * 55, flush=True)

for ctx_len in context_levels:
    r_full = []; r_trunc = []
    for seq in all_seqs[:20]:
        if ctx_len + eval_len > len(seq): continue
        ctx_full = seq[:ctx_len]
        tgt = seq[ctx_len:ctx_len+eval_len]
        
        # Full
        ctx_t = torch.tensor([ctx_full], dtype=torch.long, device=device)
        tgt_t = torch.tensor(tgt, dtype=torch.long, device=device)
        with torch.no_grad():
            logits, h, _ = sft_model(ctx_t, return_state=True, compute_critical_loss=False)
            loss = 0.0
            for i in range(len(tgt_t)):
                if i == 0: pred = logits[:, -1, :]
                else: pred, h = sft_model.generate_step(torch.tensor([[tgt_t[i-1].item()]], device=device), h)
                loss += F.cross_entropy(pred, tgt_t[i:i+1], reduction='sum').item()
        r_full.append(math.exp(loss/eval_len) if loss/eval_len < 20 else 99999)
        
        # Truncated
        ctx_trunc = ctx_full[-truncation_len:]
        ctx_t2 = torch.tensor([ctx_trunc], dtype=torch.long, device=device)
        with torch.no_grad():
            logits2, h2, _ = sft_model(ctx_t2, return_state=True, compute_critical_loss=False)
            loss2 = 0.0
            for i in range(len(tgt_t)):
                if i == 0: pred2 = logits2[:, -1, :]
                else: pred2, h2 = sft_model.generate_step(torch.tensor([[tgt_t[i-1].item()]], device=device), h2)
                loss2 += F.cross_entropy(pred2, tgt_t[i:i+1], reduction='sum').item()
        r_trunc.append(math.exp(loss2/eval_len) if loss2/eval_len < 20 else 99999)
    
    if r_full and r_trunc:
        avg_f = sum(r_full)/len(r_full)
        avg_t = sum(r_trunc)/len(r_trunc)
        delta = avg_t - avg_f
        verdict = "USES LONG" if delta > 5 else ("NO DIFF" if abs(delta) <= 5 else "REVERSED")
        print(f"  {ctx_len:6d} | {avg_f:9.1f} | {avg_t:9.1f} | {delta:+8.1f} | {verdict:>10}", flush=True)

# ============================================================
# 4. 生成质量
# ============================================================
print(f"\n--- 4. Generation Quality ---", flush=True)
prompts = [
    "写一首关于秋天的诗",
    "你好，请问你是谁？",
    "解释一下什么是人工智能",
    "给我讲一个故事",
]
for prompt in prompts:
    full = f"<|im_start|><|user|>{prompt}<|im_end|><|im_start|><|agent|>"
    ids = voc.encode(full)
    gen_ids = generate(sft_model, ids, max_new=80, temp=0.8)
    resp = voc.decode(gen_ids[len(ids):])
    safe = resp[:200]
    print(f"  Q: {prompt}", flush=True)
    print(f"  A: {safe}", flush=True)
    print(flush=True)

# ============================================================
# 5. 1M 状态稳定性
# ============================================================
print(f"--- 5. 1M State Stability ---", flush=True)
giant_1m = []
si = 0
while len(giant_1m) < 1000000:
    giant_1m.extend(all_seqs[si % len(all_seqs)])
    si += 1
giant_1m = giant_1m[:1000000]
tokens_t = torch.tensor(giant_1m, dtype=torch.long)

chunk_size = 4096
h = [torch.zeros(1, 256, device=device) for _ in range(4)]
torch.cuda.synchronize(); t0 = time.time()
for start in range(0, 1000000, chunk_size):
    end = min(start + chunk_size, 1000000)
    chunk = tokens_t[start:end].unsqueeze(0).to(device)
    with torch.no_grad():
        _, h, _ = sft_model(chunk, h_prev=h, return_state=True, compute_critical_loss=False)
torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"  1M tokens in {elapsed:.1f}s ({1000000/elapsed:.0f} tok/s)", flush=True)
for s in range(4):
    norm = h[s].norm(dim=-1).mean().item()
    has_nan = torch.isnan(h[s]).any().item()
    print(f"  Scale {s}: norm={norm:.4f} NaN={has_nan}", flush=True)

print(f"\n{'='*70}", flush=True)
print(f"  Evaluation Complete", flush=True)
print(f"{'='*70}", flush=True)
