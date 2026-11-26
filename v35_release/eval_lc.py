"""
FRSM 长上下文微调后测试:
1. 消融实验 (完整 vs 截断128)
2. 大海捞针
3. PPL
"""
import os, sys, math, json, random, time
import torch, torch.nn.functional as F

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, 'F:/OpenASH2605')
from config import agent_voc_path
from open_ash_voc import OpenASHVoc
from frsm.model import FractalRecursiveStateMachine

device = torch.device("cuda")
voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1

def load_model(path):
    ckpt = torch.load(path, map_location='cpu')
    m = FractalRecursiveStateMachine(vocab_size=vs, d_model=256, num_scales=4)
    m.load_state_dict(ckpt['model_state_dict'], strict=False)
    return m.to(device).eval()

# 加载两个模型对比
sft_model = load_model("frsm_checkpoints/frsm_sft_final.pt")
lc_model = load_model("frsm_checkpoints/frsm_lc_step2500.pt")
print("Models loaded: SFT vs LC-finetuned", flush=True)

# 准备测试数据
novel_dir = r"F:\小说\女生小说"
novel_texts = []
for f in os.listdir(novel_dir):
    if f.endswith('.txt'):
        try:
            with open(os.path.join(novel_dir, f), 'r', encoding='utf-8', errors='ignore') as fp:
                t = fp.read(200000)
            if len(t) > 5000: novel_texts.append(t)
            if len(novel_texts) >= 10: break
        except: continue

# PPL 测试数据
all_seqs = []
with open('minimind_data/pretrain_t2t_mini.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 5000: break
        try: text = json.loads(line).get('text', '')
        except: continue
        ids = voc.encode(text)
        if len(ids) >= 128: all_seqs.append(ids)

is_ = voc.token_to_id.get('<|im_start|>')
ie_ = voc.token_to_id.get('<|im_end|>')
uid_ = voc.token_to_id.get('<|user|>')
aid_ = voc.token_to_id.get('<|agent|>')

# ============================================================
# 1. 消融实验
# ============================================================
print(f"\n{'='*70}", flush=True)
print(f"  1. Ablation: Full vs Truncated(128) — SFT vs LC", flush=True)
print(f"{'='*70}", flush=True)

@torch.no_grad()
def eval_ppl_at_ctx(model, ctx_ids, tgt_ids):
    """计算给定上下文下预测tgt的PPL"""
    ctx_t = torch.tensor([ctx_ids], dtype=torch.long, device=device)
    tgt_t = torch.tensor(tgt_ids, dtype=torch.long, device=device)
    logits, h, _ = model(ctx_t, return_state=True, compute_critical_loss=False)
    total_loss = 0.0
    for i in range(len(tgt_t)):
        if i == 0: pred = logits[:, -1, :]
        else: pred, h = model.generate_step(torch.tensor([[tgt_t[i-1].item()]], device=device), h)
        total_loss += F.cross_entropy(pred, tgt_t[i:i+1], reduction='sum').item()
    avg = total_loss / len(tgt_t)
    return math.exp(avg) if avg < 20 else 99999

trunc_len = 128
ctx_levels = [128, 256, 384, 512]

print(f"\n  {'Ctx':>5} | {'SFT Full':>9} | {'SFT Trunc':>9} | {'SFT Δ':>7} | {'LC Full':>9} | {'LC Trunc':>9} | {'LC Δ':>7}", flush=True)
print(f"  " + "-" * 72, flush=True)

for ctx_len in ctx_levels:
    sft_full_ppls = []; sft_trunc_ppls = []
    lc_full_ppls = []; lc_trunc_ppls = []
    
    for seq in all_seqs[:15]:
        if ctx_len + 64 > len(seq): continue
        ctx_full = seq[:ctx_len]
        tgt = seq[ctx_len:ctx_len+64]
        ctx_trunc = ctx_full[-trunc_len:]
        
        sft_full_ppls.append(eval_ppl_at_ctx(sft_model, ctx_full, tgt))
        sft_trunc_ppls.append(eval_ppl_at_ctx(sft_model, ctx_trunc, tgt))
        lc_full_ppls.append(eval_ppl_at_ctx(lc_model, ctx_full, tgt))
        lc_trunc_ppls.append(eval_ppl_at_ctx(lc_model, ctx_trunc, tgt))
    
    sf = sum(sft_full_ppls)/len(sft_full_ppls) if sft_full_ppls else 0
    st = sum(sft_trunc_ppls)/len(sft_trunc_ppls) if sft_trunc_ppls else 0
    lf = sum(lc_full_ppls)/len(lc_full_ppls) if lc_full_ppls else 0
    lt = sum(lc_trunc_ppls)/len(lc_trunc_ppls) if lc_trunc_ppls else 0
    
    print(f"  {ctx_len:5d} | {sf:9.1f} | {st:9.1f} | {st-sf:+7.1f} | {lf:9.1f} | {lt:9.1f} | {lt-lf:+7.1f}", flush=True)

# ============================================================
# 2. 大海捞针
# ============================================================
print(f"\n{'='*70}", flush=True)
print(f"  2. Needle in a Haystack — SFT vs LC", flush=True)
print(f"{'='*70}", flush=True)

NEEDLES = [
    ("我的手机密码是8473", "我的手机密码是什么", "8473"),
    ("钥匙藏在门口花盆下面第三个位置", "钥匙藏在哪里", "花盆"),
    ("会议室在三楼302房间", "会议室在哪个房间", "302"),
    ("密码箱的密码是9527", "密码箱的密码是多少", "9527"),
    ("小明家的猫叫橘子", "小明家的猫叫什么", "橘子"),
    ("李经理的工号是A20250314", "李经理的工号是什么", "A20250314"),
]

@torch.no_grad()
def needle_test(model, model_name, use_cd=False):
    scores = {512: [], 1024: [], 2048: []}
    
    for ctx_target in [512, 1024, 2048]:
        for trial in range(5):
            needle_stmt, question, answer = random.choice(NEEDLES)
            
            # 构建haystack
            novel = random.choice(novel_texts)
            novel_ids = voc.encode(novel)
            if len(novel_ids) < ctx_target + 100: continue
            
            needle_ids = voc.encode(needle_stmt)
            insert_pos = random.randint(50, ctx_target - len(needle_ids) - 50)
            
            ctx = novel_ids[:insert_pos] + needle_ids + novel_ids[insert_pos:ctx_target - len(needle_ids)]
            question_ids = voc.encode(question)
            
            prompt = [is_, uid_] + ctx + [ie_] + [is_, uid_] + question_ids + [ie_] + [is_, aid_]
            prompt = prompt[:768]  # 限制长度
            
            input_t = torch.tensor([prompt], dtype=torch.long, device=device)
            
            # 生成回答
            logits, h, _ = model(input_t, return_state=True, compute_critical_loss=False)
            generated = []
            for _ in range(30):
                last_logit = logits[:, -1, :] if not generated else logits
                v, _ = torch.topk(last_logit / 0.7, 40)
                masked = last_logit / 0.7
                masked = masked.masked_fill(masked < v[:, [-1]], float('-inf'))
                probs = F.softmax(masked, dim=-1)
                nt = torch.multinomial(probs, 1)
                nid = nt.item()
                if nid == ie_ or nid == 0: break
                generated.append(nid)
                logits, h = model.generate_step(nt, h)
            
            resp = voc.decode(generated)
            hit = 1 if answer in resp else 0
            scores[ctx_target].append(hit)
            
            mark = "HIT " if hit else "MISS"
            if trial < 2:
                print(f"    [{model_name}] ctx~{ctx_target} trial{trial+1} {mark} ans={answer} resp={resp[:40]}", flush=True)
    
    print(f"\n  [{model_name}] NIAH Results:", flush=True)
    for ctx in [512, 1024, 2048]:
        if scores[ctx]:
            acc = sum(scores[ctx]) / len(scores[ctx]) * 100
            print(f"    ctx~{ctx}: {acc:.0f}% ({sum(scores[ctx])}/{len(scores[ctx])})", flush=True)
    return scores

print(f"\n  Testing SFT model...", flush=True)
sft_scores = needle_test(sft_model, "SFT")

print(f"\n  Testing LC-finetuned model...", flush=True)
lc_scores = needle_test(lc_model, "LC")

# ============================================================
# 3. PPL 对比
# ============================================================
print(f"\n{'='*70}", flush=True)
print(f"  3. PPL Comparison", flush=True)
print(f"{'='*70}", flush=True)

@torch.no_grad()
def quick_ppl(model, seqs, max_batches=30):
    total_loss = 0; total_tokens = 0
    for seq in seqs[:max_batches]:
        if len(seq) < 10: continue
        ids = torch.tensor([seq], dtype=torch.long, device=device)
        logits = model(ids[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, vs), ids[:, 1:].reshape(-1), ignore_index=0, reduction='sum')
        total_loss += loss.item()
        total_tokens += (ids[:, 1:] != 0).sum().item()
    avg = total_loss / max(1, total_tokens)
    return avg, math.exp(avg) if avg < 20 else 99999

sft_loss, sft_ppl = quick_ppl(sft_model, all_seqs)
lc_loss, lc_ppl = quick_ppl(lc_model, all_seqs)
print(f"  SFT model:  PPL={sft_ppl:.2f} loss={sft_loss:.4f}", flush=True)
print(f"  LC  model:  PPL={lc_ppl:.2f} loss={lc_loss:.4f}", flush=True)

print(f"\n{'='*70}", flush=True)
print(f"  Done.", flush=True)
print(f"{'='*70}", flush=True)
