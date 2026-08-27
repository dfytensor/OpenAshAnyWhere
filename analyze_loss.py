"""快速 loss 分析 - 只测 3 个关键 checkpoint"""
import os, sys, math, torch
import torch.nn.functional as F
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, 'F:/OpenASH2605')
from config import agent_voc_path
from open_ash_voc import OpenASHVoc
from frsm.model import FractalRecursiveStateMachine

device = torch.device("cuda")
voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1

# 100 条评估集
eval_seqs = []
with open('minimind_data/pretrain_t2t_mini.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 100: break
        try: text = __import__('json').loads(line).get('text', '')
        except: continue
        ids = voc.encode(text)
        if len(ids) >= 20: eval_seqs.append(ids[:384])

@torch.no_grad()
def eval_loss(model):
    tl=0; tt=0
    for seq in eval_seqs:
        ids = torch.tensor([seq], dtype=torch.long, device=device)
        logits = model(ids[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, vs), ids[:, 1:].reshape(-1), ignore_index=0, reduction='sum')
        tl += loss.item(); tt += len(seq)-1
    return tl/tt

points = [
    (18000, "frsm_checkpoints/frsm_pretrain_step18000.pt"),
    (19000, "frsm_checkpoints/frsm_pretrain_step19000.pt"),
    (20000, "frsm_checkpoints/frsm_pretrain_final.pt"),
    (None,   "frsm_checkpoints/frsm_sft_final.pt"),
]

print(f"{'Step':>8} | {'Loss':>8} | {'PPL':>8} | {'ΔLoss':>8} | {'Δ/500步':>10}")
print("-" * 55)
prev = None
for step, path in points:
    ckpt = torch.load(path, map_location='cpu')
    m = FractalRecursiveStateMachine(vocab_size=vs, d_model=256, num_scales=4)
    m.load_state_dict(ckpt['model_state_dict'], strict=False)
    m = m.to(device).eval()
    loss = eval_loss(m)
    ppl = math.exp(loss)
    delta = f"{loss-prev:+.4f}" if prev else "—"
    rate = f"{(loss-prev)/500:+.6f}" if prev else "—"
    label = f"PT{step}" if step else "SFT_final"
    print(f"{label:>8} | {loss:8.4f} | {ppl:8.2f} | {delta:>8} | {rate:>10}")
    prev = loss
    del m; torch.cuda.empty_cache()

print(f"\n随机基线: loss={math.log(vs):.2f} PPL={vs}")
print(f"\n分析:")
print(f"  PT18000→20000 每步改善: ~{abs(eval_loss.__doc__ or 0):.6f}")
