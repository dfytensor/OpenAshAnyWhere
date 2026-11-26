import os, sys, math, torch
import torch.nn.functional as F

sys.path.insert(0, 'F:/OpenASH2605')
from config import agent_voc_path
from open_ash_voc import OpenASHVoc
from frsm.model import FractalRecursiveStateMachine
from frsm.dataset import create_dataloaders
from frsm.config import FRSMConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1
print(f"Vocabulary size: {vs}")

@torch.no_grad()
def eval_ppl(model, loader, vs, max_batches=20):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, (x, t) in enumerate(loader):
        if i >= max_batches: break
        x = x.to(device); t = t.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vs), t.reshape(-1), ignore_index=0, reduction='sum')
        total_loss += loss.item()
        total_tokens += (t != 0).sum().item()
    avg = total_loss / max(1, total_tokens)
    ppl = math.exp(avg) if avg < 20 else float('inf')
    return avg, ppl

@torch.no_grad()
def generate(model, voc, prompt, max_len=128, temp=0.8):
    model.eval()
    ids = voc.encode(prompt)
    if not ids: return ""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    h = None
    gen = list(ids)
    im_end = voc.token_to_id.get('<|im_end|>')
    for _ in range(max_len):
        if h is None:
            logits_seq, h, _ = model(x, return_state=True, compute_critical_loss=False)
            logits = logits_seq[:, -1, :]
        else:
            logits, h = model.generate_step(torch.tensor([[gen[-1]]], device=device), h)
        logits = logits / temp
        probs = F.softmax(logits, dim=-1)
        top_p, top_i = torch.topk(probs, min(50, probs.size(-1)), dim=-1)
        top_p = top_p / top_p.sum(dim=-1, keepdim=True)
        nt = torch.multinomial(top_p, 1)
        nid = top_i[0, nt[0,0]].item()
        if nid == im_end or nid == 0: break
        gen.append(nid)
    return voc.decode(gen[len(ids):])

ckpt = torch.load("frsm_checkpoints/frsm_sft_final.pt", map_location='cpu')
model = FractalRecursiveStateMachine(
    vocab_size=vs,
    d_model=ckpt.get('config_d_model', 256),
    num_scales=ckpt.get('config_num_scales', 4),
)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model = model.to(device)
n = sum(p.numel() for p in model.parameters())

print(f"\n{'='*60}")
print(f"  FRSM-SFT Model ({n:,} params)")
print(f"{'='*60}")

config = FRSMConfig(d_model=256, num_scales=4, max_seq_len=256, batch_size=4, max_pretrain_lines=2000)
loader = create_dataloaders(voc, mode='pretrain', config=config)
avg_loss, ppl = eval_ppl(model, loader, vs, max_batches=20)
print(f"  Pretrain PPL: {ppl:.2f} (loss={avg_loss:.4f})")

config2 = FRSMConfig(d_model=256, num_scales=4, max_seq_len=256, batch_size=4, max_sft_lines=1000)
loader2 = create_dataloaders(voc, mode='sft', config=config2)
avg_loss2, ppl2 = eval_ppl(model, loader2, vs, max_batches=10)
print(f"  SFT PPL: {ppl2:.2f} (loss={avg_loss2:.4f})")

# Count total samples for overall stats
pt_total = 0
with open('minimind_data/pretrain_t2t_mini.jsonl', 'r', encoding='utf-8') as f:
    for _ in f: pt_total += 1
sft_total = 0
with open('minimind_data/sft_t2t_mini.jsonl', 'r', encoding='utf-8') as f:
    for _ in f: sft_total += 1
print(f"\n  Dataset: {pt_total} pretrain lines, {sft_total} SFT lines")

print(f"\n  --- Generation Samples ---")
prompts = [
    ("SFT", "你好，请问你是谁？"),
    ("SFT", "写一首关于春天的诗"),
    ("SFT", "解释一下什么是人工智能"),
]
for mode, prompt in prompts:
    full = f"<|im_start|><|user|>{prompt}<|im_end|><|im_start|><|agent|>"
    resp = generate(model, voc, full, max_len=100, temp=0.8)
    safe_resp = resp.encode('latin-1', errors='replace').decode('latin-1', errors='replace')
    print(f"  Prompt: {prompt}")
    print(f"  Output: {resp[:300]}")
    print()
