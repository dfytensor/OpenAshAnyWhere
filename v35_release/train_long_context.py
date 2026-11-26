"""
FRSM 长上下文微调: 在 SFT 模型上用长上下文检索数据继续训练
目标: 让模型学会使用远距离上下文
"""
import os, sys, time, math, json, torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, 'F:/OpenASH2605')
from config import agent_voc_path
from open_ash_voc import OpenASHVoc
from frsm.model import FractalRecursiveStateMachine

device = torch.device("cuda")
voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1

# 加载已训练的 SFT 模型
ckpt = torch.load("frsm_checkpoints/frsm_sft_final.pt", map_location='cpu')
model = FractalRecursiveStateMachine(
    vocab_size=vs,
    d_model=ckpt.get('config_d_model', 256),
    num_scales=ckpt.get('config_num_scales', 4),
)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model = model.to(device)
model.critical_reg_coeff = 0.01
print(f"Loaded SFT model: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

# 长上下文数据集
class LongContextDataset(Dataset):
    def __init__(self, path, voc, max_len=768):
        self.voc = voc; self.max_len = max_len
        self.data = []
        is_ = voc.token_to_id.get('<|im_start|>')
        ie_ = voc.token_to_id.get('<|im_end|>')
        uid_ = voc.token_to_id.get('<|user|>')
        aid_ = voc.token_to_id.get('<|agent|>')
        
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                convs = json.loads(line).get('conversations', [])
                m = []
                for msg in convs:
                    role = msg.get('role', '')
                    ct = msg.get('content', '')
                    if role == 'user':
                        m += [is_, uid_] + voc.encode(ct) + [ie_]
                    elif role == 'assistant':
                        m += [is_, aid_] + voc.encode(ct) + [ie_]
                if len(m) >= 16:
                    if len(m) > max_len + 1:
                        m = m[:max_len + 1]
                    self.data.append(torch.tensor(m, dtype=torch.long))
        print(f'Long-context: {len(self.data)} samples, max_len={max_len}')

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

    @staticmethod
    def collate_fn(items):
        padded = pad_sequence(items, batch_first=True, padding_value=0)
        return padded[:, :-1], padded[:, 1:]

# 数据
dataset = LongContextDataset("minimind_data/long_context_sft.jsonl", voc, max_len=768)
loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=LongContextDataset.collate_fn, drop_last=True)

# 训练
optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01, betas=(0.9, 0.95))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5000)

STEPS = 5000
LOG_EVERY = 100
SAVE_EVERY = 1000
OUT_DIR = "frsm_checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)

model.train()
global_step = 0
loss_accum = 0
start_time = time.time()
data_iter = iter(loader)

print(f"\nStarting long-context fine-tuning ({STEPS} steps, bs=16, lr=3e-5)...", flush=True)
print("-" * 60, flush=True)

while global_step < STEPS:
    try:
        x, t = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        x, t = next(data_iter)
    
    x, t = x.to(device), t.to(device)
    
    logits, _, crit_loss = model(x, return_state=True, compute_critical_loss=True)
    lm_loss = F.cross_entropy(logits.reshape(-1, vs), t.reshape(-1), ignore_index=0)
    total_loss = lm_loss + model.critical_reg_coeff * crit_loss
    
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
    global_step += 1
    loss_accum += lm_loss.item()
    
    if global_step % LOG_EVERY == 0:
        avg = loss_accum / LOG_EVERY
        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']
        print(f"  step {global_step:5d}/{STEPS} | lm_loss={avg:.4f} | crit={crit_loss.item():.4f} | lr={lr:.2e} | {elapsed:.0f}s", flush=True)
        loss_accum = 0
    
    if global_step % SAVE_EVERY == 0:
        save_path = os.path.join(OUT_DIR, f"frsm_lc_step{global_step}.pt")
        torch.save({
            'step': global_step,
            'model_state_dict': model.state_dict(),
            'config_d_model': 256,
            'config_num_scales': 4,
        }, save_path)
        print(f"  Saved: {save_path}", flush=True)

# 最终保存
final_path = os.path.join(OUT_DIR, "frsm_lc_final.pt")
torch.save({
    'step': global_step,
    'model_state_dict': model.state_dict(),
    'config_d_model': 256,
    'config_num_scales': 4,
}, final_path)
print(f"\nDone! Saved to {final_path} ({time.time()-start_time:.0f}s)", flush=True)
