"""继续长上下文微调 - 从step1000断点恢复"""
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

ckpt = torch.load("frsm_checkpoints/frsm_lc_step1500.pt", map_location='cpu')
model = FractalRecursiveStateMachine(vocab_size=vs, d_model=256, num_scales=4)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model = model.to(device); model.critical_reg_coeff = 0.01

class LongContextDataset(Dataset):
    def __init__(self, path, voc, max_len=768):
        self.data = []
        is_ = voc.token_to_id.get('<|im_start|>')
        ie_ = voc.token_to_id.get('<|im_end|>')
        uid_ = voc.token_to_id.get('<|user|>')
        aid_ = voc.token_to_id.get('<|agent|>')
        with open(path, encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                convs = json.loads(line).get('conversations', [])
                m = []
                for msg in convs:
                    role = msg.get('role',''); ct = msg.get('content','')
                    if role == 'user': m += [is_, uid_] + voc.encode(ct) + [ie_]
                    elif role == 'assistant': m += [is_, aid_] + voc.encode(ct) + [ie_]
                if len(m) >= 16:
                    if len(m) > max_len+1: m = m[:max_len+1]
                    self.data.append(torch.tensor(m, dtype=torch.long))
        print(f'Dataset: {len(self.data)} samples')
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]
    @staticmethod
    def cf(items):
        p = pad_sequence(items, batch_first=True, padding_value=0)
        return p[:, :-1], p[:, 1:]

dataset = LongContextDataset("minimind_data/long_context_sft.jsonl", voc, max_len=768)
loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=LongContextDataset.cf, drop_last=True)

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01, betas=(0.9, 0.95))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2000)

model.train()
step = ckpt['step']
loss_accum = 0; t0 = time.time()
data_iter = iter(loader)

print(f"Resuming from step {step}, training 3000 more steps, bs=32...", flush=True)

for i in range(2000):
    try: x, t = next(data_iter)
    except StopIteration: data_iter = iter(loader); x, t = next(data_iter)
    
    x, t = x.to(device), t.to(device)
    logits, _, crit = model(x, return_state=True, compute_critical_loss=True)
    loss = F.cross_entropy(logits.reshape(-1, vs), t.reshape(-1), ignore_index=0) + 0.01 * crit
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step(); scheduler.step()
    
    step += 1; loss_accum += loss.item()
    
    if step % 100 == 0:
        print(f"  step {step} loss={loss_accum/100:.4f} lr={optimizer.param_groups[0]['lr']:.2e} {time.time()-t0:.0f}s", flush=True)
        loss_accum = 0
    
    if step % 500 == 0:
        torch.save({'step': step, 'model_state_dict': model.state_dict(), 'config_d_model': 256, 'config_num_scales': 4},
                   f"frsm_checkpoints/frsm_lc_step{step}.pt")
        print(f"  Saved step {step}", flush=True)

torch.save({'step': step, 'model_state_dict': model.state_dict(), 'config_d_model': 256, 'config_num_scales': 4},
           "frsm_checkpoints/frsm_lc_final.pt")
print(f"Done. step={step}", flush=True)
