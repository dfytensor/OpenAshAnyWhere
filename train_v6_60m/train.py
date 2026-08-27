"""
FRSM V6a Fast @ 60M — 全量 MiniMind 预训练 + SFT
参考 train_60m 结构，适配 frsm_v6a_fast.py 模型
"""
import torch, time, sys, math, os, json, gc, tempfile
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, 'F:/OpenASH2605')
os.chdir('F:/OpenASH2605')
from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from frsm_v6a_fast import FRSM_V6_Fast

# ============================================================
# Config
# ============================================================
D_MODEL = 1024       # ~56M params
NUM_SCALES = 4
PRETRAIN_SEQ = 384
SFT_SEQ = 768
BATCH_SIZE = 64    # d_model=1024, logits B×384×23005≈B×35MB, 128→4.5GB可过
GRAD_ACCUM = 1      # 有效batch=128
LR = 3e-4            # 预训练lr
SFT_LR = 3e-5        # SFT lr
WEIGHT_DECAY = 0.01
PRETRAIN_EPOCHS = 3
SFT_EPOCHS = 2
SAVE_EVERY = 500
LOG_EVERY = 20

DATA_DIR = './minimind_data'
OUT_DIR = './train_v6_60m'
CACHE_DIR = './train_v6_60m/cache'
PRETRAIN_FILE = 'pretrain_t2t_mini.jsonl'
SFT_FILE = 'sft_t2t_mini.jsonl'

# ============================================================
# Dataset
# ============================================================
class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, path, voc, max_len=384, cache_path=None):
        self.voc = voc; self.max_len = max_len
        # Cache 加速重复加载
        if cache_path and os.path.exists(cache_path):
            self.data = torch.load(cache_path)
            print(f'Loaded cached data: {len(self.data)} samples')
            return
        self.data = []
        with open(os.path.join(DATA_DIR, path), encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                text = json.loads(line).get('text', '')
                ids = voc.encode(text)
                if len(ids) >= 4:
                    self.data.append(torch.tensor(ids, dtype=torch.long))
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(self.data, cache_path)
            print(f'Cached {len(self.data)} samples to {cache_path}')
        print(f'Pretrain: {len(self.data)} samples')

    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        ids = self.data[i]
        if len(ids) > self.max_len + 1: ids = ids[:self.max_len + 1]
        return ids

    @staticmethod
    def collate(items):
        from torch.nn.utils.rnn import pad_sequence
        p = pad_sequence(items, batch_first=True, padding_value=0)
        return p[:, :-1], p[:, 1:]

class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, path, voc, max_len=768):
        self.voc = voc; self.max_len = max_len; self.data = []
        is_ = voc.token_to_id.get('<|im_start|>')
        ie_ = voc.token_to_id.get('<|im_end|>')
        uid_ = voc.token_to_id.get('<|user|>')
        aid_ = voc.token_to_id.get('<|agent|>')
        with open(os.path.join(DATA_DIR, path), encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                convs = json.loads(line).get('conversations', [])
                m = []
                for msg in convs:
                    role = msg.get('role', ''); ct = msg.get('content', '')
                    if role == 'user':
                        m += [is_, uid_] + voc.encode(ct) + [ie_]
                    elif role == 'assistant':
                        m += [is_, aid_]
                        if msg.get('reasoning_content'):
                            ts = voc.token_to_id.get('<|think|>')
                            te = voc.token_to_id.get('<|end_think|>')
                            m += [ts] + voc.encode(msg['reasoning_content']) + [te]
                        m += voc.encode(ct) + [ie_]
                if len(m) >= 4:
                    if len(m) > self.max_len + 1: m = m[:self.max_len + 1]
                    self.data.append(torch.tensor(m, dtype=torch.long))
        print(f'SFT: {len(self.data)} samples')

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

    @staticmethod
    def collate(items):
        from torch.nn.utils.rnn import pad_sequence
        p = pad_sequence(items, batch_first=True, padding_value=0)
        return p[:, :-1], p[:, 1:]

# ============================================================
# Training
# ============================================================
def get_lr_scheduler(optimizer, warmup, total):
    def lr_lambda(step):
        if step < warmup: return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def safe_save(obj, path):
    fd, tmp = tempfile.mkstemp(suffix='.tmp', dir=os.path.dirname(path))
    try: os.close(fd); torch.save(obj, tmp); os.replace(tmp, path)
    except: os.unlink(tmp); raise

def compute_loss(model, x, t, vs):
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, vs), t.reshape(-1), ignore_index=0)

def train_phase(model, dataset, epochs, seq_len, batch_size, lr, dev, vs, tag, ckpt_name):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=dataset.collate, drop_last=True)
    total_steps = len(loader) * epochs
    warmup = min(500, total_steps // 10)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY,
                            betas=(0.9, 0.95))
    sch = get_lr_scheduler(opt, warmup, total_steps)
    scaler = torch.amp.GradScaler('cuda')

    model.train()
    step = 0; loss_hist = []; best_loss = float('inf')
    accum_step = 0
    t_start = time.time()

    print(f'\n{"="*60}')
    print(f'  {tag}: {total_steps} steps, seq={seq_len}, bs={batch_size}×{GRAD_ACCUM}')
    print(f'  lr={lr:.1e}, warmup={warmup}, epochs={epochs}')
    print(f'{"="*60}')

    for epoch in range(epochs):
        for x, t in loader:
            x, t = x.to(dev, non_blocking=True), t.to(dev, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = compute_loss(model, x, t, vs)

            loss = loss / GRAD_ACCUM
            scaler.scale(loss).backward()
            accum_step += 1

            if accum_step % GRAD_ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sch.step()
                opt.zero_grad(set_to_none=True)

            step += 1
            loss_hist.append(loss.item() * GRAD_ACCUM)

            if loss.item() * GRAD_ACCUM < best_loss:
                best_loss = loss.item() * GRAD_ACCUM

            if step % LOG_EVERY == 0:
                avg = sum(loss_hist[-50:]) / min(50, len(loss_hist))
                elapsed = time.time() - t_start
                lr_now = opt.param_groups[0]['lr']
                tok_s = step * x.size(1) * batch_size / elapsed if elapsed > 0 else 0
                print(f'  step{step:7d}/{total_steps} | loss={avg:.4f} best={best_loss:.4f} | lr={lr_now:.2e} | {tok_s:.0f} tok/s | {elapsed:.0f}s', flush=True)

            if step % SAVE_EVERY == 0:
                ckpt = {'step': step, 'model_state_dict': model.state_dict(),
                        'opt': opt.state_dict(), 'sch': sch.state_dict(),
                        'config': (D_MODEL, NUM_SCALES)}
                safe_save(ckpt, os.path.join(OUT_DIR, f'{ckpt_name}_latest.pt'))
                safe_save(ckpt, os.path.join(OUT_DIR, f'{ckpt_name}_step{step}.pt'))
                print(f'  Saved checkpoint at step {step}', flush=True)

            if step >= total_steps: break
        if step >= total_steps: break

    final_ckpt = {'step': step, 'model_state_dict': model.state_dict(),
                  'config': (D_MODEL, NUM_SCALES)}
    safe_save(final_ckpt, os.path.join(OUT_DIR, f'{ckpt_name}_final.pt'))
    elapsed = time.time() - t_start
    avg_loss = sum(loss_hist[-100:]) / 100
    print(f'\n  {tag} done: {elapsed:.0f}s, avg_loss={avg_loss:.4f}', flush=True)
    return avg_loss, elapsed

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    dev = torch.device('cuda')
    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    vs = len(voc.token_to_id) + 1

    # Build model
    model = FRSM_V6_Fast(vocab_size=vs, d_model=D_MODEL, num_scales=NUM_SCALES)
    n_params = sum(p.numel() for p in model.parameters())
    model = model.to(dev)
    print(f'FRSM V6a Fast: {n_params:,} params')
    print(f'd_model={D_MODEL}, scales={NUM_SCALES}')
    print(f'batch={BATCH_SIZE}×{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}')

    # Pretrain
    pt_ds = PretrainDataset(PRETRAIN_FILE, voc, PRETRAIN_SEQ,
                            cache_path=os.path.join(CACHE_DIR, 'pretrain.pt'))
    pt_loss, pt_time = train_phase(model, pt_ds, PRETRAIN_EPOCHS, PRETRAIN_SEQ,
                                    BATCH_SIZE, LR, dev, vs,
                                    'PRETRAIN', 'v6_60m_pretrain')
    torch.cuda.empty_cache()

    # SFT
    sft_ds = SFTDataset(SFT_FILE, voc, SFT_SEQ)
    sft_loss, sft_time = train_phase(model, sft_ds, SFT_EPOCHS, SFT_SEQ,
                                      max(4, BATCH_SIZE // 4), SFT_LR, dev, vs,  # SFT用更小batch(seq=768)
                                      'SFT', 'v6_60m_sft')

    print(f'\n{"="*60}')
    print(f'  Training Complete')
    print(f'  Pretrain: {pt_time:.0f}s, loss={pt_loss:.4f}')
    print(f'  SFT:      {sft_time:.0f}s, loss={sft_loss:.4f}')
    print(f'  Params:   {n_params:,}')
    print(f'{"="*60}')