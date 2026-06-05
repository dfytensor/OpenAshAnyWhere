"""
WDLM-Neural vs Transformer @ 20M params
Pretrain (seq=384) → SFT (seq=512) → Benchmark
"""
import torch, time, sys, math, os, json, argparse
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, 'F:/OpenASH2605')
sys.path.insert(0, 'F:/OpenASH2605/wdlm_verification')
os.chdir('F:/OpenASH2605')
from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from open_ash_v2 import OpenASH_V2
from open_ash import OpenASH
from wdlm_neural import WaveDynamicsLanguageModel as WN
from wdlm_real import WaveDynamicsLM_Real as WR

# ============================================================
# Config
# ============================================================
DATA_DIR = './minimind_data'
PRETRAIN_FILE = 'pretrain_t2t_mini.jsonl'
SFT_FILE = 'sft_t2t_mini.jsonl'
OUT_DIR = './train_20m'

# ============================================================
# Transformer
# ============================================================
class TBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.atn = nn.MultiheadAttention(dim, heads, batch_first=True, bias=False)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.n1 = nn.LayerNorm(dim); self.n2 = nn.LayerNorm(dim)

    def forward(self, x):
        S = x.size(1)
        m = torch.triu(torch.ones(S, S, device=x.device) * float('-inf'), 1)
        a, _ = self.atn(x, x, x, attn_mask=m)
        return self.n2(self.n1(x + a) + self.ffn(self.n1(x + a)))

class TransformerLM(nn.Module):
    def __init__(self, vs, dim, heads, layers):
        super().__init__()
        self.emb = nn.Embedding(vs, dim)
        self.layers = nn.ModuleList([TBlock(dim, heads) for _ in range(layers)])
        self.head = nn.Linear(dim, vs, bias=False)

    def forward(self, x):
        h = self.emb(x)
        for l in self.layers: h = l(h)
        return self.head(h)


class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, path, tok, max_len=384, max_lines=50000):
        self.tok = tok; self.max_len = max_len; self.data = []
        with open(path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines: break
                line = line.strip()
                if not line: continue
                text = json.loads(line).get('text', '')
                ids = tok.encode(text)
                if len(ids) >= 4:
                    self.data.append(torch.tensor(ids, dtype=torch.long))
        print(f'Pretrain: {len(self.data)} samples from {path} (max_lines={max_lines})')

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        ids = self.data[i]
        if len(ids) > self.max_len + 1:
            ids = ids[:self.max_len + 1]
        return ids

    @staticmethod
    def cf(items):
        p = pad_sequence(items, batch_first=True, padding_value=0)
        return p[:, :-1], p[:, 1:]


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, path, tok, max_len=512, max_lines=50000):
        self.tok = tok; self.max_len = max_len; self.data = []
        is_ = tok.token_to_id.get('<|im_start|>'); ie_ = tok.token_to_id.get('<|im_end|>')
        uid_ = tok.token_to_id.get('<|user|>'); aid_ = tok.token_to_id.get('<|agent|>')

        with open(path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines: break
                line = line.strip()
                if not line: continue
                convs = json.loads(line).get('conversations', [])
                m = []
                for msg in convs:
                    role = msg.get('role', '')
                    ct = msg.get('content', '')
                    if role == 'user':
                        m += [is_, uid_] + tok.encode(ct) + [ie_]
                    elif role == 'assistant':
                        m += [is_, aid_]
                        if msg.get('reasoning_content'):
                            ts = tok.token_to_id.get('<|think|>')
                            te = tok.token_to_id.get('<|end_think|>')
                            m += [ts] + tok.encode(msg['reasoning_content']) + [te]
                        m += tok.encode(ct) + [ie_]
                if len(m) >= 4:
                    if len(m) > self.max_len + 1:
                        m = m[:self.max_len + 1]
                    self.data.append(torch.tensor(m, dtype=torch.long))
        print(f'SFT: {len(self.data)} samples from {path}')

    def __len__(self): return len(self.data)

    def __getitem__(self, i): return self.data[i]

    @staticmethod
    def cf(items):
        p = pad_sequence(items, batch_first=True, padding_value=0)
        return p[:, :-1], p[:, 1:]


# ============================================================
# Training
# ============================================================
def train(model, loader, steps, dev, vs):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler()
    crit = nn.CrossEntropyLoss(ignore_index=0)
    hist, it, t0 = [], iter(loader), time.time()

    for s in range(1, steps + 1):
        try: x, t = next(it)
        except StopIteration: it = iter(loader); x, t = next(it)
        x = x.to(dev, non_blocking=True); t = t.to(dev, non_blocking=True)
        lr = 5e-4 * (0.1 + 0.45 * (1 + math.cos(math.pi * s / steps)))
        for pg in opt.param_groups: pg['lr'] = lr
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)
            if isinstance(out, tuple): o = out[0]
            else: o = out
            loss = crit(o.reshape(-1, vs), t.reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        hist.append(float(loss.detach()))
        if s % 100 == 0 or s == 1:
            a = sum(hist[-min(50, s):]) / min(50, s)
            elapsed = time.time() - t0
            print(f'  s{s:5d} loss={loss.item():.4f} avg50={a:.4f}  {s*x.size(1)/elapsed:.0f} tok/s', flush=True)
    return hist, time.time() - t0, sum(hist[-100:]) / 100


def count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run_phase(model, ds_class, data_path, max_len, steps, dev, vs, tag, max_lines=50000):
    ds = ds_class(os.path.join(DATA_DIR, data_path), voc, max_len, max_lines)
    n = min(len(ds), steps * 4 * 2)
    idx = torch.randperm(len(ds))[:n].tolist()
    sub = torch.utils.data.Subset(ds, idx)
    ld = DataLoader(sub, batch_size=4, shuffle=True, num_workers=0, collate_fn=ds_class.cf, drop_last=True)
    print(f'  Training subset: {n} samples, batches: {len(ld)}', flush=True)
    print(f'\n=== {tag} (seq={max_len}, steps={steps}) ===', flush=True)
    hist, elapsed, avg_loss = train(model, ld, steps, dev, vs)
    return avg_loss, elapsed, hist


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_steps', type=int, default=1000)
    parser.add_argument('--sft_steps', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    vs = len(voc.token_to_id) + 1
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}
    models = [
        ('WDLM-Neural+gen', lambda: WN(vs, hidden_dim=256, num_layers=9).to(dev)),
        ('WDLM-Real',       lambda: WR(vs, hidden_dim=256, num_layers=12, evo_steps=1).to(dev)),
        ('OpenASH_V2',      lambda: OpenASH_V2(vs, hidden_size=288, num_heads=9, num_layers=12).to(dev)),
        ('OpenASH (ReLU)',  lambda: OpenASH(vs, hidden_size=288, num_heads=9, num_layers=12).to(dev)),
        ('Transformer',     lambda: TransformerLM(vs, 256, 8, 10).to(dev)),
    ]

    for name, factory in models:
        torch.manual_seed(42)
        model = factory()
        p = count(model)
        print(f'\n{"="*60}')
        print(f'{name}: {p:,} params (target ~20M)')
        print(f'{"="*60}')

        # Pretrain
        pt_loss, pt_time, pt_hist = run_phase(
            model, PretrainDataset, PRETRAIN_FILE, 384, args.pretrain_steps, dev, vs, 'Pretrain'
        )
        torch.save(model.state_dict(), f'{OUT_DIR}/{name}_pretrain.pth')
        torch.cuda.empty_cache()

        # SFT
        sft_loss, sft_time, sft_hist = run_phase(
            model, SFTDataset, SFT_FILE, 512, args.sft_steps, dev, vs, 'SFT'
        )

        results[name] = {
            'params': p,
            'pretrain_loss': pt_loss,
            'pretrain_time': pt_time,
            'sft_loss': sft_loss,
            'sft_time': sft_time,
        }
        torch.cuda.empty_cache()

    # Report
    print(f'\n{"="*70}')
    print(f'  Final Comparison')
    print(f'{"="*70}')
    print(f'{"Model":>15s} {"Params":>10s} {"PT Loss":>10s} {"SFT Loss":>10s} {"PT tok/s":>10s} {"SFT tok/s":>10s}')
    for name, r in results.items():
        pt_tok = args.pretrain_steps * 4 * 384 / r['pretrain_time']
        sft_tok = args.sft_steps * 4 * 512 / r['sft_time']
        print(f'{name:>15s} {r["params"]:>10,} {r["pretrain_loss"]:>10.4f} {r["sft_loss"]:>10.4f} {pt_tok:>10.0f} {sft_tok:>10.0f}')
