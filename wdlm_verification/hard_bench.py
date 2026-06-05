"""
Hard benchmark: WDLM-Real vs WDLM-Neural vs Transformer baseline
Equal params, equal data, equal compute budget.
"""
import torch, time, sys, math, os, json
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, 'F:/OpenASH2605')
sys.path.insert(0, 'F:/OpenASH2605/wdlm_verification')
os.chdir('F:/OpenASH2605')
from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from open_ash_v2 import OpenASH_V2
from wdlm_real import WaveDynamicsLM_Real
from wdlm_neural import WaveDynamicsLanguageModel as WDLM_Neural

voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1

# ============================================================
# 1. Pre-tokenized dataset
# ============================================================
tk = voc
is_ = tk.token_to_id.get('<|im_start|>'); ie = tk.token_to_id.get('<|im_end|>')
uid_ = tk.token_to_id.get('<|user|>'); aid_ = tk.token_to_id.get('<|agent|>')
ts_ = tk.token_to_id.get('<|think|>'); te_ = tk.token_to_id.get('<|end_think|>')

data = []; cnt = 0
with open('./data/science.jsonl', encoding='utf-8') as f:
    for line in f:
        if cnt >= 3000: break
        line = line.strip()
        if not line: continue
        c = json.loads(line)['conversations']
        m = []
        for msg in c:
            r = msg.get('from', msg.get('role', ''))
            ct = msg.get('value', msg.get('content', ''))
            if r in ('human', 'user'):
                m += [is_, uid_] + tk.encode(ct) + [ie]
            elif r in ('gpt', 'assistant', 'agent'):
                th = None; rp = ct
                if '<think>' in ct:
                    a = ct.find('<think>') + 7; b = ct.find('</think>')
                    if b != -1: th = ct[a:b].strip(); rp = ct[:a - 7] + ct[b + 8:]
                import re
                rm = re.search(r'<\s*response\s*>(.*?)$', rp, re.DOTALL | re.I)
                if rm: rp = rm.group(1).strip()
                m += [is_, aid_]
                if th: m += [ts_] + tk.encode(th) + [te_]
                if rp: m += tk.encode(rp); m += [ie]
        if len(m) > 256: m = m[:256]
        if len(m) >= 4:
            data.append(torch.tensor(m, dtype=torch.long))
        cnt += 1
print(f'Pre-tokenized: {len(data)} samples')
print(f'Models: Transformer, WDLM-Real(MH), OpenASH_V2, WDLM-Neural', flush=True)

class FastDS(torch.utils.data.Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]
    @staticmethod
    def cf(items):
        p = pad_sequence(items, batch_first=True, padding_value=0)
        return p[:, :-1], p[:, 1:]

# ============================================================
# 2. Transformer baseline (multi-head self-attention + FFN)
# ============================================================
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        S = x.size(1)
        mask = torch.triu(torch.ones(S, S, device=x.device) * float('-inf'), diagonal=1)
        a, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ffn(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, dim, heads, layers):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(512, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(layers)])
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        B, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.emb(x) + self.pos(pos)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


# ============================================================
# 3. Train
# ============================================================
def train(model, loader, steps, dev, max_seq=256):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler()
    crit = nn.CrossEntropyLoss(ignore_index=0)
    hist = []; it = iter(loader); t0 = time.time()
    for s in range(1, steps + 1):
        try: x, t = next(it)
        except: it = iter(loader); x, t = next(it)
        x = x[:, :max_seq].to(dev, non_blocking=True)
        t = t[:, :max_seq].to(dev, non_blocking=True)
        lr = 5e-5 * (0.1 + 0.45 * (1 + math.cos(math.pi * s / steps)))
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
        if s % 50 == 0 or s == 1:
            a = sum(hist[-min(50, s):]) / min(50, s)
            elapsed = time.time() - t0
            print(f'  s{s:4d} loss={loss.item():.4f} avg={a:.4f}  {s*4*max_seq/elapsed:.0f}tok/s', flush=True)
    return hist, (time.time() - t0)

# ============================================================
# 4. Run
# ============================================================
n_data = min(len(data), 2000)
ds = FastDS(data[:n_data])
ld = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=FastDS.cf, drop_last=True)
dev = torch.device('cuda:0')
H, L = 128, 2
STEPS = 300
max_seq = 256

results = {}

# Transformer baseline
torch.manual_seed(42)
m_tr = TransformerLM(vs, H, 4, L).to(dev)  # dim=128, heads=4
p_tr = sum(p.numel() for p in m_tr.parameters() if p.requires_grad)
print(f'\n=== Transformer: {p_tr:,}p ===', flush=True)
h_tr, t_tr = train(m_tr, ld, STEPS, dev, max_seq)
results['Transformer'] = (p_tr, sum(h_tr[-50:]) / 50, t_tr)
torch.cuda.empty_cache()

# WDLM-Real
torch.manual_seed(42)
m_real = WaveDynamicsLM_Real(vs, H, L, 1).to(dev)
p_real = sum(p.numel() for p in m_real.parameters() if p.requires_grad)
print(f'\n=== WDLM-Real: {p_real:,}p ===', flush=True)
h_real, t_real = train(m_real, ld, STEPS, dev, max_seq)
results['WDLM-Real'] = (p_real, sum(h_real[-50:]) / 50, t_real)
torch.cuda.empty_cache()

# OpenASH_V2
torch.manual_seed(42)
m_ash = OpenASH_V2(vs, H, 4, L).to(dev)
p_ash = sum(p.numel() for p in m_ash.parameters() if p.requires_grad)
print(f'\n=== OpenASH_V2: {p_ash:,}p ===', flush=True)
h_ash, t_ash = train(m_ash, ld, STEPS, dev, max_seq)
results['OpenASH_V2'] = (p_ash, sum(h_ash[-50:]) / 50, t_ash)
torch.cuda.empty_cache()

# WDLM-Neural
torch.manual_seed(42)
m_neural = WDLM_Neural(vs, H, L).to(dev)
p_neural = sum(p.numel() for p in m_neural.parameters() if p.requires_grad)
print(f'\n=== WDLM-Neural: {p_neural:,}p ===', flush=True)
h_neural, t_neural = train(m_neural, ld, STEPS, dev, max_seq)
results['WDLM-Neural'] = (p_neural, sum(h_neural[-50:]) / 50, t_neural)
torch.cuda.empty_cache()

# ============================================================
# 5. Report
# ============================================================
print(f'\n{"="*70}')
print(f'  Baseline Comparison (H={H}, L={L}, steps={STEPS}, seq={max_seq})')
print(f'{"="*70}')
print(f'{"Model":20s} {"Params":>10s} {"Loss":>8s} {"Tok/s":>8s} {"Tok/s/M":>10s}')
best = min(r[1] for r in results.values())
for name, (p, a, t) in results.items():
    tok = STEPS * 4 * max_seq / t
    marker = ' <--' if a == best else ''
    print(f'{name:20s} {p:>10,} {a:>8.4f} {tok:>8.0f} {tok/(p/1e6):>10.0f}{marker}')
