"""Long convergence test: WDLM-Neural vs Transformer, 2000 steps"""
import torch, time, sys, math, os, json
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, 'F:/OpenASH2605')
sys.path.insert(0, 'F:/OpenASH2605/wdlm_verification')
os.chdir('F:/OpenASH2605')
from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from wdlm_neural import WaveDynamicsLanguageModel as WN

voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1

# Pre-tokenize
tk = voc
is_ = tk.token_to_id.get('<|im_start|>'); ie = tk.token_to_id.get('<|im_end|>')
uid_ = tk.token_to_id.get('<|user|>'); aid_ = tk.token_to_id.get('<|agent|>')
ts_ = tk.token_to_id.get('<|think|>'); te_ = tk.token_to_id.get('<|end_think|>')

data = []; cnt = 0
with open('./data/science.jsonl', encoding='utf-8') as f:
    for line in f:
        if cnt >= 4000: break
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
print(f'Pre-tokenized: {len(data)} samples', flush=True)

class DS(torch.utils.data.Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]
    @staticmethod
    def cf(items):
        p = pad_sequence(items, batch_first=True, padding_value=0)
        return p[:, :-1], p[:, 1:]

n = min(len(data), 4000)
ds = DS(data[:n])
ld = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=DS.cf, drop_last=True)
dev = torch.device('cuda:0')
STEPS = 2000
H, L = 128, 2

# Transformer baseline
class TBlk(nn.Module):
    def __init__(self):
        super().__init__()
        self.atn = nn.MultiheadAttention(H, 4, batch_first=True, bias=False)
        self.ffn = nn.Sequential(nn.Linear(H, H * 4), nn.GELU(), nn.Linear(H * 4, H))
        self.n1 = nn.LayerNorm(H); self.n2 = nn.LayerNorm(H)
    def forward(self, x):
        S = x.size(1)
        m = torch.triu(torch.ones(S, S, device=x.device) * float('-inf'), 1)
        a, _ = self.atn(x, x, x, attn_mask=m)
        x = self.n1(x + a)
        return self.n2(x + self.ffn(x))

class TLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(vs, H)
        self.layers = nn.ModuleList([TBlk() for _ in range(L)])
        self.head = nn.Linear(H, vs, bias=False)
    def forward(self, x):
        h = self.emb(x)
        for l in self.layers: h = l(h)
        return self.head(h)

def train(model, loader, steps, dev):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler()
    crit = nn.CrossEntropyLoss(ignore_index=0)
    hist = []; it = iter(loader); t0 = time.time()
    for s in range(1, steps + 1):
        try: x, t = next(it)
        except: it = iter(loader); x, t = next(it)
        x = x[:, :256].to(dev, non_blocking=True)
        t = t[:, :256].to(dev, non_blocking=True)
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
        if s % 200 == 0 or s == 1:
            a = sum(hist[-min(50, s):]) / min(50, s)
            elapsed = time.time() - t0
            print(f'  s{s:4d} loss={loss.item():.4f} avg50={a:.4f}  {s*4*256/elapsed:.0f}tok/s', flush=True)
    return hist

results = {}

# Train WDLM-Neural
torch.manual_seed(42)
m_wdl = WN(vs, H, L).to(dev)
p_wdl = sum(p.numel() for p in m_wdl.parameters() if p.requires_grad)
print(f'\n=== WDLM-Neural: {p_wdl:,}p ===', flush=True)
h_wdl = train(m_wdl, ld, STEPS, dev)
results['WDLM-Neural'] = h_wdl
torch.cuda.empty_cache()

# Train Transformer
torch.manual_seed(42)
m_tr = TLM().to(dev)
p_tr = sum(p.numel() for p in m_tr.parameters() if p.requires_grad)
print(f'\n=== Transformer: {p_tr:,}p ===', flush=True)
h_tr = train(m_tr, ld, STEPS, dev)
results['Transformer'] = h_tr
torch.cuda.empty_cache()

# Report
aw = sum(h_wdl[-100:]) / 100
at = sum(h_tr[-100:]) / 100
print(f'\n=== Convergence (last 100 steps avg) ===')
print(f'WDLM-Neural: {h_wdl[0]:.2f} -> {aw:.4f}  ({p_wdl:,}p)')
print(f'Transformer:  {h_tr[0]:.2f} -> {at:.4f}  ({p_tr:,}p)')
print(f'delta: {aw-at:+.4f}')

# Plot
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    w = 50
    sw = [sum(h_wdl[max(0,i-w+1):i+1])/len(h_wdl[max(0,i-w+1):i+1]) for i in range(len(h_wdl))]
    st = [sum(h_tr[max(0,i-w+1):i+1])/len(h_tr[max(0,i-w+1):i+1]) for i in range(len(h_tr))]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(sw, label='WDLM-Neural', color='#FF5722', lw=2.5)
    ax.plot(st, label='Transformer', color='#2196F3', lw=2.5)
    ax.legend(); ax.set_title(f'Convergence (H={H}, L={L})')
    ax.set_xlabel('Step'); ax.set_ylabel('Loss (SMA-50)'); ax.grid(alpha=0.3)
    fig.text(0.5, 0.01, f'WDLM-Neural({p_wdl:,}p)={aw:.4f}  Transformer({p_tr:,}p)={at:.4f}  delta={aw-at:+.4f}',
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    os.makedirs('./out', exist_ok=True)
    plt.savefig('./out/converge_wdl_vs_tr.png', dpi=150, bbox_inches='tight')
    json.dump({'wdlm_neural': h_wdl, 'transformer': h_tr}, open('./out/converge_wdl_vs_tr.json', 'w'))
    print('[Plot] ./out/converge_wdl_vs_tr.png')
except Exception as e:
    print(f'plot err: {e}')
