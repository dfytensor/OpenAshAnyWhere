"""WDLM-Neural vs WDLM-Real State -- CUDA GPU 训练速度 & Loss 对比"""
import torch, time, sys, math, os, json
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, 'F:/OpenASH2605')
sys.path.insert(0, 'F:/OpenASH2605/wdlm_verification')
os.chdir('F:/OpenASH2605')
from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from wdlm_real import WaveDynamicsLM_Real
from wdlm_neural import WaveDynamicsLanguageModel as WDLM_Neural

voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1

# Pre-tokenized dataset
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

print(f'Pre-tokenized: {len(data)} samples', flush=True)

class FastDS(torch.utils.data.Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]
    @staticmethod
    def cf(items):
        p = pad_sequence(items, batch_first=True, padding_value=0)
        return p[:, :-1], p[:, 1:]

n = len(data)
dev = torch.device('cuda:0')
H, L = 128, 2
STEPS = 300

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
            if isinstance(out, tuple):
                o = out[0]
            else:
                o = out
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
    return hist, (time.time() - t0), sum(hist[-50:]) / 50

# Create DataLoader
ds = FastDS(data[:min(n, 2000)])
ld = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=FastDS.cf, drop_last=True)
print(f'Batches: {len(ld)}', flush=True)

results = {}

# WDLM-Real State (evo=1)
torch.manual_seed(42)
m_real = WaveDynamicsLM_Real(vs, H, L, 1).to(dev)
p_real = sum(p.numel() for p in m_real.parameters() if p.requires_grad)
print(f'\n=== WDLM-Real State: {p_real:,}p ===', flush=True)
h_real, t_real, a_real = train(m_real, ld, STEPS, dev)
results['WDLM-Real'] = (p_real, a_real, t_real, h_real)
torch.cuda.empty_cache()

# WDLM-Neural
torch.manual_seed(42)
m_neural = WDLM_Neural(vs, H, L).to(dev)
p_neural = sum(p.numel() for p in m_neural.parameters() if p.requires_grad)
print(f'\n=== WDLM-Neural: {p_neural:,}p ===', flush=True)
h_neural, t_neural, a_neural = train(m_neural, ld, STEPS, dev)
results['WDLM-Neural'] = (p_neural, a_neural, t_neural, h_neural)
torch.cuda.empty_cache()

# Report
print(f'\n{"="*60}')
print(f'Results (H={H}, L={L}, steps={STEPS})')
print(f'{"="*60}')
print(f'{"Model":20s} {"Params":>10s} {"Loss":>8s} {"Tok/s":>8s} {"Tok/s/M":>10s}')
for name, (p, a, t, _) in results.items():
    tok = STEPS * 4 * 256 / t
    print(f'{name:20s} {p:>10,} {a:>8.4f} {tok:>8.0f} {tok/(p/1e6):>10.0f}')
