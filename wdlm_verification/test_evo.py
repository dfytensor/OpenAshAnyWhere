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

voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1

# Pre-tokenized dataset
class D(torch.utils.data.Dataset):
    def __init__(s): pass
    @staticmethod
    def cf(items):
        p = pad_sequence(items, batch_first=True, padding_value=0)
        return p[:, :-1], p[:, 1:]

# Build pre-tokenized data once
tk = voc
is_ = tk.token_to_id.get('<|im_start|>'); ie = tk.token_to_id.get('<|im_end|>')
uid = tk.token_to_id.get('<|user|>'); aid = tk.token_to_id.get('<|agent|>')
ts = tk.token_to_id.get('<|think|>'); te = tk.token_to_id.get('<|end_think|>')

data = []
cnt = 0
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
                m += [is_, uid] + tk.encode(ct) + [ie]
            elif r in ('gpt', 'assistant', 'agent'):
                th = None; rp = ct
                if '<think>' in ct:
                    a = ct.find('<think>') + 7; b = ct.find('</think>')
                    if b != -1: th = ct[a:b].strip(); rp = ct[:a - 7] + ct[b + 8:]
                import re
                rm = re.search(r'<\s*response\s*>(.*?)$', rp, re.DOTALL | re.I)
                if rm: rp = rm.group(1).strip()
                m += [is_, aid]
                if th: m += [ts] + tk.encode(th) + [te]
                if rp: m += tk.encode(rp)
                m += [ie]
        if len(m) > 256: m = m[:256]
        if len(m) >= 4:
            data.append(torch.tensor(m, dtype=torch.long))
        cnt += 1

print(f'Pre-tokenized: {len(data)} samples', flush=True)

class FastDS(torch.utils.data.Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

n = min(len(data), 3000)
idx = torch.randperm(len(data))[:n].tolist()
ds = FastDS([data[i] for i in idx])
ld = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=D.cf, drop_last=True)
dev = torch.device('cuda:0')
STEPS = 300

def train(model, loader, steps, dev):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler()
    crit = nn.CrossEntropyLoss(ignore_index=0)
    hist = []; it = iter(loader); t0 = time.time()
    for s in range(1, steps + 1):
        try: x, t = next(it)
        except: it = iter(loader); x, t = next(it)
        x = x.to(dev, non_blocking=True); t = t.to(dev, non_blocking=True)
        lr = 5e-5 * (0.1 + 0.45 * (1 + math.cos(math.pi * s / steps)))
        for pg in opt.param_groups: pg['lr'] = lr
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            o, _ = model(x); loss = crit(o.view(-1, vs), t.view(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        hist.append(float(loss.detach()))
        if s % 100 == 0 or s == 1:
            a = sum(hist[-50:]) / min(50, len(hist))
            elapsed = time.time() - t0
            print(f'  s{s:4d} loss={loss.item():.4f} avg50={a:.4f}  {s*4*256/elapsed:.0f}tok/s')
    return hist

results = {}
for es, nm in [(1, 'evo=1'), (3, 'evo=3'), (5, 'evo=5')]:
    torch.manual_seed(42)
    m = WaveDynamicsLM_Real(vs, 128, 2, es).to(dev)
    p = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f'\n=== {nm}: {p:,}p ===')
    h = train(m, ld, STEPS, dev); torch.cuda.empty_cache()
    results[nm] = h

print(f'\n=== Final avg50 ===')
bl = sum(results['evo=1'][-50:]) / 50
for nm, h in results.items():
    a = sum(h[-50:]) / 50
    print(f'{nm:10s} loss={a:.4f}  delta={a-bl:+.4f}')
