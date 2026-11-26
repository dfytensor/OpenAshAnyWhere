"""V6 vs V1 LM Loss 对比"""
import os, sys, time, math, torch
import torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, 'F:/OpenASH2605')
from config import agent_voc_path
from open_ash_voc import OpenASHVoc
from frsm.dataset import PretrainDataset

device = torch.device("cuda")
voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1

class RawBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.Wf=nn.Linear(d*2,d); self.Wi=nn.Linear(d*2,d); self.Wc=nn.Linear(d*2,d)
        nn.init.constant_(self.Wf.bias,1.0); nn.init.constant_(self.Wi.bias,-2.0)
    def forward(self,h,i):
        c=torch.cat([h,i],-1); f=torch.sigmoid(self.Wf(c)); i_=torch.sigmoid(self.Wi(c))
        return f*h+i_*torch.tanh(self.Wc(c))

class V1_LM(nn.Module):
    def __init__(self, vs, H=256, ns=4):
        super().__init__(); self.H=H; self.ns=ns
        self.e=nn.Embedding(vs,H); self.ii=nn.Linear(H,H)
        self.sc=nn.ModuleList([RawBlock(H) for _ in range(ns)])
        self.fu=nn.Linear(H*ns,H); self.fn=nn.LayerNorm(H); self.h=nn.Linear(H,vs)
    def forward(self,x):
        B,T=x.shape; h=[torch.zeros(B,self.H,device=x.device) for _ in range(self.ns)]
        xe=self.e(x); o=[]
        for t in range(T):
            inp=self.ii(xe[:,t,:]); nh=[]
            for s in range(self.ns):
                if t%(2**s)==0: nh.append(self.sc[s](h[s],inp))
                else: nh.append(h[s])
            h=nh; fu=self.fn(self.fu(torch.cat(h,-1)))
            o.append(self.h(fu).unsqueeze(1))
        return torch.cat(o,1)

class V6_LM(nn.Module):
    def __init__(self, vs, H=256, ns=4):
        super().__init__(); self.H=H; self.ns=ns
        self.e=nn.Embedding(vs,H); self.ii=nn.Linear(H,H)
        self.sc=nn.ModuleList([RawBlock(H) for _ in range(ns)])
        self.gates=nn.ModuleList([nn.Sequential(nn.Linear(H*2,H//4),nn.GELU(),nn.Linear(H//4,1),nn.Sigmoid()) for _ in range(ns)])
        self.fu=nn.Linear(H*ns,H); self.fn=nn.LayerNorm(H); self.h=nn.Linear(H,vs)
    def forward(self,x):
        B,T=x.shape; h=[torch.zeros(B,self.H,device=x.device) for _ in range(self.ns)]
        xe=self.e(x); o=[]
        for t in range(T):
            inp=self.ii(xe[:,t,:]); nh=[]
            for s in range(self.ns):
                cand=self.sc[s](h[s],inp)
                g=self.gates[s](torch.cat([h[s],inp],-1))
                nh.append(g*cand+(1-g)*h[s])
            h=nh; fu=self.fn(self.fu(torch.cat(h,-1)))
            o.append(self.h(fu).unsqueeze(1))
        return torch.cat(o,1)

dataset = PretrainDataset("minimind_data/pretrain_t2t_mini.jsonl",voc,max_len=256,max_lines=2000)

def train_lm(model, name, steps=500):
    loader = DataLoader(dataset, batch_size=4, shuffle=True,
                       collate_fn=PretrainDataset.collate_fn, drop_last=True)
    torch.manual_seed(42); model = model.to(device)
    opt = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9,0.95))
    def lrs(o,w,t):
        def f(s):
            if s<w: return s/max(1,w)
            p = (s-w)/max(1,t-w)
            return max(0.0, 0.5*(1.0+math.cos(math.pi*p)))
        return torch.optim.lr_scheduler.LambdaLR(o,f)
    sch=lrs(opt,50,steps); model.train(); best=float('inf'); di=iter(loader); t0=time.time()
    print(f"\n  [{name}] {sum(p.numel() for p in model.parameters()):,}p",flush=True)
    for step in range(steps):
        try: x,t=next(di)
        except: di=iter(loader); x,t=next(di)
        x,t=x.to(device),t.to(device); log=model(x)
        loss=F.cross_entropy(log.reshape(-1,vs),t.reshape(-1),ignore_index=0)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step()
        if loss.item()<best: best=loss.item()
        if (step+1)%100==0: print(f"    step{step+1:4d} loss={loss.item():.4f} best={best:.4f} {time.time()-t0:.0f}s",flush=True)
    model.eval(); tl=0;tt=0
    with torch.no_grad():
        for x,t in loader:
            x,t=x.to(device),t.to(device); log=model(x)
            l=F.cross_entropy(log.reshape(-1,vs),t.reshape(-1),ignore_index=0,reduction='sum')
            tl+=l.item();tt+=(t!=0).sum().item()
        el=tl/tt
    print(f"    => eval={el:.4f} ppl={math.exp(el):.2f}",flush=True)
    del model; torch.cuda.empty_cache()
    return el,best

print(f"{'='*60}")
print(f"  V1 vs V6a LM Loss (500 steps)")
print(f"{'='*60}")

el1,b1 = train_lm(V1_LM(vs,256,4), "V1")
el6,b6 = train_lm(V6_LM(vs,256,4), "V6a")

print(f"\n{'='*60}")
print(f"  Results")
print(f"{'='*60}")
print(f"  V1:  loss={el1:.4f} ppl={math.exp(el1):.2f} best={b1:.4f}")
print(f"  V6a: loss={el6:.4f} ppl={math.exp(el6):.2f} best={b6:.4f}")
print(f"  Δ = {el6-el1:+.4f}  ({abs(el6-el1)/el1*100:.1f}%)")
if abs(el6-el1) < 0.05:
    print(f"  => V6a LM quality ~ EQUIVALENT to V1")
elif el6 < el1:
    print(f"  => V6a BETTER than V1 on LM")
else:
    print(f"  => V6a WORSE than V1 on LM")
print(f"\nDone.")
