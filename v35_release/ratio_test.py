"""HybridFRSM 快慢尺度比例消融"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
device = torch.device("cuda"); V=32; E=0; I=1; H=128

class SlowScaleCell(nn.Module):
    def __init__(self, num_slow, d):
        super().__init__(); self.ns=num_slow
        self.Wf=nn.Parameter(torch.empty(num_slow,d,2*d)); self.bf=nn.Parameter(torch.empty(num_slow,d))
        self.Wi=nn.Parameter(torch.empty(num_slow,d,2*d)); self.bi=nn.Parameter(torch.empty(num_slow,d))
        self.Wc=nn.Parameter(torch.empty(num_slow,d,2*d)); self.bc=nn.Parameter(torch.empty(num_slow,d))
        dh=max(d//4,1)
        self.gW1=nn.Parameter(torch.empty(num_slow,dh,2*d)); self.gb1=nn.Parameter(torch.empty(num_slow,dh))
        self.gW2=nn.Parameter(torch.empty(num_slow,1,dh)); self.gb2=nn.Parameter(torch.empty(num_slow,1))
        self._init(d)
    def _init(self,d):
        for p in [self.Wf,self.Wi,self.Wc,self.gW1,self.gW2]:
            for s in range(self.ns): nn.init.kaiming_uniform_(p[s],a=math.sqrt(5))
        for p in [self.bf,self.bi,self.bc,self.gb1,self.gb2]: nn.init.zeros_(p)
        nn.init.constant_(self.bf,1.0); nn.init.constant_(self.bi,-2.0)
    def forward(self,x_t,h):
        x=x_t.unsqueeze(1).expand(-1,self.ns,-1); g=torch.cat([h,x],-1)
        f=torch.sigmoid(torch.einsum('bnj,nij->bni',g,self.Wf)+self.bf)
        i=torch.sigmoid(torch.einsum('bnj,nij->bni',g,self.Wi)+self.bi)
        c=torch.tanh(torch.einsum('bnj,nij->bni',g,self.Wc)+self.bc)
        cand=f*h+i*c
        h1=F.gelu(torch.einsum('bnj,nij->bni',g,self.gW1)+self.gb1)
        a=torch.sigmoid(torch.einsum('bni,noi->bno',h1,self.gW2)+self.gb2)
        return a*cand+(1-a)*h

class Hybrid(nn.Module):
    def __init__(self,vs,H=128,nf=3,ns=1,K=8):
        super().__init__(); self.H=H; self.nf=nf; self.ns=ns; self.K=K
        self.e=nn.Embedding(vs,H); self.ip=nn.Linear(H,H)
        self.fp=nn.Linear(H,nf*4*H)
        self.sc=SlowScaleCell(ns,H)
        self.fu=nn.Linear((nf+ns)*H,H); self.fn=nn.LayerNorm(H); self.op=nn.Linear(H,vs)
        nn.init.kaiming_uniform_(self.fp.weight,a=math.sqrt(5)); nn.init.zeros_(self.fp.bias)
        nn.init.kaiming_uniform_(self.fu.weight,a=math.sqrt(5)); nn.init.zeros_(self.fu.bias)
    def forward(self,x):
        B,T=x.shape; D=self.H; xe=self.ip(self.e(x))
        # Fast
        fg=self.fp(xe).reshape(B,T,self.nf,4,D)
        a=torch.sigmoid(fg[...,0,:]); f=torch.sigmoid(fg[...,1,:])
        i=torch.sigmoid(fg[...,2,:]); c=torch.tanh(fg[...,3,:])
        A=a*f+(1-a); Bf=a*i*c
        hf=torch.zeros(B,self.nf,D,device=x.device); Hf=[]
        for t in range(T): hf=A[:,t]*hf+Bf[:,t]; Hf.append(hf)
        Hf=torch.stack(Hf,dim=1)
        # Slow
        hs=torch.zeros(B,self.ns,D,device=x.device)
        Hs=torch.zeros(B,T,self.ns,D,device=x.device,dtype=xe.dtype)
        prev=0
        for t in range(0,T,self.K): hs=self.sc(xe[:,t,:],hs); Hs[:,prev:t+1]=hs.unsqueeze(1); prev=t+1
        if prev<T: Hs[:,prev:]=hs.unsqueeze(1)
        # Fusion
        Ha=torch.cat([Hf,Hs],dim=2).reshape(B,T,-1)
        return self.op(self.fn(self.fu(Ha)))

def mf(bs,nl):
    t=torch.randint(2,V,(bs,)); n=torch.randint(2,V,(bs,nl))
    e=torch.full((bs,1),E,dtype=torch.long)
    x=torch.cat([t.unsqueeze(1),n,e],1); y=torch.full_like(x,I); y[:,-1]=t
    return x,y

configs = [
    (3, 1, "3F+1S"),
    (2, 2, "2F+2S"),
    (1, 3, "1F+3S"),
    (2, 1, "2F+1S"),
    (1, 1, "1F+1S"),
    (4, 0, "4F+0S"),
    (0, 4, "0F+4S"),
]

print(f"{'='*70}")
print(f"  HybridFRSM: Fast:Slow Ratio Ablation")
print(f"{'='*70}")

results = {}
for nf, ns, tag in configs:
    torch.manual_seed(42)
    m = Hybrid(V, H, nf, ns, 8).to(device)
    n = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 2500)
    m.train(); best = float('inf'); t0 = time.time()
    for st in range(1, 2501):
        x, y = mf(64, random.randint(4, 64)); x, y = x.to(device), y.to(device)
        log = m(x); loss = F.cross_entropy(log[:, -1, :], y[:, -1], ignore_index=I)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        if loss.item() < best: best = loss.item()
    train_time = time.time() - t0
    # Eval
    m.eval(); accs = {}
    for d in [4, 64, 1024, 4096, 16384, 65536]:
        eb = 64 if d <= 4096 else 8; c = 0; total = 0
        for _ in range(4 if d <= 4096 else 2):
            x, y = mf(eb, d); x, y = x.to(device), y.to(device)
            log = m(x); c += (log[:, -1, :].argmax(-1) == y[:, -1]).sum().item(); total += eb
        accs[d] = c / total * 100
    results[tag] = {'n': n, 'best': best, 'time': train_time, 'accs': accs}
    print(f"  {tag:>8}: {n:>7,}p best={best:.5f} {train_time:.0f}s  CF@65K={accs.get(65536,0):.0f}%", flush=True)
    del m; torch.cuda.empty_cache()

# Summary
print(f"\n{'='*70}")
print(f"  {'Config':>8} | {'Params':>8} | {'best':>8} | {'Time':>6} | {'4':>5} | {'1K':>5} | {'4K':>5} | {'16K':>5} | {'65K':>5}")
print(f"  "+"-"*68)
for tag in [t for _,_,t in configs]:
    r = results[tag]
    a = r['accs']
    print(f"  {tag:>8} | {r['n']:>8,} | {r['best']:8.5f} | {r['time']:5.0f}s | {a.get(4,0):4.0f}% | {a.get(1024,0):4.0f}% | {a.get(4096,0):4.0f}% | {a.get(16384,0):4.0f}% | {a.get(65536,0):4.0f}%")

# Best
best_cf = max(results.keys(), key=lambda k: results[k]['accs'].get(65536, 0))
best_lm = min(results.keys(), key=lambda k: results[k]['best'])
best_speed = min(results.keys(), key=lambda k: results[k]['time'])
print(f"\n  Best CF@65K: {best_cf} ({results[best_cf]['accs'].get(65536,0):.0f}%)")
print(f"  Best LM:     {best_lm} (best={results[best_lm]['best']:.5f})")
print(f"  Fastest:     {best_speed} ({results[best_speed]['time']:.0f}s)")
print(f"\nDone.")
