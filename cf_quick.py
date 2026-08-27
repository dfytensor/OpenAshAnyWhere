"""只测 CopyFirst eval: α=0.50 vs α=0.88"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
device = torch.device("cuda")
V=32; E=0; I=1; H=128

class B(nn.Module):
    def __init__(s, a):
        super().__init__(); s.H=H; s.ns=4; hd=H*2
        s.e=nn.Embedding(V,H); s.ii=nn.Linear(H,H); s.inm=nn.LayerNorm(H)
        s.ph=nn.ModuleList([nn.Linear(H,hd,0) for _ in range(4)])
        s.pi=nn.ModuleList([nn.Linear(H,hd,0) for _ in range(4)])
        s.Wf=nn.ModuleList([nn.Linear(hd*2,hd) for _ in range(4)])
        s.Wi=nn.ModuleList([nn.Linear(hd*2,hd) for _ in range(4)])
        s.Wc=nn.ModuleList([nn.Linear(hd*2,hd) for _ in range(4)])
        s.po=nn.ModuleList([nn.Linear(hd,H) for _ in range(4)])
        s.n=nn.ModuleList([nn.LayerNorm(H) for _ in range(4)])
        s.sp=nn.ModuleList([nn.Linear(H,H) for _ in range(4)])
        for w in s.Wf: nn.init.constant_(w.bias,1.0)
        for w in s.Wi: nn.init.constant_(w.bias,-2.0)
        s.register_buffer('al',torch.sigmoid(torch.tensor(a)))
        s.fu=nn.Linear(H*4,H); s.fn=nn.LayerNorm(H); s.h=nn.Linear(H,V)
    def forward(s,x,hp=None):
        B,T=x.shape
        if hp is None: h=[torch.zeros(B,H,device=device) for _ in range(4)]
        else: h=[c.clone() for c in hp]
        xe=s.e(x); o=[]
        for t in range(T):
            inp=s.inm(s.ii(xe[:,t,:])); nh=[]
            for ss in range(4):
                if t%(2**ss)==0:
                    hp_=s.ph[ss](h[ss]); ip_=s.pi[ss](inp); c=torch.cat([hp_,ip_],-1)
                    f=torch.sigmoid(s.Wf[ss](c)); i=torch.sigmoid(s.Wi[ss](c))
                    cand=s.po[ss](f*hp_+i*torch.tanh(s.Wc[ss](c)))
                    nh.append(s.n[ss](s.al*h[ss]+(1-s.al)*cand))
                else: nh.append(h[ss])
            h=nh; fused=s.fu(torch.cat(h,-1))
            fused=fused+sum(s.sp[ss](h[ss]) for ss in range(4))/4
            o.append(s.h(s.fn(fused+inp)).unsqueeze(1))
        return torch.cat(o,1),h

def mf(bs,nl):
    t=torch.randint(2,V,(bs,)); n=torch.randint(2,V,(bs,nl))
    e=torch.full((bs,1),E,dtype=torch.long)
    x=torch.cat([t.unsqueeze(1),n,e],1); y=torch.full_like(x,I); y[:,-1]=t
    return x,y

for a_init, a_val in [(0.5, 0.50), (2.0, 0.88)]:
    torch.manual_seed(42); m=B(a_init).to(device)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,2500)
    m.train(); best=float('inf'); t0=time.time()
    for st in range(1,2501):
        x,y=mf(64,random.randint(4,64)); x,y=x.to(device),y.to(device)
        log,_=m(x); loss=F.cross_entropy(log[:,-1,:],y[:,-1],ignore_index=I)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
        opt.step(); sch.step()
        if loss.item()<best: best=loss.item()
    print(f"α={a_val:.2f}: best={best:.5f} {time.time()-t0:.0f}s",flush=True)
    m.eval(); dists=[4,64,256,1024,4096,8192,16384,32768,65536]
    for d in dists:
        eb=64 if d<=4096 else 8; c=0;total=0
        for _ in range(4 if d<=4096 else 2):
            x,y=mf(eb,d); x,y=x.to(device),y.to(device)
            log,_=m(x); c+=(log[:,-1,:].argmax(-1)==y[:,-1]).sum().item(); total+=eb
        print(f"  dist={d:5d} acc={c/total*100:.1f}%",flush=True)
    del m; torch.cuda.empty_cache()

# Also baseline Orig-4sc for comparison
class Orig(nn.Module):
    def __init__(s):
        super().__init__(); s.H=H; s.ns=4
        s.e=nn.Embedding(V,H); s.ii=nn.Linear(H,H)
        s.Wf=nn.ModuleList([nn.Linear(H*2,H) for _ in range(4)])
        s.Wi=nn.ModuleList([nn.Linear(H*2,H) for _ in range(4)])
        s.Wc=nn.ModuleList([nn.Linear(H*2,H) for _ in range(4)])
        for w in s.Wf: nn.init.constant_(w.bias,1.0)
        for w in s.Wi: nn.init.constant_(w.bias,-2.0)
        s.fu=nn.Linear(H*4,H); s.n=nn.LayerNorm(H); s.h=nn.Linear(H,V)
    def forward(s,x,hp=None):
        B,T=x.shape
        if hp is None: h=[torch.zeros(B,H,device=device) for _ in range(4)]
        else: h=[c.clone() for c in hp]
        xe=s.e(x); o=[]
        for t in range(T):
            inp=s.ii(xe[:,t,:]); nh=[]
            for ss in range(4):
                if t%(2**ss)==0:
                    c=torch.cat([h[ss],inp],-1)
                    f=torch.sigmoid(s.Wf[ss](c)); i=torch.sigmoid(s.Wi[ss](c))
                    nh.append(f*h[ss]+i*torch.tanh(s.Wc[ss](c)))
                else: nh.append(h[ss])
            h=nh; fused=s.n(s.fu(torch.cat(h,-1)))
            o.append(s.h(fused).unsqueeze(1))
        return torch.cat(o,1),h

torch.manual_seed(42); mo=Orig().to(device)
opt=torch.optim.AdamW(mo.parameters(),lr=1e-3,weight_decay=0.01)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,2500)
mo.train(); best_o=float('inf'); t0=time.time()
for st in range(1,2501):
    x,y=mf(64,random.randint(4,64)); x,y=x.to(device),y.to(device)
    log,_=mo(x); loss=F.cross_entropy(log[:,-1,:],y[:,-1],ignore_index=I)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(mo.parameters(),1.0)
    opt.step(); sch.step()
    if loss.item()<best_o: best_o=loss.item()
print(f"\nOrig-4sc: best={best_o:.5f} {time.time()-t0:.0f}s",flush=True)
mo.eval()
for d in dists:
    eb=64 if d<=4096 else 8; c=0;total=0
    for _ in range(4 if d<=4096 else 2):
        x,y=mf(eb,d); x,y=x.to(device),y.to(device)
        log,_=mo(x); c+=(log[:,-1,:].argmax(-1)==y[:,-1]).sum().item(); total+=eb
    print(f"  dist={d:5d} acc={c/total*100:.1f}%",flush=True)

print("\nDone.")
