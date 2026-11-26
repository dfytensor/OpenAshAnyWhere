"""V5a Dual-Path vs V1 — 全无 expansion, 公平对比"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
device = torch.device("cuda")
VCF=32; ECF=0; ICF=1; HCF=128

class SBlock(nn.Module):
    """无 expansion 的简单门控块"""
    def __init__(s, d):
        super().__init__()
        s.Wf=nn.Linear(d*2,d); s.Wi=nn.Linear(d*2,d); s.Wc=nn.Linear(d*2,d)
        nn.init.constant_(s.Wf.bias,1.0); nn.init.constant_(s.Wi.bias,-2.0)
        s.n=nn.LayerNorm(d)
    def forward(s,h,i):
        c=torch.cat([h,i],-1); f=torch.sigmoid(s.Wf(c)); i_=torch.sigmoid(s.Wi(c))
        return s.n(f*h + i_*torch.tanh(s.Wc(c)))

class V1(nn.Module):
    def __init__(s,vs,H=256,ns=4):
        super().__init__(); s.H=H; s.ns=ns
        s.e=nn.Embedding(vs,H); s.ii=nn.Linear(H,H); s.inm=nn.LayerNorm(H)
        s.sc=nn.ModuleList([SBlock(H) for _ in range(ns)])
        s.sp=nn.ModuleList([nn.Linear(H,H) for _ in range(ns)])
        s.fu=nn.Linear(H*ns,H); s.fn=nn.LayerNorm(H); s.h=nn.Linear(H,vs)
    def forward(s,x,hp=None):
        B,T=x.shape
        if hp is None: h=[torch.zeros(B,s.H,device=x.device) for _ in range(s.ns)]
        else: h=[c.clone() for c in hp]
        xe=s.e(x); o=[]
        for t in range(T):
            inp=s.inm(s.ii(xe[:,t,:])); nh=[]
            for ss in range(s.ns):
                if t%(2**ss)==0: nh.append(s.sc[ss](h[ss],inp))
                else: nh.append(h[ss])
            h=nh; fu=s.fu(torch.cat(h,-1))
            fu=fu+sum(s.sp[ss](h[ss]) for ss in range(s.ns))/s.ns
            o.append(s.h(s.fn(fu+inp)).unsqueeze(1))
        return torch.cat(o,1),h

class V5a(nn.Module):
    """Dual-Path: LM(2scales全替换) + Memory(4scales全替换) → gate blend"""
    def __init__(s,vs,H=256):
        super().__init__(); s.H=H
        s.e=nn.Embedding(vs,H); s.ii=nn.Linear(H,H); s.inm=nn.LayerNorm(H)
        # LM Path: 2 scales
        s.lm_sc=nn.ModuleList([SBlock(H) for _ in range(2)])
        s.lm_sp=nn.ModuleList([nn.Linear(H,H) for _ in range(2)])
        s.lm_fu=nn.Linear(H*2,H); s.lm_fn=nn.LayerNorm(H)
        # Memory Path: 4 scales
        s.mem_sc=nn.ModuleList([SBlock(H) for _ in range(4)])
        s.mem_sp=nn.ModuleList([nn.Linear(H,H) for _ in range(4)])
        s.mem_fu=nn.Linear(H*4,H); s.mem_fn=nn.LayerNorm(H)
        # Gate: input decides blend ratio
        s.gate=nn.Sequential(nn.Linear(H,H//4),nn.GELU(),nn.Linear(H//4,1),nn.Sigmoid())
        s.out_n=nn.LayerNorm(H); s.h=nn.Linear(H,vs)

    def forward(s,x,hp=None):
        B,T=x.shape
        lm_h=[torch.zeros(B,s.H,device=x.device) for _ in range(2)]
        mem_h=[torch.zeros(B,s.H,device=x.device) for _ in range(4)]
        xe=s.e(x); o=[]
        for t in range(T):
            inp=s.inm(s.ii(xe[:,t,:]))
            # LM: scales 0,1
            lm_nh=[]
            for ss in range(2):
                if t%(2**ss)==0: lm_nh.append(s.lm_sc[ss](lm_h[ss],inp))
                else: lm_nh.append(lm_h[ss])
            lm_h=lm_nh
            lm_out=s.lm_fn(s.lm_fu(torch.cat(lm_h,-1))
                          +sum(s.lm_sp[ss](lm_h[ss]) for ss in range(2))/2)
            # Memory: scales 0-3
            mem_nh=[]
            for ss in range(4):
                if t%(2**ss)==0: mem_nh.append(s.mem_sc[ss](mem_h[ss],inp))
                else: mem_nh.append(mem_h[ss])
            mem_h=mem_nh
            mem_out=s.mem_fn(s.mem_fu(torch.cat(mem_h,-1))
                            +sum(s.mem_sp[ss](mem_h[ss]) for ss in range(4))/4)
            # Blend
            g=s.gate(inp)  # (B,1)
            out=s.out_n(g*lm_out+(1-g)*mem_out+inp)
            o.append(s.h(out).unsqueeze(1))
        return torch.cat(o,1),lm_h+mem_h

def make_cf(bs,nl):
    t=torch.randint(2,VCF,(bs,)); n=torch.randint(2,VCF,(bs,nl))
    e=torch.full((bs,1),ECF,dtype=torch.long)
    x=torch.cat([t.unsqueeze(1),n,e],1); y=torch.full_like(x,ICF); y[:,-1]=t
    return x,y

# 只跑 CopyFirst (关键对比)
for name, model in [("V1", V1(VCF,HCF,4)), ("V5a-Dual", V5a(VCF,HCF))]:
    torch.manual_seed(42); m=model.to(device)
    n=sum(p.numel() for p in m.parameters())
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,2500)
    m.train(); best=float('inf'); t0=time.time()
    for st in range(1,2501):
        x,y=make_cf(64,random.randint(4,64)); x,y=x.to(device),y.to(device)
        log,_=m(x); loss=F.cross_entropy(log[:,-1,:],y[:,-1],ignore_index=ICF)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
        opt.step(); sch.step()
        if loss.item()<best: best=loss.item()
        if st%500==0: print(f"  [{name}] step{st:5d} best={best:.5f} {time.time()-t0:.0f}s",flush=True)
    print(f"  [{name}] converged: best={best:.5f} params={n:,}",flush=True)
    m.eval(); dists=[4,64,256,1024,4096,8192,16384,32768,65536]
    print(f"  {'Dist':>6} | Acc",flush=True)
    for d in dists:
        eb=64 if d<=4096 else 8; c=0;total=0
        for _ in range(4 if d<=4096 else 2):
            x,y=make_cf(eb,d); x,y=x.to(device),y.to(device)
            log,_=m(x); c+=(log[:,-1,:].argmax(-1)==y[:,-1]).sum().item(); total+=eb
        print(f"  {d:6d} | {c/total*100:.1f}%",flush=True)
    del m; torch.cuda.empty_cache()

print(f"\nDone.")
