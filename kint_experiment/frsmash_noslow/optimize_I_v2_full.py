"""I_v2 三方向全面实验:
A. 更多种子(5+): p=113 取 I_v2 grokking 后分布
B. 更多任务(p=29/59/113/197): I_v2 跨任务归一化
C. 跨规模(d=128/256/384): I_v2 是否随规模下降(更高效=更智能)
"""
import torch, torch.nn.functional as F, math, os, sys, numpy as np
sys.path.insert(0, r'F:\rwkv\frsmash_v36')
from frsmash_v36 import FRSMASHv36
DEV='cuda'

def make_modadd(p, seed=0):
    g=torch.Generator().manual_seed(seed)
    a=torch.arange(p); A,B=torch.meshgrid(a,a,indexing='ij')
    pairs=torch.stack([A.flatten(),B.flatten()],1)
    Y=(pairs[:,0]+pairs[:,1])%p
    perm=torch.randperm(p*p,generator=g); ntr=int(0.4*p*p); tr,te=perm[:ntr],perm[ntr:]
    EQ=p
    def seq(idx):
        n=idx.numel(); return torch.cat([pairs[idx],torch.full((n,1),EQ,dtype=torch.long)],1)
    return seq(tr).to(DEV),Y[tr].to(DEV),seq(te).to(DEV),Y[te].to(DEV),p

@torch.no_grad()
def mean_stable_rank(m):
    rs=[]
    for p in m.parameters():
        if p.dim()==2 and min(p.shape)>1:
            W=p.detach().float(); sv=torch.linalg.svdvals(W)
            fro=(sv**2).sum().item(); spec=sv[0].item()**2
            if spec>1e-12: rs.append(fro/spec)
    return float(np.mean(rs)) if rs else 1.0

def eval_acc(model, X, Y):
    model.eval()
    with torch.no_grad():
        bs=512; c=0; n=0
        for i in range(0,X.size(0),bs):
            pred=model(X[i:i+bs])[:,-1,:].argmax(-1)
            c+=int((pred==Y[i:i+bs]).sum()); n+=pred.size(0)
    return c/max(n,1)

def run(p, seed, d=128, L=4, steps=10000, wd=0.1):
    torch.manual_seed(seed)
    Xtr,Ytr,Xte,Yte,_=make_modadd(p,seed)
    model=FRSMASHv36(p+2,d,8,L,n_slots=4).to(DEV)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=wd)
    Henv=math.log(p); LN2=math.log(2)
    # 分块训练 (GLA kernel 限制大 batch)
    n=Xtr.size(0); bs=min(2048,n); nchunks=(n+bs-1)//bs
    chunks=[(Xtr[i*bs:(i+1)*bs],Ytr[i*bs:(i+1)*bs]) for i in range(nchunks)]
    log=[]
    for st in range(1,steps+1):
        model.train()
        x,y=chunks[st%nchunks]
        lo=model(x)[:,-1,:]; V=F.cross_entropy(lo,y)
        opt.zero_grad(); V.backward(); opt.step()
        if st%100==0:
            te=eval_acc(model,Xte,Yte)
            rank=mean_stable_rank(model)
            V_val=float(V.detach())
            I_v2=(rank+V_val)/(Henv*LN2)
            log.append((st,V_val,rank,te,I_v2))
            if te>0.5 and len([l for l in log if l[3]>0.5])==1:
                print(f'    GROK p{p} s{seed} d{d} st{st}: rank={rank:.1f} I_v2={I_v2:.3f} te={te:.3f}',flush=True)
    plat=[l for l in log if l[3]>0.5]
    if not plat: plat=log[int(len(log)*0.8):]
    return dict(p=p,seed=seed,d=d,
                I_v2=np.median([l[4] for l in plat]),
                rank=np.median([l[2] for l in plat]),
                te_best=max(l[3] for l in log),
                grok=max(l[3] for l in log)>0.5)

# ============ A: 5 seeds p=113 ============
print('='*60)
print('A: 5 seeds p=113 d=128 (I_v2 grokking 分布)')
print('='*60,flush=True)
resA=[run(113,s,d=128,steps=10000) for s in [0,1,2,3,4]]
I_A=np.array([r['I_v2'] for r in resA])
print(f'\n  I_v2: {[f"{v:.3f}" for v in I_A]}')
print(f'  mean={I_A.mean():.3f} std={I_A.std():.3f} CV={I_A.std()/I_A.mean()*100:.1f}%')
print(f'  grokked: {sum(r["grok"] for r in resA)}/5')

# ============ B: 跨任务 p=29/59/113 ============
print(f'\n{"="*60}')
print('B: 跨任务 p=29/59/113 (I_v2 归一化)')
print('='*60,flush=True)
resB=[]
for p in [29,59,113]:
    steps={29:6000,59:8000,113:10000}[p]
    r=run(p,0,d=128,steps=steps)
    resB.append(r)
    print(f'  p={p:>3} H_env={math.log(p):.2f}: I_v2={r["I_v2"]:.3f} rank={r["rank"]:.1f} '
          f'grok={r["grok"]} te={r["te_best"]:.3f}',flush=True)
print(f'\n  I_v2*H_env*ln2 = rank+V (应该跨任务接近?):')
for r in resB:
    raw=r['I_v2']*math.log(r['p'])*math.log(2)
    print(f'    p={r["p"]:>3}: raw={raw:.2f} (rank+V at convergence)')

# ============ C: 跨规模 d=128/256/384 ============
print(f'\n{"="*60}')
print('C: 跨规模 d=128/256/384 p=59 (I_v2 是否随规模下降)')
print('='*60,flush=True)
resC=[]
for d in [128,256,384]:
    r=run(59,0,d=d,L=4,steps=8000)
    resC.append(r)
    n=sum(pp.numel() for pp in FRSMASHv36(61,d,8,4,n_slots=4).parameters())
    print(f'  d={d:>3} ({n/1e6:.1f}M): I_v2={r["I_v2"]:.3f} rank={r["rank"]:.1f} '
          f'grok={r["grok"]} te={r["te_best"]:.3f}',flush=True)
print(f'\n  趋势: I_v2 随 d {"下降(大模型更高效)" if resC[-1]["I_v2"]<resC[0]["I_v2"] else "不降"}')

print(f'\n{"="*60}')
print('总结')
print('='*60)
print(f'A (5 seeds p=113): I_v2 mean={I_A.mean():.3f} CV={I_A.std()/I_A.mean()*100:.1f}%')
print('B (跨任务):', [round(r['I_v2'],3) for r in resB])
print('C (跨规模):', [round(r['I_v2'],3) for r in resC])
