"""高频采样: I_v2 (rank+V) vs test_accuracy 的时序领先/滞后关系.
每 10 步采样 rank/V/I_v2/test_acc, 看 rank 变化是否领先 test_acc 跳变.
这是正确的验证——用 test_acc (干净的理解/记忆二元判据) 替代 ppl.
"""
import torch, torch.nn.functional as F, math, os, sys, numpy as np, time
sys.path.insert(0, r'F:\rwkv\frsmash_v36')
from frsmash_v36 import FRSMASHv36
DEV='cuda'

def make_modadd(p, seed=0):
    g=torch.Generator().manual_seed(seed)
    a=torch.arange(p); A,B=torch.meshgrid(a,a,indexing='ij')
    pairs=torch.stack([A.flatten(),B.flatten()],1)
    Y=(pairs[:,0]+pairs[:,1])%p
    perm=torch.randperm(p*p,generator=g); ntr=int(0.3*p*p); tr,te=perm[:ntr],perm[ntr:]
    EQ=p
    def seq(idx):
        n=idx.numel(); return torch.cat([pairs[idx],torch.full((n,1),EQ,dtype=torch.long)],1)
    return seq(tr).to(DEV),Y[tr].to(DEV),seq(te).to(DEV),Y[te].to(DEV),p

@torch.no_grad()
def msr(m):
    rs=[]
    for pp in m.parameters():
        if pp.dim()==2 and min(pp.shape)>1:
            W=pp.detach().float(); sv=torch.linalg.svdvals(W)
            fro=(sv**2).sum().item(); spec=sv[0].item()**2
            if spec>1e-12: rs.append(fro/spec)
    return float(np.mean(rs)) if rs else 1.0

@torch.no_grad()
def eacc(model, X, Y):
    model.eval()
    bs=512; c=0; n=0
    with torch.no_grad():
        for i in range(0,X.size(0),bs):
            pred=model(X[i:i+bs])[:,-1,:].argmax(-1)
            c+=int((pred==Y[i:i+bs]).sum()); n+=pred.size(0)
    return c/max(n,1)

def run(p=113, seed=0, steps=8000, sample_every=10):
    torch.manual_seed(seed)
    Xtr,Ytr,Xte,Yte,_=make_modadd(p,seed)
    model=FRSMASHv36(p+2,128,8,4,n_slots=4).to(DEV)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.1)
    Henv=math.log(p); LN2=math.log(2)
    # 分块训练
    n=Xtr.size(0); bs=1024; nc=(n+bs-1)//bs
    chunks=[(Xtr[i*bs:(i+1)*bs],Ytr[i*bs:(i+1)*bs]) for i in range(nc)]
    log=[]
    for st in range(1,steps+1):
        model.train()
        x,y=chunks[st%nc]
        lo=model(x)[:,-1,:]; V=F.cross_entropy(lo,y)
        opt.zero_grad(); V.backward(); opt.step()
        if st%sample_every==0:
            te=eacc(model,Xte,Yte)
            tr=eacc(model,Xtr,Ytr)
            rank=msr(model)
            vv=float(V.detach())
            I_v2=(rank+vv)/(Henv*LN2)
            log.append((st,vv,rank,I_v2,tr,te))
            if st%500==0:
                print(f'  st{st}: V={vv:.4f} rank={rank:.2f} I_v2={I_v2:.3f} tr={tr:.3f} te={te:.3f}',flush=True)
    return np.array(log)

print('=== 高频采样: rank vs test_acc 时序 (p=113, 每10步) ===\n',flush=True)

all_results={}
for seed in [0,1,2]:
    print(f'--- seed {seed} ---',flush=True)
    log=run(113,seed,8000,10)
    all_results[seed]=log
    # 找 test_acc 首次>0.5 的步
    te=log[:,5]; grok_step=log[np.argmax(te>0.5),0] if any(te>0.5) else -1
    # 找 rank 首次下降>30% (相对前100步均值)
    rank=log[:,2]
    rank_drop_step=-1
    for i in range(10,len(rank)):
        baseline=np.mean(rank[max(0,i-10):i])
        if rank[i]<baseline*0.7:  # 下降30%
            rank_drop_step=log[i,0]; break
    # 找 I_v2 首次下降>20%
    Iv2=log[:,3]; iv2_drop_step=-1
    for i in range(10,len(Iv2)):
        baseline=np.mean(Iv2[max(0,i-10):i])
        if Iv2[i]<baseline*0.8:
            iv2_drop_step=log[i,0]; break
    print(f'  rank 持续下降起点: step {rank_drop_step}')
    print(f'  I_v2 持续下降起点: step {iv2_drop_step}')
    print(f'  test_acc 首次>0.5: step {int(grok_step)}')
    if rank_drop_step>0 and grok_step>0:
        lead_rank=rank_drop_step-grok_step
        print(f'  => rank {"领先" if lead_rank<0 else "滞后"} test_acc {abs(int(lead_rank))}步')
    if iv2_drop_step>0 and grok_step>0:
        lead_iv2=iv2_drop_step-grok_step
        print(f'  => I_v2  {"领先" if lead_iv2<0 else "滞后"} test_acc {abs(int(lead_iv2))}步')
    print()

# 互相关分析 (seed 0, 最详细)
print('=== 互相关: rank/I_v2 vs test_acc (seed 0) ===')
log=all_results[0]
steps=log[:,0]; rank=log[:,2]; Iv2=log[:,3]; te=log[:,5]
# 只取 grokking 附近 [grok-2000, grok+2000]
te_grok=steps[np.argmax(te>0.5)] if any(te>0.5) else len(steps)//2
mask=(steps>=max(0,te_grok-2000))&(steps<=te_grok+2000)
if mask.sum()>20:
    r_s=rank[mask]; i_s=Iv2[mask]; t_s=te[mask]
    def ncc(x,y):
        x=(x-x.mean())/(x.std()+1e-9); y=(y-y.mean())/(y.std()+1e-9)
        c=np.correlate(x,y,'full'); return c
    cc_r=ncc(-r_s, t_s)  # rank下降 vs te上升 (取负让方向一致)
    cc_i=ncc(-i_s, t_s)
    best_r=np.argmax(cc_r)-len(r_s)//2
    best_i=np.argmax(cc_i)-len(i_s)//2
    print(f'  rank vs test_acc: lag={best_r*10:+d}步 ({"rank领先" if best_r<0 else "rank滞后" if best_r>0 else "同步"})')
    print(f'  I_v2 vs test_acc: lag={best_i*10:+d}步 ({"I_v2领先" if best_i<0 else "I_v2滞后" if best_i>0 else "同步"})')

# 总结
print(f'\n=== 总结 ===')
for s in [0,1,2]:
    log=all_results[s]; te=log[:,5]; rank=log[:,2]; Iv2=log[:,3]
    gs=steps[np.argmax(te>0.5)] if any(te>0.5) else -1
    rd=next((log[i,0] for i in range(10,len(rank)) if rank[i]<np.mean(rank[max(0,i-10):i])*0.7),-1)
    idd=next((log[i,0] for i in range(10,len(Iv2)) if Iv2[i]<np.mean(Iv2[max(0,i-10):i])*0.8),-1)
    print(f'  seed{s}: grok={int(gs)} rank_drop={int(rd)} I_v2_drop={int(idd)} | '
          f'rank_lead={int(rd-gs) if rd>0 and gs>0 else "?":>5} I_v2_lead={int(idd-gs) if idd>0 and gs>0 else "?":>5}')
