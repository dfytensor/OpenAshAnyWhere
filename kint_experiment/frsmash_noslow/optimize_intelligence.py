r"""优化 I: 用 stable_rank 替代 weight_norm 做 C (回路复杂度), 去掉 eta (已内含).
I_v2 = (rank_eff + V) / (H_env * tau) / ln2
- rank_eff: 权重矩阵平均稳定秩 (grokking 时从 ~30 塌到 ~3)
- V: 交叉熵损失
- H_env: 环境熵率 (log(p) for modadd)
- tau=1, ln2 归一化

关键: memorization 时 rank 高 -> I 高 (蠢); grokking 后 rank 塌 -> I 低 (智能).
跨种子 + 跨任务(p=59/113/197) 验证稳定性.
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
    perm=torch.randperm(p*p,generator=g); ntr=int(0.3*p*p); tr,te=perm[:ntr],perm[ntr:]
    EQ=p
    def seq(idx):
        n=idx.numel(); return torch.cat([pairs[idx],torch.full((n,1),EQ,dtype=torch.long)],1)
    return seq(tr).to(DEV),Y[tr].to(DEV),seq(te).to(DEV),Y[te].to(DEV),p

@torch.no_grad()
def mean_stable_rank(m):
    rs=[]
    for p in m.parameters():
        if p.dim()==2 and p.shape[0]>1:
            W=p.detach().float()
            sv=torch.linalg.svdvals(W)
            fro=(sv**2).sum().item()
            spec=sv[0].item()**2
            if spec>0: rs.append(fro/spec)
    return float(np.mean(rs)) if rs else 1.0

@torch.no_grad()
def weight_norm_sq(m):
    return float(sum((p.detach().float()**2).sum() for p in m.parameters()))

def run_one(p, seed, steps):
    torch.manual_seed(seed)
    Xtr,Ytr,Xte,Yte,_=make_modadd(p,seed)
    model=FRSMASHv36(p+2,128,8,4,n_slots=4).to(DEV)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.1)
    Henv=math.log(p); LN2=math.log(2)
    log=[]
    for st in range(1,steps+1):
        model.train()
        lo=model(Xtr)[:, -1, :]
        V=F.cross_entropy(lo, Ytr)
        opt.zero_grad(); V.backward(); opt.step()
        if st%100==0:
            model.eval()
            with torch.no_grad():
                def acc(X,Y):
                    bs=1024;c=0;n=0
                    for i in range(0,X.size(0),bs):
                        pred=model(X[i:i+bs])[:,-1,:].argmax(-1)
                        c+=int((pred==Y[i:i+bs]).sum());n+=pred.size(0)
                    return c/max(n,1)
                tr_a=acc(Xtr,Ytr); te_a=acc(Xte,Yte)
            rank=mean_stable_rank(model)
            V_val=float(V.detach()); C_wn=weight_norm_sq(model)
            # I_v1 (原版: weight_norm)
            K=C_wn*V_val
            I_v1=(C_wn*1e-6+V_val)/(Henv*LN2)
            # I_v2 (新版: stable_rank)
            I_v2=(rank+V_val)/(Henv*LN2)
            log.append((st,V_val,C_wn,K,rank,tr_a,te_a,I_v1,I_v2))
            if st%1000==0:
                print(f'    p{p} s{seed} st{st}: V={V_val:.3f} rank={rank:.1f} '
                      f'tr={tr_a:.2f} te={te_a:.3f} I_v2={I_v2:.3f}',flush=True)
    # 收敛平台 = 末20%
    plat=log[int(len(log)*0.8):]
    te_best=max(l[6] for l in log)
    return dict(
        p=p, seed=seed, te_best=te_best,
        K_med=np.median([l[3] for l in plat]),
        I_v1_med=np.median([l[7] for l in plat]),
        I_v2_med=np.median([l[8] for l in plat]),
        rank_med=np.median([l[5] for l in plat]),
        V_med=np.median([l[1] for l in plat]),
        grokked=te_best>0.5,
    )

print('='*70)
print('I_v2 = (stable_rank + V) / (H_env * ln2): 跨种子+跨任务')
print('='*70)
all_results=[]
for p in [59, 113, 197]:
    steps = {59:8000, 113:12000, 197:15000}[p]
    print(f'\n--- p={p} (H_env={math.log(p):.2f} nats, steps={steps}) ---',flush=True)
    for seed in [0,1,2]:
        r=run_one(p, seed, steps)
        all_results.append(r)
        grok='GROK' if r['grokked'] else 'memorize'
        print(f'  seed{seed}: {grok} te_best={r["te_best"]:.3f} '
              f'K={r["K_med"]:.0f} I_v1={r["I_v1_med"]:.4f} I_v2={r["I_v2_med"]:.3f} '
              f'rank={r["rank_med"]:.1f}',flush=True)

print('\n'+'='*70)
print('跨种子 CV (每个 p 内 3 seeds):')
print(f'{"p":>5} {"指标":>10} {"seed0":>10} {"seed1":>10} {"seed2":>10} {"CV":>8}')
for p in [59,113,197]:
    rs=[r for r in all_results if r['p']==p]
    for metric,name in [('K_med','K_int'),('I_v1_med','I_v1(wn)'),('I_v2_med','I_v2(rank)')]:
        vals=np.array([r[metric] for r in rs])
        cv=vals.std()/(abs(vals.mean())+1e-9)*100
        print(f'{p:>5} {name:>10} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {cv:>7.1f}%')

print('\n跨任务归一化 (不同 p 的 I_v2 应该接近吗? 如果 I 正确归一化了任务难度):')
for p in [59,113,197]:
    rs=[r for r in all_results if r['p']==p]
    I2=np.array([r['I_v2_med'] for r in rs])
    print(f'  p={p}: I_v2 mean={I2.mean():.3f} CV={I2.std()/I2.mean()*100:.1f}%')
