"""在 FRSMASH grokking 曲线上实测 \mathcal{I}[A] = F_min / (H_env * tau_loop) * eta / ln2
对比 K_int (C*V 下界) 和 \mathcal{I} (智能度) 的跨种子稳定性.
"""
import torch, torch.nn.functional as F, math, os, sys, csv, numpy as np
sys.path.insert(0, r'F:\rwkv\frsmash_v36')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frsmash_v36 import FRSMASHv36
from torch.utils.data import Dataset, DataLoader
DEV='cuda'; VOCAB=23005

# === Modular Addition: 环境熵率可精确计算 ===
def make_modadd(p=113, seed=0):
    g=torch.Generator().manual_seed(seed)
    a=torch.arange(p); A,B=torch.meshgrid(a,a,indexing='ij')
    pairs=torch.stack([A.flatten(),B.flatten()],1)
    Y=(pairs[:,0]+pairs[:,1])%p
    perm=torch.randperm(p*p,generator=g); ntr=int(0.3*p*p); tr,te=perm[:ntr],perm[ntr:]
    EQ=p
    def seq(idx):
        n=idx.numel(); return torch.cat([pairs[idx],torch.full((n,1),EQ,dtype=torch.long)],1)
    return seq(tr).to(DEV),Y[tr].to(DEV),seq(te).to(DEV),Y[te].to(DEV),p

def H_env_modadd(p):
    """环境熵率: modular addition 的条件熵 H(Y|X).
    均匀分布下 Y=(a+b)%p 也是均匀(每个值概率 1/p), H=log(p) nats."""
    return math.log(p)

@torch.no_grad()
def weight_norm_sq(m):
    return float(sum((p.detach().float()**2).sum() for p in m.parameters()))

@torch.no_grad()
def stable_rank_mean(m):
    rs=[]
    for p in m.parameters():
        if p.dim()==2:
            W=p.detach().float()
            fro=(W**2).sum().item(); spec=torch.linalg.svdvals(W)[0].item()**2
            if spec>0: rs.append(fro/spec)
    return float(np.mean(rs)) if rs else 1.0

@torch.no_grad()
def coding_efficiency(m):
    """eta_coding 近似: stable_rank / max_possible_rank(=min(d_in,d_out)).
    越低 = 越压缩 = eta 越高(接近 Landauer 极限)."""
    etas=[]
    for p in m.parameters():
        if p.dim()==2:
            W=p.detach().float()
            fro=(W**2).sum().item(); spec=torch.linalg.svdvals(W)[0].item()**2
            if spec>0:
                sr=fro/spec; max_r=min(W.shape)
                etas.append(sr/max_r)  # 越低越高效
    return float(np.mean(etas)) if etas else 0.5

def run(seed, p=113, steps=5000):
    torch.manual_seed(seed)
    Xtr,Ytr,Xte,Yte,p=make_modadd(p,seed)
    model=FRSMASHv36(p+2,128,8,4,n_slots=4).to(DEV)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.1)
    Henv=H_env_modadd(p)  # nats
    tau_loop=1.0          # 1 training step = 1 因果环路周期

    log=[]
    for st in range(1,steps+1):
        model.train()
        logits=model(Xtr)[:, -1, :]  # 取最后位置(=)
        V=F.cross_entropy(logits, Ytr)
        C=weight_norm_sq(model)
        opt.zero_grad(); V.backward(); opt.step()

        if st%100==0 or st<=10:
            model.eval()
            with torch.no_grad():
                # 分块 eval (GLA kernel 限制大 batch)
                def eval_acc(X,Y):
                    bs=1024; correct=0; total=0
                    for i in range(0,X.size(0),bs):
                        xb=X[i:i+bs]
                        lo=model(xb)[:, -1, :].argmax(-1)
                        correct+=int((lo==Y[i:i+bs]).sum()); total+=lo.size(0)
                    return correct/max(total,1)
                tr_acc=eval_acc(Xtr,Ytr); te_acc=eval_acc(Xte,Yte)
            C_post=weight_norm_sq(model); V_post=float(V.detach())
            F_val=C_post*1e-6+V_post  # F=C*V 是下界约束; F_min取grokking后平台
            K=C_post*V_post
            eta=coding_efficiency(model)
            # \mathcal{I} = F_min / (Henv * tau) * eta / ln2
            # F_min 用当前 C+V (训练中 F 会下降到极小)
            F_cv = C_post*1e-6 + V_post  # 归一化 C 到同量级
            I_val = F_cv / (Henv * tau_loop) * eta / math.log(2)
            log.append((st, V_post, C_post, K, tr_acc, te_acc, eta, I_val))
            if st%500==0:
                print(f'  s{seed} st{st}: V={V_post:.3f} C={C_post:.0f} K={K:.0f} '
                      f'tr={tr_acc:.3f} te={te_acc:.3f} eta={eta:.3f} I={I_val:.4f}',flush=True)

    # 取末 20% 步作为收敛平台 (不要求 grokking)
    platform=log[int(len(log)*0.8):]
    if platform:
        F_min=np.median([l[1]+l[2]*1e-6 for l in platform])
        K_plat=np.median([l[3] for l in platform])
        I_plat=np.median([l[7] for l in platform])
        eta_plat=np.median([l[6] for l in platform])
        te_best=max(l[5] for l in platform)
    else:
        F_min=float('nan'); K_plat=float('nan'); I_plat=float('nan'); eta_plat=float('nan')
    print(f'  [seed{seed}] 末20%: K={K_plat:.0f}  I={I_plat:.4f}  eta={eta_plat:.3f}  te_best={te_best:.3f}',flush=True)
    return K_plat, I_plat, eta_plat, F_min

print('=== \mathcal{I} vs K_int: 跨种子稳定性 (modular addition p=113) ===\n')
results=[]
for s in [0,1,2]:
    K,I,eta,Fmin=run(s)
    results.append((s,K,I,eta,Fmin))

Ks=[r[1] for r in results]; Is=[r[3] for r in results]  # 注意 I 在 index 2
# 修正 index
Ks=[r[1] for r in results]; Is=[r[2] for r in results]; etas=[r[3] for r in results]
print('\n=== 跨种子对比 ===')
print(f'{"指标":>10} {"seed0":>12} {"seed1":>12} {"seed2":>12} {"mean":>12} {"CV":>8}')
K_arr=np.array(Ks); I_arr=np.array(Is)
print(f'{"K_int":>10} {K_arr[0]:>12.0f} {K_arr[1]:>12.0f} {K_arr[2]:>12.0f} {K_arr.mean():>12.0f} {K_arr.std()/K_arr.mean()*100:>7.1f}%')
print(f'{"I (智能)":>10} {I_arr[0]:>12.4f} {I_arr[1]:>12.4f} {I_arr[2]:>12.4f} {I_arr.mean():>12.4f} {I_arr.std()/(abs(I_arr.mean())+1e-9)*100:>7.1f}%')
print(f'\n=> I 的 CV {"< K_int 的 CV (I 更稳定)" if I_arr.std()/(abs(I_arr.mean())+1e-9) < K_arr.std()/K_arr.mean() else "> K_int (I 不更稳)"}')
