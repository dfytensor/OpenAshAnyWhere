"""探索真实 LM 上对过拟合敏感的训练侧量, 替代 stable rank 做 I 的 C 分量.
候选: 梯度噪声scale / Hessian trace近似 / loss曲率 / 权重更新方差 / 梯度余弦.
测每个量在过拟合起点(step775)附近是否有明显信号, 选最好的做 I_v3.
"""
import torch, torch.nn.functional as F, math, os, sys, numpy as np, time
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
sys.path.insert(0, r'F:\rwkv\frsmash_v36'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frsmash_directadd import FRSMASHDirectAdd
DEV='cuda'; VOCAB=23005
CACHE=os.environ.get('KINT_CACHE', r'F:\OpenASH2605\train_60m\cache\pt_cache_openash_512_openash.pt')

class DS(Dataset):
    def __init__(s,d,se): s.d,s.se=d,se
    def __len__(s): return len(s.d)
    def __getitem__(s,i): return s.d[i][:s.se+1]
    @staticmethod
    def collate(it): p=pad_sequence(it,batch_first=True,padding_value=0); return p[:,:-1],p[:,1:]

def grad_noise_scale(model, tr_d, seq, n_samples=3):
    """梯度噪声 scale: 不同 batch 采样梯度的方差/均值²."""
    grads=[]
    dl=DataLoader(DS(tr_d,seq),batch_size=8,shuffle=True,collate_fn=DS.collate)
    ti=iter(dl)
    for _ in range(n_samples):
        try: x,y=next(ti)
        except StopIteration: ti=iter(dl); x,y=next(ti)
        x=x.clamp(0,VOCAB-1).to(DEV); y=y.clamp(0,VOCAB-1).to(DEV)
        model.train()
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            o=model(x); loss=F.cross_entropy(o.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0)
        model.zero_grad(set_to_none=True)
        loss.backward()
        g=torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        grads.append(g.float().detach())
        model.zero_grad(set_to_none=True)
    G=torch.stack(grads)
    mean_g=G.mean(0); var_g=G.var(0)
    ns=float(var_g.norm()/ (mean_g.norm()**2 + 1e-12))
    return ns

@torch.no_grad()
def weight_update_jitter(model, prev_state, curr_state):
    """权重更新的'抖动': 相邻更新的方向变化(余弦相似度的 1-cos)."""
    diffs=[]
    for k in prev_state:
        if prev_state[k].dim()>1:
            d_prev=curr_state[k]-prev_state[k]  # 本次更新
            d_curr=curr_state[k]-prev_state[k]  # placeholder
            diffs.append(float(d_prev.norm()))
    return float(np.std(diffs)/ (np.mean(diffs)+1e-12))

@torch.no_grad()
def loss_landscape_curvature(model, x, y, eps=1e-3):
    """loss 曲率近似: 在随机方向扰动后 loss 的二阶变化."""
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            base_loss=F.cross_entropy(model(x).reshape(-1,VOCAB),y.reshape(-1),ignore_index=0).item()
    # 随机方向扰动
    directions=[torch.randn_like(p)*eps for p in model.parameters()]
    saved=[p.data.clone() for p in model.parameters()]
    with torch.no_grad():
        for p,d in zip(model.parameters(),directions): p.data+=d
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            pert_loss=F.cross_entropy(model(x).reshape(-1,VOCAB),y.reshape(-1),ignore_index=0).item()
        for p,s in zip(model.parameters(),saved): p.data=s
    curvature=abs(pert_loss-base_loss)/ (eps**2+1e-12)
    return float(curvature)

@torch.no_grad()
def train_val_loss_gap_proxy(model, x, y):
    """train loss 本身就是 C 分量候选(最简单)."""
    with torch.amp.autocast('cuda',dtype=torch.bfloat16):
        o=model(x)
    return float(F.cross_entropy(o.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0).item())

def run(steps=2000, seq=512, batch=32, lr=5e-4, wd=0.01, n_train=5000, n_val=2000):
    torch.manual_seed(0)
    data=torch.load(CACHE,weights_only=False); val_d,tr_d=data[:n_val],data[n_val:n_val+n_train]
    model=FRSMASHDirectAdd(VOCAB,512,8,8,4).to(DEV)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd,betas=(0.9,0.95))
    scaler=torch.amp.GradScaler()
    tr=DataLoader(DS(tr_d,seq),batch_size=batch,shuffle=True,collate_fn=DS.collate,drop_last=True)
    vl=DataLoader(DS(val_d,seq),batch_size=16,shuffle=False,collate_fn=DS.collate)
    # H_env
    from collections import Counter
    counts=Counter()
    for s in tr_d[:1000]:
        for t in s[s!=0]: counts[t.item()]+=1
    total=sum(counts.values())
    H_env=-sum((c/total)*math.log(c/total) for c in counts.values())
    LN2=math.log(2)

    log=[]; ti=iter(tr); t0=time.time()
    for st in range(1,steps+1):
        try: x,y=next(ti)
        except StopIteration: ti=iter(tr); x,y=next(ti)
        x=x.clamp(0,VOCAB-1).to(DEV); y=y.clamp(0,VOCAB-1).to(DEV)
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            o=model(x); loss=F.cross_entropy(o.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update()

        if st%25==0:
            model.eval()
            # val
            vt=0.0; vc=0
            with torch.no_grad():
                for vx,vy in vl:
                    vx,vy=vx.to(DEV),vy.to(DEV)
                    with torch.amp.autocast('cuda',dtype=torch.bfloat16): vo=model(vx)
                    vl_=F.cross_entropy(vo.reshape(-1,VOCAB),vy.reshape(-1),ignore_index=0,reduction='sum')
                    vt+=float(vl_); vc+=int((vy!=0).sum())
            val_loss=vt/vc; val_ppl=math.exp(val_loss)
            train_V=float(loss.detach())
            # 候选 C 量 (每 50 步算一次, 太贵的)
            if st%50==0:
                gns=grad_noise_scale(model,tr_d,seq,n_samples=3)
                # 用当前训练 batch 算曲率
                xb,yb = x[:8].to(DEV), y[:8].to(DEV) if x.size(0)>=8 else (x.to(DEV),y.to(DEV))
                curv=loss_landscape_curvature(model,xb,yb,eps=1e-3)
            else:
                gns=float('nan'); curv=float('nan')
            # train-val gap (近似)
            gap=val_loss-train_V
            log.append((st, train_V, val_loss, val_ppl, gap, gns, curv))
            model.train()
            if st%200==0:
                print(f'  st{st}: trV={train_V:.3f} val_ppl={val_ppl:.1f} gap={gap:+.3f} '
                      f'gns={gns:.1f} curv={curv:.1f} ({time.time()-t0:.0f}s)',flush=True)

    log=np.array(log)
    # 只取有 gns/curv 的行(每50步)
    sub=log[~np.isnan(log[:,5])]
    steps_s=sub[:,0]; trV=sub[:,1]; vloss=sub[:,2]; vppl=sub[:,3]; gap=sub[:,4]; gns=sub[:,5]; curv=sub[:,6]

    # val 过拟合起点
    all_vppl=log[:,3]; val_best_step=log[np.argmin(all_vppl),0]

    # 每个候选量的"转折点"(局部最小值)
    def find_turning_point(arr, steps_arr, window=3):
        """找首次持续上升的点"""
        d=np.diff(arr)
        for i in range(window,len(d)):
            if all(d[j]>0 for j in range(i-window+1,i+1)):
                return steps_arr[i]
        return None

    print(f'\n=== 过拟合预测能力 ===')
    print(f'  val ppl 最低点: step {int(val_best_step)}')

    for name, arr in [('train_val_gap', gap), ('grad_noise', gns), ('curvature', curv)]:
        tp=find_turning_point(arr, steps_s)
        if tp:
            lead=tp-val_best_step
            tag='LEAD' if lead<0 else 'LAG'
            print(f'  {name:>15}: 转折 step {int(tp)}  ({tag} {abs(int(lead))}步)  '
                  f'{"PREDICTIVE" if lead<0 else "not predictive"}')
        else:
            print(f'  {name:>15}: 无转折点(单调)')

    # 额外: train-val gap 的变化率
    d_gap=np.diff(gap)
    gap_turn=find_turning_point(gap, steps_s)
    print(f'\n  train-val gap 从缩小→扩大的拐点 = 过拟合信号')
    if gap_turn:
        lead=gap_turn-val_best_step
        print(f'  gap 拐点 step {int(gap_turn)} vs val step {int(val_best_step)}: '
              f'{"领先" if lead<0 else "滞后"} {abs(int(lead))}步')

    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),'log_explore.npy'), log)
    return log

if __name__=='__main__':
    run()
