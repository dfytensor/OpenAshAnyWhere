"""颠覆性验证: I_v2 能否在训练早期预测 val ppl 走向(不用 val 集)?
在真实 LM (minimind pretrain) 上训练 FRSMASH, 记录 I_v2=(rank+V)/(H_env*ln2),
看 I_v2 的变化是否领先于 val ppl 的变化.

关键测试: 用 I_v2 的导数(变化率)做 early-stopping 信号, 对比 val-ppl early-stopping.
如果 I_v2 能提前预测 val ppl 的转折点 => 颠覆(纯训练侧预测泛化).
"""
import torch, torch.nn.functional as F, math, os, sys, numpy as np, time
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
sys.path.insert(0, r'F:\rwkv\frsmash_v36'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frsmash_v36 import FRSMASHv36
from frsmash_directadd import FRSMASHDirectAdd
DEV='cuda'; VOCAB=23005
CACHE=os.environ.get('KINT_CACHE', r'F:\OpenASH2605\train_60m\cache\pt_cache_openash_512_openash.pt')

class DS(Dataset):
    def __init__(s,d,se): s.d,s.se=d,se
    def __len__(s): return len(s.d)
    def __getitem__(s,i): return s.d[i][:s.se+1]
    @staticmethod
    def collate(it): p=pad_sequence(it,batch_first=True,padding_value=0); return p[:,:-1],p[:,1:]

@torch.no_grad()
def mean_stable_rank(m):
    rs=[]
    for p in m.parameters():
        if p.dim()==2 and min(p.shape)>1:
            W=p.detach().float(); sv=torch.linalg.svdvals(W)
            fro=(sv**2).sum().item(); spec=sv[0].item()**2
            if spec>1e-12: rs.append(fro/spec)
    return float(np.mean(rs)) if rs else 1.0

def run(steps=2000, seq=512, batch=32, lr=5e-4, wd=0.01, n_train=5000, n_val=2000):
    """小训练集制造过拟合 => val ppl 会先降后升, 看 I_v2 能否提前预警."""
    torch.manual_seed(0)
    data=torch.load(CACHE,weights_only=False); val_d,tr_d=data[:n_val],data[n_val:n_val+n_train]
    model=FRSMASHDirectAdd(VOCAB,512,8,8,4).to(DEV)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd,betas=(0.9,0.95))
    scaler=torch.amp.GradScaler()
    tr=DataLoader(DS(tr_d,seq),batch_size=batch,shuffle=True,collate_fn=DS.collate,drop_last=True)
    vl=DataLoader(DS(val_d,seq),batch_size=16,shuffle=False,collate_fn=DS.collate)
    # H_env 估计: 训练数据的经验熵率 (用 unigram 频率)
    from collections import Counter
    counts=Counter()
    for s in tr_d[:1000]:
        for t in s[s!=0]: counts[t.item()]+=1
    total=sum(counts.values())
    H_env=-sum((c/total)*math.log(c/total) for c in counts.values())  # nats
    LN2=math.log(2)
    print(f'H_env(empirical unigram)={H_env:.3f} nats',flush=True)

    log=[]
    ti=iter(tr); t0=time.time()
    for st in range(1,steps+1):
        # train
        try: x,y=next(ti)
        except StopIteration: ti=iter(tr); x,y=next(ti)
        x=x.clamp(0,VOCAB-1).to(DEV); y=y.clamp(0,VOCAB-1).to(DEV)
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            o=model(x); loss=F.cross_entropy(o.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update()

        # 每 25 步记录: train_loss + stable_rank + I_v2 + val_loss
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
            rank=mean_stable_rank(model)
            I_v2=(rank+train_V)/(H_env*LN2)
            log.append((st, train_V, rank, I_v2, val_loss, val_ppl))
            model.train()
            if st%200==0:
                print(f'  st{st}: trV={train_V:.3f} rank={rank:.1f} I_v2={I_v2:.3f} '
                      f'val_ppl={val_ppl:.1f} ({time.time()-t0:.0f}s)',flush=True)

    # 分析: I_v2 能否预测 val 转折点
    log=np.array(log)
    steps_arr=log[:,0]; trV=log[:,1]; rank=log[:,2]; Iv2=log[:,3]; vloss=log[:,4]; vppl=log[:,5]

    # val 最佳步 (val ppl 最小 = 过拟合起点)
    val_best_idx=np.argmin(vppl)
    val_best_step=steps_arr[val_best_idx]

    # I_v2 最小步 (如果 I_v2 最低点 < val 最低点 => I_v2 提前预警)
    Iv2_best_idx=np.argmin(Iv2)
    Iv2_best_step=steps_arr[Iv2_best_idx]

    # I_v2 变化率(dI/dt) 过零点 (I_v2 开始上升 = 可能过拟合信号)
    dI=np.diff(Iv2)
    # 找第一个 dI>0 持续 3 步的位置
    early_stop_step=None
    for i in range(3,len(dI)):
        if all(d>0 for d in dI[i-3:i]):
            early_stop_step=steps_arr[i]
            break

    print(f'\n=== 预测能力分析 ===')
    print(f'  val ppl 最低点(过拟合起点): step {int(val_best_step)} (ppl={vppl[val_best_idx]:.1f})')
    print(f'  I_v2 最低点:               step {int(Iv2_best_step)} (I_v2={Iv2[Iv2_best_idx]:.3f})')
    print(f'  I_v2 变化率首次持续上升:    step {early_stop_step}')
    lead = val_best_step - (early_stop_step or Iv2_best_step)
    print(f'  => I_v2 预警 {"领先" if lead>0 else "滞后"} {abs(lead)} 步')
    print(f'  => {"YES: I_v2 能提前预测 val 过拟合!" if lead>0 else "NO: I_v2 不能提前预测"}')

    # 互相关: I_v2 与 val_ppl 的 lag
    from numpy import correlate
    def ncc(x,y):
        x=(x-x.mean())/(x.std()+1e-9); y=(y-y.mean())/(y.std()+1e-9)
        return correlate(x,y,'full')
    cc=ncc(Iv2, vppl)
    best_lag=np.argmax(np.abs(cc))-len(Iv2)//2
    lag_steps=best_lag*25
    print(f'  互相关峰值 lag={best_lag} ({lag_steps:+d}步): '
          f'{"I_v2 领先 val" if best_lag<0 else "I_v2 滞后 val" if best_lag>0 else "同步"}')

    # 保存数据
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),'log_iq_predict.npy'), log)
    print(f'\n  数据已存 log_iq_predict.npy')
    return log, val_best_step, early_stop_step

if __name__=='__main__':
    run()
