"""闭环思路(CLR-wd) 应用到 FRSMASH v3.6 和 GLA 骨干: 能否提升 val ppl?
CLR: λ(t)=λ_max·compress(V_train)·gap_signal(val-train), AdamW wd 每步动态设(解耦).
对照: fixed wd. 同数据/步数, 比 val ppl.
"""
import torch, torch.nn.functional as F, math, os, sys, csv, time, argparse
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

DEV='cuda'; OUT=os.path.dirname(os.path.abspath(__file__))
CACHE=os.environ.get('KINT_CACHE', r'F:\OpenASH2605\train_60m\cache\pt_cache_openash_512_openash.pt')
VOCAB=23005

class DS(Dataset):
    def __init__(s,d,se): s.d,s.se=d,se
    def __len__(s): return len(s.d)
    def __getitem__(s,i): return s.d[i][:s.se+1]
    @staticmethod
    def collate(it):
        p=pad_sequence(it,batch_first=True,padding_value=0); return p[:,:-1],p[:,1:]

def get_data(seq, n_val=4000):
    data=torch.load(CACHE,weights_only=False); return data[:n_val], data[n_val:300000]

def run(backbone, cond, steps, seq, batch, lr, wd, seed=0,
        vfrac=0.7, tau=0.15, gap_margin=0.02, gap_tau=0.03):
    torch.manual_seed(seed)
    val_d,tr_d=get_data(seq)
    if backbone=='frsmash':
        sys.path.insert(0, r'F:\rwkv\frsmash_v36')
        from frsmash_v36 import FRSMASHv36
        model=FRSMASHv36(VOCAB,432,8,8,n_slots=4).to(DEV)
    else:
        sys.path.insert(0, OUT)
        from arch_compare import LM
        model=LM(VOCAB,d=256,h=8,L=6,attn_t='gla',ffn_t='dense',d_ffn=1024,max_len=seq+8).to(DEV)
    n=sum(p.numel() for p in model.parameters())
    print(f'[{backbone}/{cond}] params={n:,}',flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.0)
    tr=DataLoader(DS(tr_d,seq),batch_size=batch,shuffle=True,num_workers=0,collate_fn=DS.collate,drop_last=True)
    vloader=DataLoader(DS(val_d,seq),batch_size=batch,shuffle=False,num_workers=0,collate_fn=DS.collate)
    warm=int(0.05*steps)
    model.eval()
    with torch.no_grad():
        xb,yb=next(iter(tr))
        if backbone=='frsmash': V0=F.cross_entropy(model(xb.to(DEV)).reshape(-1,VOCAB),yb.to(DEV).reshape(-1),ignore_index=0).item()
        else:
            lo,_=model(xb.to(DEV)); V0=F.cross_entropy(lo.reshape(-1,VOCAB),yb.to(DEV).reshape(-1),ignore_index=0).item()
    def vmet():
        model.eval();t=0.0;c=0
        with torch.no_grad():
            for x,y in vloader:
                x,y=x.to(DEV),y.to(DEV)
                if backbone=='frsmash': lo=model(x)
                else: lo,_=model(x)
                l=F.cross_entropy(lo.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0,reduction='sum');t+=float(l);c+=int((y!=0).sum())
        model.train(); return t/c
    val_ema=V0; csv_p=os.path.join(OUT,f'log_clrapply_{backbone}_{cond}.csv')
    open(csv_p,'w').write('step,train_loss,val_loss,val_ppl,gap,lambda\n')
    t0=time.time()
    for st in range(1,steps+1):
        cur_lr=lr*min(1.0,st/warm)
        for g in opt.param_groups: g['lr']=cur_lr
        x,y=next(iter(tr));x,y=x.to(DEV),y.to(DEV)
        if backbone=='frsmash': lo=model(x)
        else: lo,_=model(x)
        V=F.cross_entropy(lo.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0)
        if cond=='clr':
            if st%30==0:
                vl_=vmet(); val_ema=0.8*val_ema+0.2*vl_
            gap=max(0.0,val_ema-float(V.detach()))
            compress=float(torch.sigmoid(torch.tensor((V0*vfrac-float(V.detach()))/tau)))
            gsig=float(torch.sigmoid(torch.tensor((gap-gap_margin)/gap_tau)))
            lam=wd*compress*gsig
        else:
            lam=wd
        for g in opt.param_groups: g['weight_decay']=lam
        opt.zero_grad(); V.backward(); opt.step()
        if st%100==0:
            vloss=vmet()
            with open(csv_p,'a') as f: f.write(f'{st},{float(V):.4f},{vloss:.4f},{math.exp(vloss):.2f},{vloss-float(V):+.3f},{lam:.4f}\n')
            if st%300==0 or st<=200: print(f'  [{backbone}/{cond}] s{st} tr={float(V):.3f} val_ppl={math.exp(vloss):.2f} gap={vloss-float(V):+.3f} lam={lam:.4f}',flush=True)
    vloss=vmet(); print(f'[{backbone}/{cond}] DONE val_ppl={math.exp(vloss):.2f} ({time.time()-t0:.0f}s)\n',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--backbone',default='frsmash')
    ap.add_argument('--cond',default='clr'); ap.add_argument('--steps',type=int,default=1500)
    ap.add_argument('--seq',type=int,default=512); ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--lr',type=float,default=5e-4); ap.add_argument('--wd',type=float,default=0.01)
    a=ap.parse_args()
    run(a.backbone,a.cond,a.steps,a.seq,a.batch,a.lr,a.wd)
