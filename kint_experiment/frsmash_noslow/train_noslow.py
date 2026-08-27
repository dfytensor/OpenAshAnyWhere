"""vanilla FRSMASH vs NoSlow(砍 SlowMemory): seq512 + seq4096, 验 ppl 不变=死代码."""
import torch, torch.nn.functional as F, math, os, sys, time, argparse
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
DEV='cuda'; OUT=os.path.dirname(os.path.abspath(__file__))
CACHE=os.environ.get('KINT_CACHE', r'F:\OpenASH2605\train_60m\cache\pt_cache_openash_512_openash.pt')
VOCAB=23005
sys.path.insert(0, r'F:\rwkv\frsmash_v36'); sys.path.insert(0, OUT)
from frsmash_v36 import FRSMASHv36
from frsmash_noslow import FRSMASHNoSlow

class DS(Dataset):
    def __init__(s,d,se): s.d,s.se=d,se
    def __len__(s): return len(s.d)
    def __getitem__(s,i): return s.d[i][:s.se+1]
    @staticmethod
    def collate(it): p=pad_sequence(it,batch_first=True,padding_value=0); return p[:,:-1],p[:,1:]

def fwd(m,x):
    o=m(x); return o[0] if isinstance(o,tuple) else o

def run(name, model, steps=1500, seq=512, batch=32, lr=5e-4, wd=0.01, n_val=3000, accum=1):
    data=torch.load(CACHE,weights_only=False); val_d,tr_d=data[:n_val],data[n_val:]
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd,betas=(0.9,0.95))
    scaler=torch.amp.GradScaler()
    micro=max(1, batch//accum)
    tr=DataLoader(DS(tr_d,seq),batch_size=micro,shuffle=True,num_workers=0,collate_fn=DS.collate,drop_last=True,pin_memory=True)
    vl=DataLoader(DS(val_d,seq),batch_size=16,shuffle=False,num_workers=0,collate_fn=DS.collate)
    def vppl():
        model.eval();t=0.0;c=0
        with torch.no_grad():
            for x,y in vl:
                x,y=x.to(DEV),y.to(DEV)
                with torch.amp.autocast('cuda',dtype=torch.bfloat16): o=fwd(model,x)
                l=F.cross_entropy(o.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0,reduction='sum');t+=float(l);c+=int((y!=0).sum())
        model.train(); return math.exp(t/c)
    print(f'[{name}] params={sum(p.numel() for p in model.parameters()):,} seq={seq}',flush=True)
    t0=time.time(); ti=iter(tr)
    for st in range(1,steps+1):
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            try: x,y=next(ti)
            except StopIteration: ti=iter(tr); x,y=next(ti)
            x=x.clamp(0,VOCAB-1).to(DEV); y=y.clamp(0,VOCAB-1).to(DEV)
            with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                o=fwd(model,x); loss=F.cross_entropy(o.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0)/accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update()
        if st%500==0: print(f'  [{name}] s{st} ({time.time()-t0:.0f}s)',flush=True)
    vp=vppl(); print(f'[{name}] DONE seq={seq} val_ppl={vp:.2f} ({time.time()-t0:.0f}s)\n',flush=True)
    return vp

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--cond',default='noslow',choices=['vanilla','noslow'])
    ap.add_argument('--steps',type=int,default=1500); ap.add_argument('--seq',type=int,default=512)
    ap.add_argument('--accum',type=int,default=1)
    a=ap.parse_args()
    m=FRSMASHv36(VOCAB,512,8,8,4).to(DEV) if a.cond=='vanilla' else FRSMASHNoSlow(VOCAB,512,8,8,4).to(DEV)
    run(a.cond, m, a.steps, a.seq, accum=a.accum)
