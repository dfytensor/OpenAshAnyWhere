"""训练 FRSMASH-Hybrid vs vanilla FRSMASH(~80M), 比 val ppl + 速度."""
import torch, torch.nn.functional as F, math, os, sys, time, argparse
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
DEV='cuda'; OUT=os.path.dirname(os.path.abspath(__file__))
CACHE=os.environ.get('KINT_CACHE', r'F:\OpenASH2605\train_60m\cache\pt_cache_openash_512_openash.pt')
VOCAB=23005
sys.path.insert(0, OUT)
from frsmash_hybrid import FRSMASHHybrid
sys.path.insert(0, r'F:\rwkv\frsmash_v36')
from frsmash_v36 import FRSMASHv36

class DS(Dataset):
    def __init__(s,d,se): s.d,s.se=d,se
    def __len__(s): return len(s.d)
    def __getitem__(s,i): return s.d[i][:s.se+1]
    @staticmethod
    def collate(it): p=pad_sequence(it,batch_first=True,padding_value=0); return p[:,:-1],p[:,1:]

def run(name, model, steps=1500, seq=512, batch=32, lr=5e-4, wd=0.01, n_val=3000):
    data=torch.load(CACHE,weights_only=False); val_d,tr_d=data[:n_val],data[n_val:]
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd,betas=(0.9,0.95))
    scaler=torch.amp.GradScaler()
    tr=DataLoader(DS(tr_d,seq),batch_size=batch,shuffle=True,num_workers=0,collate_fn=DS.collate,drop_last=True,pin_memory=True)
    vl=DataLoader(DS(val_d,seq),batch_size=16,shuffle=False,num_workers=0,collate_fn=DS.collate)
    def vppl():
        model.eval();t=0.0;c=0
        with torch.no_grad():
            for x,y in vl:
                x,y=x.to(DEV),y.to(DEV)
                with torch.amp.autocast('cuda',dtype=torch.bfloat16): o=model(x)
                l=F.cross_entropy(o.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0,reduction='sum');t+=float(l);c+=int((y!=0).sum())
        model.train(); return t/c, math.exp(t/c)
    print(f'[{name}] params={sum(p.numel() for p in model.parameters()):,}',flush=True)
    t0=time.time(); rl=0.0
    for st in range(1,steps+1):
        x,y=next(iter(tr)); x=x.clamp(0,VOCAB-1).to(DEV); y=y.clamp(0,VOCAB-1).to(DEV)
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            o=model(x); loss=F.cross_entropy(o.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update()
        rl+=loss.item()
        if st%300==0:
            el=time.time()-t0; tps=st*batch*seq/el
            print(f'  [{name}] s{st} loss={rl/300:.4f} {tps:.0f}tok/s',flush=True); rl=0.0
    vl_l,vp=vppl(); print(f'[{name}] DONE val_ppl={vp:.2f} ({time.time()-t0:.0f}s)\n',flush=True)
    return vp

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--cond',default='hybrid',choices=['vanilla','hybrid'])
    ap.add_argument('--steps',type=int,default=1500); ap.add_argument('--d',type=int,default=512)
    ap.add_argument('--L',type=int,default=8)
    a=ap.parse_args()
    m = FRSMASHv36(VOCAB,a.d,8,a.L,4).to(DEV) if a.cond=='vanilla' else FRSMASHHybrid(VOCAB,a.d,8,a.L,4).to(DEV)
    run(a.cond, m, steps=a.steps)
