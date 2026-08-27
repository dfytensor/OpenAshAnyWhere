"""梯度累计 + 超长上下文(seq4096): 验 state 容量是否变瓶颈.
vanilla vs dense-StateMoE, seq4096, micro-batch + grad_accum.
"""
import torch, torch.nn.functional as F, math, os, sys, time, argparse
from torch.utils.data import Dataset, DataLoader
DEV='cuda'; OUT=os.path.dirname(os.path.abspath(__file__))
CACHE=os.environ.get('KINT_CACHE', r'F:\OpenASH2605\train_60m\cache\pt_cache_openash_512_openash.pt')
VOCAB=23005
sys.path.insert(0, r'F:\rwkv\frsmash_v36'); sys.path.insert(0, OUT)
from frsmash_v36 import FRSMASHv36
from frsmash_dense_statemoe import FRSMASHDenseStateMoE

class DS(Dataset):
    def __init__(s,d): s.d=d
    def __len__(s): return len(s.d)
    def __getitem__(s,i): return s.d[i]
    @staticmethod
    def collate(it):
        from torch.nn.utils.rnn import pad_sequence
        p=pad_sequence(it,batch_first=True,padding_value=0); return p[:,:-1],p[:,1:]

def make_streams(seq, n_val=64):
    data=torch.load(CACHE,weights_only=False)
    flat=torch.cat([t[t!=0] for t in data])
    n=(flat.size(0)-1)//(seq+1)
    chunks=[flat[i*(seq+1):(i+1)*(seq+1)] for i in range(n)]
    return chunks[:n_val], chunks[n_val:n_val+8000]

def fwd(model,x):
    o=model(x); return o[0] if isinstance(o,tuple) else o

@torch.no_grad()
def vppl(model, loader):
    t=0.0;c=0
    for x,y in loader:
        x,y=x.to(DEV),y.to(DEV)
        with torch.amp.autocast('cuda',dtype=torch.bfloat16): o=fwd(model,x)
        l=F.cross_entropy(o.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0,reduction='sum');t+=float(l);c+=int((y!=0).sum())
    return t/c, math.exp(t/c)

def run(name, model, steps=200, seq=4096, micro_bs=2, grad_accum=4, lr=5e-4, wd=0.01):
    val_c,tr_c=make_streams(seq)
    TRUNC=seq//4  # 截断 = seq 的 1/4, 做 Δppl 长程度量
    val_trc=[c[-(TRUNC+1):] for c in val_c]
    vlf=DataLoader(DS(val_c),batch_size=2,shuffle=False,collate_fn=DS.collate)
    vlt=DataLoader(DS(val_trc),batch_size=2,shuffle=False,collate_fn=DS.collate)
    tr=DataLoader(DS(tr_c),batch_size=micro_bs,shuffle=True,collate_fn=DS.collate,drop_last=True)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd,betas=(0.9,0.95))
    scaler=torch.amp.GradScaler()
    eff_bs=micro_bs*grad_accum
    print(f'[{name}] params={sum(p.numel() for p in model.parameters()):,} seq={seq} micro_bs={micro_bs} accum={grad_accum} eff_bs={eff_bs}',flush=True)
    t0=time.time(); rl=0.0; ti=iter(tr)
    for st in range(1,steps+1):
        opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            try: x,y=next(ti)
            except StopIteration: ti=iter(tr); x,y=next(ti)
            x=x.clamp(0,VOCAB-1).to(DEV); y=y.clamp(0,VOCAB-1).to(DEV)
            with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                o=fwd(model,x); loss=F.cross_entropy(o.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0)/grad_accum
            scaler.scale(loss).backward()
            rl+=loss.item()*grad_accum
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update()
        if st%50==0:
            el=time.time()-t0; print(f'  [{name}] s{st} loss={rl/(50*grad_accum):.4f} {st*eff_bs*seq/el:.0f}tok/s',flush=True); rl=0.0
    lf,pf=vppl(model,vlf); lt,pt=vppl(model,vlt)
    print(f'[{name}] DONE val_ppl(4096)={pf:.2f} val_ppl(trunc1024)={pt:.2f} Δppl={pt-pf:+.2f} ({time.time()-t0:.0f}s)\n',flush=True)
    return pf, pt

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--cond',default='dense',choices=['vanilla','dense'])
    ap.add_argument('--steps',type=int,default=200); ap.add_argument('--seq',type=int,default=4096)
    ap.add_argument('--micro',type=int,default=2); ap.add_argument('--accum',type=int,default=4)
    ap.add_argument('--d',type=int,default=512); ap.add_argument('--L',type=int,default=8)
    a=ap.parse_args()
    if a.cond=='vanilla': run('vanilla',FRSMASHv36(VOCAB,a.d,8,a.L,4).to(DEV),a.steps,a.seq,a.micro,a.accum)
    else: run('dense',FRSMASHDenseStateMoE(VOCAB,a.d,8,a.L,4,n_state_experts=10).to(DEV),a.steps,a.seq,a.micro,a.accum)
