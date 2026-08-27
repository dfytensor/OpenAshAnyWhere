"""Loop 8 最后一里: 真实长上下文验证 hybrid 是否用上长程 + 速度优势.
训练 softmax/GLA/hybrid 于拼接的 seq=1024 连续文本流, 测:
  ppl_full(看全1024) vs ppl_trunc(只看末256) -> Δppl = 长程收益(越大越会用长上下文)
对比三架构 Δppl, 并复述速度(loop7 已得 hybrid 7.8x).
注: 完整 NIAH/LongBench 需更大指令微调模型, 超本算力, 列为 future work.
"""
import torch, torch.nn.functional as F, math, os, sys, csv, time, argparse
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
DEV='cuda'; OUT=os.path.dirname(os.path.abspath(__file__))
CACHE=os.environ.get('KINT_CACHE', r'F:\OpenASH2605\train_60m\cache\pt_cache_openash_512_openash.pt')
sys.path.insert(0, OUT)
from hybrid_attn import LM, VOCAB

SEQ=1024; TRUNC=256

class DS(Dataset):
    def __init__(s,d): s.d=d
    def __len__(s): return len(s.d)
    def __getitem__(s,i): return s.d[i]
    @staticmethod
    def collate(it): p=pad_sequence(it,batch_first=True,padding_value=0); return p[:,:-1],p[:,1:]

def make_streams(seq, n_val=512):
    data=torch.load(CACHE,weights_only=False)
    flat=torch.cat([t[t!=0] for t in data])
    n=(flat.size(0)-1)//(seq+1)
    chunks=[flat[i*(seq+1):(i+1)*(seq+1)] for i in range(n)]
    return chunks[:n_val], chunks[n_val:n_val+40000]

@torch.no_grad()
def ppl_on(model, loader):
    t=0.0;c=0
    for x,y in loader:
        x,y=x.to(DEV),y.to(DEV); lo,_=model(x)
        l=F.cross_entropy(lo.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0,reduction='sum');t+=float(l);c+=int((y!=0).sum())
    return t/c

def run(attn_t, steps=1000, batch=8, lr=3e-4, wd=0.1):
    torch.manual_seed(0)
    val_c, tr_c = make_streams(SEQ)
    val_full=DS(val_c)
    # trunc val: 只取每样本末 TRUNC
    val_trc=DS([c[-(TRUNC+1):] for c in val_c])
    vlf=DataLoader(val_full,batch_size=8,shuffle=False,collate_fn=DS.collate)
    vlt=DataLoader(val_trc,batch_size=8,shuffle=False,collate_fn=DS.collate)
    model=LM(attn_t,d=256,h=8,L=6,W=128,max_len=SEQ+8).to(DEV)
    print(f'[{attn_t}] params={sum(p.numel() for p in model.parameters()):,}',flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    tr=DataLoader(DS(tr_c),batch_size=batch,shuffle=True,collate_fn=DS.collate,drop_last=True)
    warm=int(0.05*steps); t0=time.time()
    for st in range(1,steps+1):
        for g in opt.param_groups: g['lr']=lr*min(1.0,st/warm)
        x,y=next(iter(tr)); x,y=x.to(DEV),y.to(DEV)
        lo,_=model(x); loss=F.cross_entropy(lo.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
        if st%300==0: print(f'  [{attn_t}] s{st} loss={float(loss):.3f}',flush=True)
    pf=ppl_on(model,vlf); pt=ppl_on(model,vlt); dt=time.time()-t0
    print(f'[{attn_t}] DONE ppl_full(1024)={math.exp(pf):.2f} ppl_trunc(256)={math.exp(pt):.2f} Δppl={math.exp(pt)-math.exp(pf):+.2f} ({dt:.0f}s)\n',flush=True)
    return math.exp(pf), math.exp(pt)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--attn',default='hybrid',choices=['softmax','gla','hybrid'])
    ap.add_argument('--steps',type=int,default=1000); ap.add_argument('--batch',type=int,default=8)
    a=ap.parse_args()
    run(a.attn, a.steps, a.batch)
