"""第3轮: K-biomarker 在真实 LM 过拟合场景的通用性测试.
小模型 + 小训练集 + 无 wd => 过拟合(train↓ val↑). 看 K=C·V 能否领先预测 val 退化.
"""
import torch, torch.nn.functional as F, math, os, sys, csv, argparse
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
DEV='cuda'; OUT=os.path.dirname(os.path.abspath(__file__))
CACHE=os.environ.get('KINT_CACHE', r'F:\OpenASH2605\train_60m\cache\pt_cache_openash_512_openash.pt')
sys.path.insert(0, OUT)
from arch_compare import LM, VOCAB

class DS(Dataset):
    def __init__(s,d,se): s.d,s.se=d,se
    def __len__(s): return len(s.d)
    def __getitem__(s,i): return s.d[i][:s.se+1]
    @staticmethod
    def collate(it): p=pad_sequence(it,batch_first=True,padding_value=0); return p[:,:-1],p[:,1:]

@torch.no_grad()
def wnq(m):
    s=0.0
    for p in m.parameters(): s+=float((p.detach().float()**2).sum())
    return s

def run(n_train=2000, steps=2500, seq=256, batch=32, lr=3e-4, seed=0):
    torch.manual_seed(seed)
    data=torch.load(CACHE,weights_only=False)
    val_d,tr_d=data[:2000], data[2000:2000+n_train]   # 小训练集 => 过拟合
    model=LM(VOCAB,d=256,h=8,L=6,attn_t='gla',ffn_t='dense',d_ffn=1024,max_len=seq+8).to(DEV)
    print(f'params={sum(p.numel() for p in model.parameters()):,} train={n_train} wd=0 (overfit)',flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.0)   # 无 wd => 过拟合
    tr=DataLoader(DS(tr_d,seq),batch_size=batch,shuffle=True,num_workers=0,collate_fn=DS.collate,drop_last=True)
    vl=DataLoader(DS(val_d,seq),batch_size=32,shuffle=False,num_workers=0,collate_fn=DS.collate)
    csv_p=os.path.join(OUT,'log_kbiolm.csv'); open(csv_p,'w').write('step,V_train,C,K,V_val,val_ppl\n')
    def vloss():
        model.eval();t=0.0;c=0
        with torch.no_grad():
            for x,y in vl:
                x,y=x.to(DEV),y.to(DEV); lo,_=model(x)
                l=F.cross_entropy(lo.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0,reduction='sum');t+=float(l);c+=int((y!=0).sum())
        model.train(); return t/c
    for st in range(1,steps+1):
        x,y=next(iter(tr));x,y=x.to(DEV),y.to(DEV)
        lo,_=model(x); V=F.cross_entropy(lo.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0)
        opt.zero_grad(); V.backward(); opt.step()
        if st%25==0:
            C=wnq(model); vv=vloss()
            with open(csv_p,'a') as f: f.write(f'{st},{float(V):.4f},{C:.1f},{C*float(V):.1f},{vv:.4f},{math.exp(vv):.2f}\n')
            if st%500==0: print(f'  s{st} Vtr={float(V):.3f} Vval={vv:.3f}(ppl{math.exp(vv):.1f}) K={C*float(V):.0f}',flush=True)
    print('DONE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--n_train',type=int,default=2000)
    ap.add_argument('--steps',type=int,default=2500); ap.add_argument('--seed',type=int,default=0)
    a=ap.parse_args(); run(n_train=a.n_train, steps=a.steps, seed=a.seed)
