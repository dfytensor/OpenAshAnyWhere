"""真 sparse top-k MoE + CLR: 验证 CLR 对 MoE 是否"决定性".
GLA 骨干 + 每 FFN 换成 top-k MoE(E 专家, 学习门控, 派发). k 可训练中调整.
对照:
  fixed  : k=2 从第0步固定(标准稀疏 MoE)
  clr    : k 从 E(dense, 无损) 退火/闭环 到 2 (CLR 阀门原则用到 MoE 路由稀疏度)
比 val ppl. 若 clr-MoE < fixed-MoE => CLR 对 MoE 训练有决定性帮助.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, os, sys, csv, time, argparse
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
DEV='cuda'; OUT=os.path.dirname(os.path.abspath(__file__))
CACHE=os.environ.get('KINT_CACHE', r'F:\OpenASH2605\train_60m\cache\pt_cache_openash_512_openash.pt')
sys.path.insert(0, OUT)
from arch_compare import GlaAttn, VOCAB


class SparseMoE(nn.Module):
    """top-k MoE: E 专家(各 d_e 宽), 学习门控, 派发. k 由外部 set_k 控制."""
    def __init__(self, d, d_e, E=8):
        super().__init__(); self.E=E; self.d_e=d_e; self.act=nn.SiLU()
        self.Wu=nn.Parameter(torch.randn(E,d,d_e)*0.02); self.Wd=nn.Parameter(torch.randn(E,d_e,d)*0.02)
        self.gate=nn.Linear(d,E,bias=False); self.k=E; self.aux=0.0
    def set_k(self,k): self.k=max(1,min(self.E,k))
    def forward(self,x):
        B,T,d=x.shape; xf=x.reshape(B*T,d)
        logits=self.gate(xf); k=self.k
        sc,idx=torch.topk(logits,k,dim=-1); w=F.softmax(sc,-1)            # (BT,k)
        out=torch.zeros(B*T,d,device=x.device,dtype=x.dtype)
        for e in range(self.E):
            m=(idx==e)                                                    # (BT,k) bool
            sel=m.any(-1)                                                 # (BT,) 该 token 是否用专家e
            if sel.any():
                wi=(w*m).sum(-1)[sel]                                     # 该 token 在专家e上的权重
                h=self.act(xf[sel]@self.Wu[e])@self.Wd[e]                 # (n,d)
                out[sel]+=wi.unsqueeze(-1)*h
        # 负载均衡 aux loss (标准 GShard)
        probs=F.softmax(logits,-1).mean(0); self.aux=self.E*(probs**2).sum()
        return out.view(B,T,d)


class DenseFFN(nn.Module):
    def __init__(s,d,d_e): super().__init__(); s.u=nn.Linear(d,d_e,False); s.dn=nn.Linear(d_e,d,False); s.a=nn.SiLU()
    def forward(s,x): return s.dn(s.a(s.u(x)))

class HetMoE(nn.Module):
    """秩异构 MoE = SGR(秩感知) 学 MoE(多专家专精). 专家有不同内秩; 学习门控 top-k.
    ranks=[512,512,128,128,...] => 2 全秩 + 6 低秩, 总参 < 等秩 MoE."""
    def __init__(self, d, ranks, k=2):
        super().__init__(); self.E=len(ranks); self.ranks=ranks; self.k=k; self.act=nn.SiLU()
        self.Wd=nn.ParameterList([nn.Parameter(torch.randn(d,r)*0.02) for r in ranks])
        self.Wu=nn.ParameterList([nn.Parameter(torch.randn(r,d)*0.02) for r in ranks])
        self.gate=nn.Linear(d,self.E,bias=False); self.aux=0.0
    def forward(self,x):
        B,T,d=x.shape; xf=x.reshape(B*T,d); k=self.k
        logits=self.gate(xf); sc,idx=torch.topk(logits,k,dim=-1); w=F.softmax(sc,-1)
        out=torch.zeros(B*T,d,device=x.device,dtype=x.dtype)
        for e in range(self.E):
            m=(idx==e).any(-1)
            if m.any():
                wi=(w*(idx==e)).sum(-1)[m]
                h=self.act(xf[m]@self.Wd[e])@self.Wu[e]
                out[m]+=wi.unsqueeze(-1)*h
        probs=F.softmax(logits,-1).mean(0); self.aux=self.E*(probs**2).sum()
        return out.view(B,T,d)

class Block(nn.Module):
    def __init__(self,d,h,ffn,d_e,E,ranks=None):
        super().__init__(); self.n1=nn.LayerNorm(d); self.n2=nn.LayerNorm(d); self.attn=GlaAttn(d,h)
        if ffn=='moe': self.ffn=SparseMoE(d,d_e,E)
        elif ffn=='hetmoe': self.ffn=HetMoE(d,ranks,k=2)
        else: self.ffn=DenseFFN(d,d_e)
    def forward(self,x):
        x=x+self.attn(self.n1(x)); x=x+self.ffn(self.n2(x)); return x

class LM(nn.Module):
    def __init__(self,ffn,d=256,h=8,L=6,d_e=512,E=8,ranks=None,max_len=600):
        super().__init__(); self.em=nn.Embedding(VOCAB,d); self.pos=nn.Embedding(max_len,d)
        self.blocks=nn.ModuleList([Block(d,h,ffn,d_e,E,ranks) for _ in range(L)])
        self.norm=nn.LayerNorm(d); self.head=nn.Linear(d,VOCAB,bias=False)
        self.is_moe=(ffn in ('moe','hetmoe'))
    def forward(self,x):
        h=self.em(x)+self.pos(torch.arange(x.size(1),device=x.device)); aux=0.0
        for b in self.blocks:
            h=b(h)
            if self.is_moe: aux=aux+b.ffn.aux
        return self.head(self.norm(h)), aux
    def set_k(self,k):
        for b in self.blocks:
            if hasattr(b.ffn,'set_k'): b.ffn.set_k(k)


class DS(Dataset):
    def __init__(s,d,se): s.d,s.se=d,se
    def __len__(s): return len(s.d)
    def __getitem__(s,i): return s.d[i][:s.se+1]
    @staticmethod
    def collate(it): p=pad_sequence(it,batch_first=True,padding_value=0); return p[:,:-1],p[:,1:]

def run(name,ffn,steps=1000,seq=256,batch=32,lr=3e-4,wd=0.1,n_val=3000,seed=0,
        mode='fixed',k_min=2,a_start=0.3,a_end=0.8,E=8):
    torch.manual_seed(seed)
    data=torch.load(CACHE,weights_only=False); val_d,tr_d=data[:n_val],data[n_val:]
    model=LM(ffn,d=256,h=8,L=6,d_e=512,E=E).to(DEV)
    n=sum(p.numel() for p in model.parameters())
    print(f'[{name}] ffn={ffn} params={n:,} mode={mode} E={E} k_min={k_min}',flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    tr=DataLoader(DS(tr_d,seq),batch_size=batch,shuffle=True,num_workers=0,collate_fn=DS.collate,drop_last=True)
    vloader=DataLoader(DS(val_d,seq),batch_size=batch,shuffle=False,num_workers=0,collate_fn=DS.collate)
    warm=int(0.05*steps)
    def vmet():
        model.eval();t=0.0;c=0
        with torch.no_grad():
            for x,y in vloader:
                x,y=x.to(DEV),y.to(DEV); lo,_=model(x)
                l=F.cross_entropy(lo.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0,reduction='sum');t+=float(l);c+=int((y!=0).sum())
        model.train(); return t/c
    cur_k = E if mode!='fixed' else k_min
    if model.is_moe: model.set_k(cur_k)
    csv_p=os.path.join(OUT,f'log_moe_{name}.csv'); open(csv_p,'w').write('step,train_loss,val_ppl,k\n')
    t0=time.time()
    for st in range(1,steps+1):
        for g in opt.param_groups: g['lr']=lr*min(1.0,st/warm)
        # k 调度
        if model.is_moe and mode!='fixed':
            if mode=='anneal':
                if st<a_start*steps: cur_k=E
                elif st>a_end*steps: cur_k=k_min
                else: cur_k=round(E+(k_min-E)*(st-a_start*steps)/((a_end-a_start)*steps))
            elif mode=='closed':
                if st%50==0:
                    ll=vmet()
                    if st>100 and ll<vmet_last: cur_k=max(k_min,cur_k-1)
                    vmet_last=ll
            model.set_k(cur_k)
        x,y=next(iter(tr));x,y=x.to(DEV),y.to(DEV)
        lo,aux=model(x); loss=F.cross_entropy(lo.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0)+0.01*aux
        opt.zero_grad(); loss.backward(); opt.step()
        if st%200==0:
            vloss=vmet()
            with open(csv_p,'a') as f: f.write(f'{st},{float(loss):.4f},{math.exp(vloss):.2f},{cur_k}\n')
            if st%400==0 or st<=200: print(f'  [{name}] s{st} tr={float(loss):.3f} val_ppl={math.exp(vloss):.2f} k={cur_k}',flush=True)
            vmet_last=vloss
    vloss=vmet(); print(f'[{name}] DONE val_ppl={math.exp(vloss):.2f} end_k={cur_k} ({time.time()-t0:.0f}s)\n',flush=True)

def run_het(name, ranks, steps=1000, seq=256, batch=32):
    torch.manual_seed(0)
    data=torch.load(CACHE,weights_only=False); val_d,tr_d=data[:3000],data[3000:300000]
    model=LM('hetmoe',d=256,h=8,L=6,ranks=ranks).to(DEV)
    n=sum(p.numel() for p in model.parameters())
    print(f'[{name}] hetmoe ranks={ranks} params={n:,}',flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.1)
    tr=DataLoader(DS(tr_d,seq),batch_size=batch,shuffle=True,num_workers=0,collate_fn=DS.collate,drop_last=True)
    vloader=DataLoader(DS(val_d,seq),batch_size=batch,shuffle=False,num_workers=0,collate_fn=DS.collate)
    warm=int(0.05*steps)
    def vmet():
        model.eval();t=0.0;c=0
        with torch.no_grad():
            for x,y in vloader:
                x,y=x.to(DEV),y.to(DEV); lo,_=model(x)
                l=F.cross_entropy(lo.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0,reduction='sum');t+=float(l);c+=int((y!=0).sum())
        model.train(); return t/c
    csv_p=os.path.join(OUT,f'log_moe_{name}.csv'); open(csv_p,'w').write('step,train_loss,val_ppl\n')
    for st in range(1,steps+1):
        for g in opt.param_groups: g['lr']=3e-4*min(1.0,st/warm)
        x,y=next(iter(tr));x,y=x.to(DEV),y.to(DEV)
        lo,aux=model(x); loss=F.cross_entropy(lo.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0)+0.01*aux
        opt.zero_grad(); loss.backward(); opt.step()
        if st%200==0:
            vloss=vmet()
            with open(csv_p,'a') as f: f.write(f'{st},{float(loss):.4f},{math.exp(vloss):.2f}\n')
            if st%400==0: print(f'  [{name}] s{st} tr={float(loss):.3f} val_ppl={math.exp(vloss):.2f}',flush=True)
    vloss=vmet(); print(f'[{name}] DONE val_ppl={math.exp(vloss):.2f}\n',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--cond',default='moe_fixed')
    ap.add_argument('--steps',type=int,default=1000); ap.add_argument('--seq',type=int,default=256)
    ap.add_argument('--batch',type=int,default=32); ap.add_argument('--E',type=int,default=8)
    a=ap.parse_args()
    if a.cond=='dense': run('dense','dense',steps=a.steps,seq=a.seq,batch=a.batch)
    elif a.cond=='moe_fixed': run('moe_fixed','moe',steps=a.steps,seq=a.seq,batch=a.batch,mode='fixed',k_min=2,E=a.E)
    elif a.cond=='moe_anneal': run('moe_anneal','moe',steps=a.steps,seq=a.seq,batch=a.batch,mode='anneal',k_min=2,E=a.E)
    elif a.cond=='hetmoe_2f6l': run_het('hetmoe_2f6l',[512,512,128,128,128,128,128,128],a.steps,a.seq,a.batch)  # 2全秩+6低秩
    elif a.cond=='hetmoe_4f4l': run_het('hetmoe_4f4l',[512,512,512,512,128,128,128,128],a.steps,a.seq,a.batch)  # 4全秩+4低秩
