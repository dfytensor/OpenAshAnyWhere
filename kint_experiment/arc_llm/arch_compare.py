"""架构对比: 同一 LM 数据/指标下, 比较多种架构族 (参数尽量对齐, tied embedding).
attn: softmax | gla(gated linear, fla) ; ffn: dense | sgr | moe
报告 val ppl + next-token top1 + 参数量(含 non-embedding).
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, os, sys, csv, argparse, time
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fla.ops.gla import chunk_gla

DEV = 'cuda'; OUT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get('KINT_CACHE', r'F:\OpenASH2605\train_60m\cache\pt_cache_openash_512_openash.pt')
VOCAB = 23005


# ============ 注意力 ============
class SoftmaxAttn(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.h=h; self.hd=d//h
        self.q=nn.Linear(d,d,False); self.k=nn.Linear(d,d,False); self.v=nn.Linear(d,d,False); self.o=nn.Linear(d,d,False)
    def forward(self,x):
        B,T,d=x.shape; H,hd=self.h,self.hd
        q=self.q(x).view(B,T,H,hd).transpose(1,2); k=self.k(x).view(B,T,H,hd).transpose(1,2); v=self.v(x).view(B,T,H,hd).transpose(1,2)
        a=(q@k.transpose(-1,-2))/math.sqrt(hd)
        a=a+torch.triu(torch.full((T,T),float('-inf'),device=x.device),1)
        return self.o((F.softmax(a,-1)@v).transpose(1,2).reshape(B,T,d))

class GlaAttn(nn.Module):
    """门控线性注意力 (O(T), fla chunk_gla). 记忆式, 长序列友好."""
    def __init__(self, d, h, d_h=32):
        super().__init__(); self.h=h; self.d_h=d_h
        self.q=nn.Linear(d,h*d_h,False); self.k=nn.Linear(d,h*d_h,False)
        self.v=nn.Linear(d,h*d_h,False); self.g=nn.Linear(d,h*d_h,True); self.o=nn.Linear(h*d_h,d,False)
        nn.init.constant_(self.g.bias, 4.0)
    def forward(self,x):
        B,T,d=x.shape; H,K=self.h,self.d_h
        q=self.q(x).view(B,T,H,K); k=self.k(x).view(B,T,H,K); v=self.v(x).view(B,T,H,K)
        g=F.logsigmoid(self.g(x).float()).view(B,T,H,K)
        out,_st=chunk_gla(q,k,v,g); return self.o(out.view(B,T,H*K))


# ============ FFN ============
class DenseFFN(nn.Module):
    def __init__(self, d, d_ffn): super().__init__(); self.u=nn.Linear(d,d_ffn,False); self.dn=nn.Linear(d_ffn,d,False); self.a=nn.SiLU()
    def forward(self,x): return self.dn(self.a(self.u(x))), None

class SgrFFN(nn.Module):
    """全秩 + 低秩, 惊讶门控混合."""
    def __init__(self, d, d_ffn, r=64):
        super().__init__(); self.a=nn.SiLU()
        self.fu=nn.Linear(d,d_ffn,False); self.fd=nn.Linear(d_ffn,d,False)
        self.lu=nn.Linear(d,r,False); self.ld=nn.Linear(r,d,False)
        self.gate=nn.Linear(d,1,True); nn.init.constant_(self.gate.bias,0.0)
    def forward(self,x):
        full=self.fd(self.a(self.fu(x))); low=self.ld(self.lu(x))   # lu:d→r, ld:r→d
        s=torch.sigmoid(self.gate(x)); return s*full+(1-s)*low, float(s.mean().detach())

class MoeFFN(nn.Module):
    """soft-mixture FFN (全专家软加权, 向量化, 快). E 个小专家."""
    def __init__(self, d, d_ffn_e, e=4):
        super().__init__(); self.e=e
        self.Wu=nn.Parameter(torch.randn(e,d,d_ffn_e)*0.02); self.Wd=nn.Parameter(torch.randn(e,d_ffn_e,d)*0.02)
        self.gate=nn.Linear(d,e,False); self.a=nn.SiLU()
    def forward(self,x):
        B,T,d=x.shape; xf=x.reshape(B*T,d)
        w=F.softmax(self.gate(xf),-1)                       # (BT,E)
        outs=[]
        for ex in range(self.e):
            outs.append(self.a(xf@self.Wu[ex])@self.Wd[ex])  # (BT,d) 向量化
        outs=torch.stack(outs,-1)                            # (BT,d,E)
        out=(outs*w.unsqueeze(1)).sum(-1).view(B,T,d)
        return out, float(w.var().detach())  # 记录门控利用率(越高=越平均路由)


# ============ 模型 ============
class Block(nn.Module):
    def __init__(self, d, h, attn_t, ffn_t, d_ffn, r_low, moe_e):
        super().__init__(); self.n1=nn.LayerNorm(d); self.n2=nn.LayerNorm(d)
        self.attn = SoftmaxAttn(d,h) if attn_t=='softmax' else GlaAttn(d,h)
        if ffn_t=='dense': self.ffn=DenseFFN(d,d_ffn)
        elif ffn_t=='sgr': self.ffn=SgrFFN(d,d_ffn,r_low)
        else: self.ffn=MoeFFN(d,max(d_ffn//2,128),moe_e)
    def forward(self,x):
        x=x+self.attn(self.n1(x)); f,aux=self.ffn(self.n2(x)); x=x+f; return x,aux

class LM(nn.Module):
    def __init__(self, vocab, d, h, L, attn_t, ffn_t, d_ffn=1024, r_low=64, moe_e=4, max_len=520):
        super().__init__(); self.em=nn.Embedding(vocab,d); self.pos=nn.Embedding(max_len,d)
        self.blocks=nn.ModuleList([Block(d,h,attn_t,ffn_t,d_ffn,r_low,moe_e) for _ in range(L)])
        self.norm=nn.LayerNorm(d); self.head=nn.Linear(d,vocab,bias=False)   # untied (tied 训练慢/ppl 高)
    def forward(self,x):
        h=self.em(x)+self.pos(torch.arange(x.size(1),device=x.device)); auxs=[]
        for b in self.blocks: h,a=b(h); auxs.append(a)
        return self.head(self.norm(h)), auxs


# ============ 数据 / 训练 ============
class DS(Dataset):
    def __init__(s,d,se): s.d,s.se=d,se
    def __len__(s): return len(s.d)
    def __getitem__(s,i): return s.d[i][:s.se+1]
    @staticmethod
    def collate(items):
        p=pad_sequence(items,batch_first=True,padding_value=0); return p[:,:-1],p[:,1:]

def nparams(m): return sum(p.numel() for p in m.parameters())
def nNonEmb(m): return nparams(m) - m.em.weight.numel()

def run(name, attn_t, ffn_t, steps, seq, batch, lr, wd, n_val, L=6, d=256, h=8, d_ffn=1024, seed=0):
    torch.manual_seed(seed)
    data=torch.load(CACHE,weights_only=False); val_d,tr_d=data[:n_val],data[n_val:]
    model=LM(VOCAB,d,h,L,attn_t,ffn_t,d_ffn=d_ffn).to(DEV)
    nemb=nNonEmb(model)
    print(f'[{name}] attn={attn_t} ffn={ffn_t} params={nparams(model):,} non-emb={nemb:,}',flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    tr=DataLoader(DS(tr_d,seq),batch_size=batch,shuffle=True,num_workers=0,collate_fn=DS.collate,drop_last=True)
    vl=DataLoader(DS(val_d,seq),batch_size=batch,shuffle=False,num_workers=0,collate_fn=DS.collate)
    def vmet():
        model.eval(); t=0.0;c=0;cor=0;nt=0
        with torch.no_grad():
            for x,y in vl:
                x,y=x.to(DEV),y.to(DEV); lo,_=model(x)
                l=F.cross_entropy(lo.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0,reduction='sum'); t+=float(l);c+=int((y!=0).sum())
                p=lo.reshape(-1,VOCAB).argmax(-1); yt=y.reshape(-1); m=yt!=0; cor+=int((p[m]==yt[m]).sum()); nt+=int(m.sum())
        model.train(); return t/c,math.exp(t/c),cor/max(nt,1)
    csv=os.path.join(OUT,f'log_arch_{name}.csv')
    open(csv,'w').write('step,train_loss,val_loss,val_ppl,val_top1\n')
    t0=time.time()
    warm=int(0.05*steps)
    for st in range(1,steps+1):
        cur_lr=lr*min(1.0, st/warm)               # 线性 warmup 防 divergence
        for g in opt.param_groups: g['lr']=cur_lr
        x,y=next(iter(tr)); x,y=x.to(DEV),y.to(DEV)
        lo,auxs=model(x); loss=F.cross_entropy(lo.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
        if st%200==0 or st<=100:
            ll,pp,ac=vmet()
            with open(csv,'a') as f: f.write(f'{st},{float(loss):.4f},{ll:.4f},{pp:.2f},{ac:.4f}\n')
            if st%500==0 or st<=200: print(f'  [{name}] s{st} tr={float(loss):.3f} ppl={pp:.2f} top1={ac:.3f}',flush=True)
    ll,pp,ac=vmet(); print(f'[{name}] DONE ppl={pp:.2f} top1={ac:.4f} ({time.time()-t0:.0f}s)\n',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--arch',default='vanilla')
    ap.add_argument('--steps',type=int,default=2000); ap.add_argument('--seq',type=int,default=512)
    ap.add_argument('--batch',type=int,default=24); ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--wd',type=float,default=0.1); ap.add_argument('--n_val',type=int,default=3000)
    a=ap.parse_args()
    cfg=dict(steps=a.steps,seq=a.seq,batch=a.batch,lr=a.lr,wd=a.wd,n_val=a.n_val)
    ARCH={'vanilla':('softmax','dense'),'sgr':('softmax','sgr'),'moe':('softmax','moe'),
          'gla':('gla','dense'),'gla_sgr':('gla','sgr')}
    attn,ffn=ARCH[a.arch]; run(a.arch,attn,ffn,**cfg)
