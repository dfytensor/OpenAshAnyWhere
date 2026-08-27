"""SGR-v2: Rank-Routed FFN, 目标全面赢过 GLA(dense).
关键修正(对比旧 SGR):
  1. 骑在 GLA 注意力上(整体 O(T), 不再用 softmax)
  2. 硬路由条件计算: 每 token 按"惊讶度"(注意力输出范数) top-k 路由到 全秩/低秩 专家
     —— 只算被选中的专家, 真正省 FLOPs(旧 SGR 两条都算=更贵)
  3. 固定比例(top-k)路由 => 不退化(旧 SGR 门控饱和到0.93)
FLOPs: frac_full * full + (1-frac_full) * low; frac=0.3 => ~0.4x dense
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, os, sys, csv, time, argparse
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch_compare import GlaAttn, VOCAB

DEV='cuda'; OUT=os.path.dirname(os.path.abspath(__file__))
CACHE=os.environ.get('KINT_CACHE', r'F:\OpenASH2605\train_60m\cache\pt_cache_openash_512_openash.pt')


class RankRoutedFFN(nn.Module):
    """硬路由: top-k(按 surprise) 走全秩, 其余走低秩. 只算被选专家."""
    def __init__(self, d, d_ffn_full=1024, d_ffn_low=128, frac_full=0.3):
        super().__init__(); self.frac_full=frac_full; self.act=nn.SiLU()
        self.fu=nn.Linear(d,d_ffn_full,False); self.fd=nn.Linear(d_ffn_full,d,False)
        self.lu=nn.Linear(d,d_ffn_low,False);  self.ld=nn.Linear(d_ffn_low,d,False)
    def forward(self, x, surprise):              # x:(B,T,d), surprise:(B,T)
        B,T,d=x.shape; k=max(1,int(round(T*self.frac_full)))
        _,idx=surprise.topk(k,dim=1)
        mask=torch.zeros(B,T,dtype=torch.bool,device=x.device); mask.scatter_(1,idx,True)
        out=torch.empty_like(x)
        mf=mask
        if mf.any():    out[mf]=self.fd(self.act(self.fu(x[mf])))
        if (~mf).any(): out[~mf]=self.ld(self.act(self.lu(x[~mf])))
        return out

class SharedRRF(nn.Module):
    """SGR-v3: 共享专家(每token都走, 廉价保底) + 条件全秩专家(仅 top-k 困难 token 叠加).
    DeepSeekMoE 式: 易 token=shared, 难 token=shared+full. 质量/算力折中更好."""
    def __init__(self, d, d_ffn_full=1024, d_shared=256, frac_full=0.3):
        super().__init__(); self.frac_full=frac_full; self.act=nn.SiLU()
        self.su=nn.Linear(d,d_shared,False); self.sd=nn.Linear(d_shared,d,False)   # 共享(每token)
        self.fu=nn.Linear(d,d_ffn_full,False); self.fd=nn.Linear(d_ffn_full,d,False)  # 全秩(困难token)
    def forward(self, x, surprise):
        B,T,d=x.shape
        out=self.sd(self.act(self.su(x)))            # 每个token的共享底座
        k=max(1,int(round(T*self.frac_full)))
        _,idx=surprise.topk(k,dim=1)
        mask=torch.zeros(B,T,dtype=torch.bool,device=x.device); mask.scatter_(1,idx,True)
        if mask.any():
            out[mask]=out[mask]+self.fd(self.act(self.fu(x[mask])))   # 困难token叠加全秩
        return out

class DenseFFN(nn.Module):
    def __init__(self,d,d_ffn=1024): super().__init__(); self.u=nn.Linear(d,d_ffn,False); self.dn=nn.Linear(d_ffn,d,False); self.a=nn.SiLU()
    def forward(self,x,*a): return self.dn(self.a(self.u(x)))

class Block(nn.Module):
    def __init__(self, d, h, ffn, d_ffn_full, d_ffn_low, frac_full):
        super().__init__(); self.n1=nn.LayerNorm(d); self.n2=nn.LayerNorm(d)
        self.attn=GlaAttn(d,h)
        self.ffn = RankRoutedFFN(d,d_ffn_full,d_ffn_low,frac_full) if ffn=='rrf' else (SharedRRF(d,d_ffn_full,d_ffn_low,frac_full) if ffn=='srrf' else DenseFFN(d,d_ffn_full))
        self.is_rrf = (ffn in ('rrf','srrf'))
    def forward(self,x):
        h=self.n1(x); ao=self.attn(h); x=x+ao
        x2=self.n2(x)
        if self.is_rrf:
            surprise=ao.norm(dim=-1)             # 注意力输出范数 = 惊讶度(免费, 不退化)
            return x+self.ffn(x2,surprise)
        return x+self.ffn(x2)

class LM(nn.Module):
    def __init__(self, ffn, d=256,h=8,L=6,d_ffn_full=1024,d_ffn_low=128,frac_full=0.3,max_len=600,vocab=None,tied=False):
        super().__init__(); V=vocab or VOCAB
        self.em=nn.Embedding(V,d); self.pos=nn.Embedding(max_len,d)
        self.blocks=nn.ModuleList([Block(d,h,ffn,d_ffn_full,d_ffn_low,frac_full) for _ in range(L)])
        self.norm=nn.LayerNorm(d); self.tied=tied
        if not tied: self.head=nn.Linear(d,V,bias=False)
    def forward(self,x):
        h=self.em(x)+self.pos(torch.arange(x.size(1),device=x.device))
        for b in self.blocks: h=b(h)
        h=self.norm(h)
        return F.linear(h,self.em.weight) if self.tied else self.head(h)

class DS(Dataset):
    def __init__(s,d,se): s.d,s.se=d,se
    def __len__(s): return len(s.d)
    def __getitem__(s,i): return s.d[i][:s.se+1]
    @staticmethod
    def collate(it): p=pad_sequence(it,batch_first=True,padding_value=0); return p[:,:-1],p[:,1:]

def run(name, ffn, steps=1000, seq=256, batch=32, lr=3e-4, wd=0.1, n_val=3000, frac_full=0.3, seed=0,
        anneal=False, target_frac=0.4, a_start=0.3, a_end=0.8, closed=False, **kw):
    torch.manual_seed(seed)
    data=torch.load(CACHE,weights_only=False); val_d,tr_d=data[:n_val],data[n_val:]
    model=LM(ffn,frac_full=frac_full,**kw).to(DEV)
    n=sum(p.numel() for p in model.parameters())
    print(f'[{name}] ffn={ffn} params={n:,} frac_full={frac_full} anneal={anneal} target={target_frac} closed={closed}',flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    tr=DataLoader(DS(tr_d,seq),batch_size=batch,shuffle=True,num_workers=0,collate_fn=DS.collate,drop_last=True)
    vl=DataLoader(DS(val_d,seq),batch_size=batch,shuffle=False,num_workers=0,collate_fn=DS.collate)
    warm=int(0.05*steps)
    def vmet():
        model.eval();t=0.0;c=0;cor=0;nt=0
        with torch.no_grad():
            for x,y in vl:
                x,y=x.to(DEV),y.to(DEV); lo=model(x)
                l=F.cross_entropy(lo.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0,reduction='sum');t+=float(l);c+=int((y!=0).sum())
                p=lo.reshape(-1,VOCAB).argmax(-1);yt=y.reshape(-1);m=yt!=0;cor+=int((p[m]==yt[m]).sum());nt+=int(m.sum())
        model.train(); return t/c,math.exp(t/c),cor/max(nt,1)
    is_routed = ffn in ('rrf','srrf')
    def set_frac(f):
        for blk in model.blocks: blk.ffn.frac_full=f
    val_ema=None; cur_frac=1.0 if (anneal or closed) else frac_full
    if is_routed: set_frac(cur_frac)
    for st in range(1,steps+1):
        for g in opt.param_groups: g['lr']=lr*min(1.0,st/warm)
        # 路由退火 / 闭环 (K-Net "先拟合再压缩" 原理)
        if is_routed and (anneal or closed):
            if closed:
                if st%50==0:
                    ll,_,_=vmet()
                    if val_ema is None:
                        val_ema=ll
                    else:
                        if ll < val_ema:           # val 改善 -> 收紧路由(压复杂度)
                            cur_frac=max(target_frac, cur_frac-0.006)
                        elif ll > val_ema*1.002:   # val 退化 -> 放松回 dense
                            cur_frac=min(1.0, cur_frac+0.015)
                        val_ema=0.8*val_ema+0.2*ll
            elif anneal:
                if st<a_start*steps: cur_frac=1.0
                elif st>a_end*steps: cur_frac=target_frac
                else: cur_frac=1.0+(target_frac-1.0)*(st-a_start*steps)/((a_end-a_start)*steps)
            set_frac(cur_frac)
        x,y=next(iter(tr));x,y=x.to(DEV),y.to(DEV)
        lo=model(x);loss=F.cross_entropy(lo.reshape(-1,VOCAB),y.reshape(-1),ignore_index=0)
        opt.zero_grad();loss.backward();opt.step()
        if st%200==0:
            ll,pp,ac=vmet(); print(f'  [{name}] s{st} tr={float(loss):.3f} ppl={pp:.2f} top1={ac:.3f} frac={cur_frac:.2f}',flush=True)
    ll,pp,ac=vmet(); print(f'[{name}] DONE ppl={pp:.2f} top1={ac:.4f} end_frac={cur_frac:.2f}\n',flush=True)
    return pp

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--ffn',default='rrf')
    ap.add_argument('--frac',type=float,default=0.3); ap.add_argument('--low',type=int,default=128)
    ap.add_argument('--steps',type=int,default=1000); ap.add_argument('--full',type=int,default=1024)
    ap.add_argument('--d',type=int,default=256); ap.add_argument('--L',type=int,default=6); ap.add_argument('--h',type=int,default=8)
    ap.add_argument('--seq',type=int,default=256); ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--anneal',action='store_true'); ap.add_argument('--closed',action='store_true')
    ap.add_argument('--target',type=float,default=0.4)
    a=ap.parse_args()
    run(a.ffn+'_'+str(a.frac), a.ffn, steps=a.steps, frac_full=a.frac, d_ffn_low=a.low, d_ffn_full=a.full,
        d=a.d, L=a.L, h=a.h, seq=a.seq, batch=a.batch, anneal=a.anneal, closed=a.closed, target_frac=a.target)