"""
FRSM v5: 两个新方向
1. Dual-Path: LM路径(全替换) + Memory路径(残差) → 门控混合
2. Selective Write: 输入决定写入强度, 重要信息全写, 噪声信息保留
对比基线: v1 Orig-4sc
"""
import os, sys, time, math, torch, random
import torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, 'F:/OpenASH2605')
from config import agent_voc_path
from open_ash_voc import OpenASHVoc
from frsm.dataset import PretrainDataset

device = torch.device("cuda")
voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1
VCF=32; ECF=0; ICF=1; HCF=128

class ScaleBlock(nn.Module):
    def __init__(s, d, exp=2.0):
        super().__init__(); hd=int(d*exp)
        s.ph=nn.Linear(d,hd,0); s.pi=nn.Linear(d,hd,0)
        s.Wf=nn.Linear(hd*2,hd); s.Wi=nn.Linear(hd*2,hd); s.Wc=nn.Linear(hd*2,hd)
        nn.init.constant_(s.Wf.bias,1.0); nn.init.constant_(s.Wi.bias,-2.0)
        s.po=nn.Linear(hd,d); s.n=nn.LayerNorm(d)
    def forward(s,h,i):
        hp=s.ph(h); ip=s.pi(i); c=torch.cat([hp,ip],-1)
        f=torch.sigmoid(s.Wf(c)); i_=torch.sigmoid(s.Wi(c))
        cand=s.po(f*hp+i_*torch.tanh(s.Wc(c)))
        return s.n(h+cand)

# ============================================================
# v1 Orig-4sc (基线)
# ============================================================
class V1(nn.Module):
    def __init__(s,vs,H=256,ns=4):
        super().__init__(); s.H=H; s.ns=ns
        s.e=nn.Embedding(vs,H); s.ii=nn.Linear(H,H); s.inm=nn.LayerNorm(H)
        s.sc=nn.ModuleList([ScaleBlock(H) for _ in range(ns)])
        s.sp=nn.ModuleList([nn.Linear(H,H) for _ in range(ns)])
        s.fu=nn.Linear(H*ns,H); s.fn=nn.LayerNorm(H); s.h=nn.Linear(H,vs)
    def forward(s,x,hp=None):
        B,T=x.shape
        if hp is None: h=[torch.zeros(B,s.H,device=x.device) for _ in range(s.ns)]
        else: h=[c.clone() for c in hp]
        xe=s.e(x); o=[]
        for t in range(T):
            inp=s.inm(s.ii(xe[:,t,:])); nh=[]
            for ss in range(s.ns):
                if t%(2**ss)==0: nh.append(s.sc[ss](h[ss],inp))
                else: nh.append(h[ss])
            h=nh; fu=s.fu(torch.cat(h,-1))
            fu=fu+sum(s.sp[ss](h[ss]) for ss in range(s.ns))/s.ns
            o.append(s.h(s.fn(fu+inp)).unsqueeze(1))
        return torch.cat(o,1),h

# ============================================================
# v5a: Dual-Path (LM残差 + Memory残差)
# ============================================================
class V5_DualPath(nn.Module):
    """
    LM Path: 全替换更新(α→0), scales=2, 专注即时预测
    Memory Path: 残差更新(α→0.88), scales=4, 专注长期记忆
    最终输出: gate * LM_path + (1-gate) * Mem_path + inp
    """
    def __init__(s, vs, H=256):
        super().__init__(); s.H=H
        s.e=nn.Embedding(vs,H); s.ii=nn.Linear(H,H); s.inm=nn.LayerNorm(H)
        # LM Path: 2 scales, fast update
        s.lm_sc=nn.ModuleList([ScaleBlock(H) for _ in range(2)])
        s.lm_sp=nn.ModuleList([nn.Linear(H,H) for _ in range(2)])
        s.lm_fu=nn.Linear(H*2,H); s.lm_fn=nn.LayerNorm(H)
        # Memory Path: 4 scales, slow update (残差)
        s.mem_sc=nn.ModuleList([ScaleBlock(H,1.0) for _ in range(4)])  # exp=1 for mem
        s.mem_sp=nn.ModuleList([nn.Linear(H,H) for _ in range(4)])
        s.mem_fu=nn.Linear(H*4,H); s.mem_fn=nn.LayerNorm(H)
        # Path gate: input decides LM vs Memory ratio
        s.path_gate=nn.Sequential(nn.Linear(H,H//4),nn.GELU(),nn.Linear(H//4,1),nn.Sigmoid())
        # Final
        s.out_n=nn.LayerNorm(H); s.h=nn.Linear(H,vs)

    def forward(s,x,hp=None):
        B,T=x.shape
        lm_h=[torch.zeros(B,s.H,device=x.device) for _ in range(2)]
        mem_h=[torch.zeros(B,s.H,device=x.device) for _ in range(4)]
        xe=s.e(x); o=[]
        for t in range(T):
            inp=s.inm(s.ii(xe[:,t,:]))
            # LM Path: scales 0,1 update every step (period=1,2)
            lm_nh=[]
            for ss in range(2):
                if t%(2**ss)==0: lm_nh.append(s.lm_sc[ss](lm_h[ss],inp))
                else: lm_nh.append(lm_h[ss])
            lm_h=lm_nh
            lm_out=s.lm_fn(s.lm_fu(torch.cat(lm_h,-1))
                          +sum(s.lm_sp[ss](lm_h[ss]) for ss in range(2))/2)
            # Memory Path: scales 0-3 update at periods 1,2,4,8
            mem_nh=[]
            for ss in range(4):
                if t%(2**ss)==0: mem_nh.append(s.mem_sc[ss](mem_h[ss],inp))
                else: mem_nh.append(mem_h[ss])
            mem_h=mem_nh
            mem_out=s.mem_fn(s.mem_fu(torch.cat(mem_h,-1))
                            +sum(s.mem_sp[ss](mem_h[ss]) for ss in range(4))/4)
            # Gate blend
            gate=s.path_gate(inp).squeeze(-1).unsqueeze(-1)  # (B,1)
            out=s.out_n(gate*lm_out+(1-gate)*mem_out+inp)
            o.append(s.h(out).unsqueeze(1))
        return torch.cat(o,1),lm_h+mem_h

# ============================================================
# v5b: Selective Write (输入决定每尺度写入强度)
# ============================================================
class V5_SelectiveWrite(nn.Module):
    """
    每个尺度每个timestep: 输入门 = 基础门 + 重要性门
    重要性门 = sigmoid(Linear(concat(h, inp))) → 决定当前token对当前尺度的重要性
    如果 token 很重要 → 输入门饱和 → 全写入 (behaves like v1)
    如果 token 不重要 → 输入门关闭 → 保留旧状态 (behaves like v3 residual)
    """
    def __init__(s, vs, H=256, ns=4):
        super().__init__(); s.H=H; s.ns=ns
        s.e=nn.Embedding(vs,H); s.ii=nn.Linear(H,H); s.inm=nn.LayerNorm(H)
        s.sc=nn.ModuleList([ScaleBlock(H) for _ in range(ns)])
        # Importance gate per scale
        s.imp_gate=nn.ModuleList([nn.Linear(H,1) for _ in range(ns)])
        s.sp=nn.ModuleList([nn.Linear(H,H) for _ in range(ns)])
        s.fu=nn.Linear(H*ns,H); s.fn=nn.LayerNorm(H); s.h=nn.Linear(H,vs)

    def forward(s,x,hp=None):
        B,T=x.shape
        if hp is None: h=[torch.zeros(B,s.H,device=x.device) for _ in range(s.ns)]
        else: h=[c.clone() for c in hp]
        xe=s.e(x); o=[]
        for t in range(T):
            inp=s.inm(s.ii(xe[:,t,:])); nh=[]
            for ss in range(s.ns):
                if t%(2**ss)==0:
                    # Standard update
                    h_new=s.sc[ss](h[ss],inp)
                    # Importance gate: should we keep new or retain old?
                    imp=torch.sigmoid(s.imp_gate[ss](inp)).squeeze(-1).unsqueeze(-1)  # (B,1)
                    # imp→1: input is important, keep new state
                    # imp→0: input is noise, retain old state
                    nh.append(imp*h_new + (1-imp)*h[ss])
                else: nh.append(h[ss])
            h=nh; fu=s.fu(torch.cat(h,-1))
            fu=fu+sum(s.sp[ss](h[ss]) for ss in range(s.ns))/s.ns
            o.append(s.h(s.fn(fu+inp)).unsqueeze(1))
        return torch.cat(o,1),h

# ============================================================
dataset=PretrainDataset("minimind_data/pretrain_t2t_mini.jsonl",voc,max_len=256,max_lines=2000)

def make_cf(bs,nl):
    t=torch.randint(2,VCF,(bs,)); n=torch.randint(2,VCF,(bs,nl))
    e=torch.full((bs,1),ECF,dtype=torch.long)
    x=torch.cat([t.unsqueeze(1),n,e],1); y=torch.full_like(x,ICF); y[:,-1]=t
    return x,y

def train_lm(model,name,steps=500):
    loader=DataLoader(dataset,batch_size=4,shuffle=True,collate_fn=PretrainDataset.collate_fn,drop_last=True)
    torch.manual_seed(42); model=model.to(device)
    opt=AdamW(model.parameters(),lr=3e-4,weight_decay=0.01,betas=(0.9,0.95))
    def lr(o,w,t):
        def f(s):
            if s<w: return s/max(1,w)
            p=(s-w)/max(1,t-w); return max(0.0,0.5*(1.0+math.cos(math.pi*p)))
        return torch.optim.lr_scheduler.LambdaLR(o,f)
    sch=lr(opt,50,steps); model.train(); best=float('inf'); di=iter(loader)
    for step in range(steps):
        try: x,t=next(di)
        except: di=iter(loader); x,t=next(di)
        x,t=x.to(device),t.to(device); log,_=model(x)
        loss=F.cross_entropy(log.reshape(-1,vs),t.reshape(-1),ignore_index=0)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sch.step()
        if loss.item()<best: best=loss.item()
    model.eval()
    with torch.no_grad():
        tl=0;tt=0
        for x,t in loader:
            x,t=x.to(device),t.to(device); log,_=model(x)
            l=F.cross_entropy(log.reshape(-1,vs),t.reshape(-1),ignore_index=0,reduction='sum')
            tl+=l.item();tt+=(t!=0).sum().item()
        el=tl/tt
    del model; torch.cuda.empty_cache()
    return el,best

def train_cf(model,name,steps=2500):
    torch.manual_seed(42); model=model.to(device)
    opt=AdamW(model.parameters(),lr=1e-3,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps)
    model.train(); best=float('inf')
    for st in range(1,steps+1):
        x,y=make_cf(64,random.randint(4,64)); x,y=x.to(device),y.to(device)
        log,_=model(x); loss=F.cross_entropy(log[:,-1,:],y[:,-1],ignore_index=ICF)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sch.step()
        if loss.item()<best: best=loss.item()
    model.eval(); accs={}
    for d in [4,64,256,1024,4096,8192,16384,32768,65536]:
        eb=64 if d<=4096 else 8; c=0;total=0
        for _ in range(4 if d<=4096 else 2):
            x,y=make_cf(eb,d); x,y=x.to(device),y.to(device)
            log,_=model(x); c+=(log[:,-1,:].argmax(-1)==y[:,-1]).sum().item(); total+=eb
        accs[d]=c/total*100
    del model; torch.cuda.empty_cache()
    return best,accs

# ============================================================
print(f"{'='*70}")
print(f"  FRSM v5: Dual-Path vs SelectiveWrite vs V1 Baseline")
print(f"{'='*70}")

results={}

# V1 baseline
el,b=train_lm(V1(vs,256,4),"V1-Orig")
results['V1']={'lm':el,'lm_b':b}
cf_b,cf_a=train_cf(V1(VCF,HCF,4),"V1-CF")
results['V1']['cf_b']=cf_b; results['V1']['cf_a']=cf_a

# V5a Dual-Path
el,b=train_lm(V5_DualPath(vs,256),"V5-DualPath")
results['V5a']={'lm':el,'lm_b':b}
cf_b,cf_a=train_cf(V5_DualPath(VCF,128),"V5a-CF")
results['V5a']['cf_b']=cf_b; results['V5a']['cf_a']=cf_a

# V5b SelectiveWrite
el,b=train_lm(V5_SelectiveWrite(vs,256,4),"V5-SelectiveWrite")
results['V5b']={'lm':el,'lm_b':b}
cf_b,cf_a=train_cf(V5_SelectiveWrite(VCF,HCF,4),"V5b-CF")
results['V5b']['cf_b']=cf_b; results['V5b']['cf_a']=cf_a

# ============================================================
print(f"\n{'='*70}")
print(f"  Final Comparison")
print(f"{'='*70}")
print(f"  {'Model':>20} | {'LM Loss':>8} | {'CF best':>8} | {'CF@1K':>7} | {'CF@16K':>7} | {'CF@65K':>7}")
print(f"  "+"-"*68)
for k in ['V1','V5a','V5b']:
    r=results[k]
    cfs=r.get('cf_a',{})
    print(f"  {k:>20} | {r['lm']:8.4f} | {r.get('cf_b',0):8.5f} | {cfs.get(1024,0):6.1f}% | {cfs.get(16384,0):6.1f}% | {cfs.get(65536,0):6.1f}%")

# Winner determination
best_lm=min(results,key=lambda k:results[k]['lm'])
best_cf_65k=max(results,key=lambda k:results[k].get('cf_a',{}).get(65536,0))
print(f"\n  Best LM: {best_lm} ({results[best_lm]['lm']:.4f})")
print(f"  Best CF@65K: {best_cf_65k} ({results[best_cf_65k].get('cf_a',{}).get(65536,0):.1f}%)")
print(f"\nDone.")
