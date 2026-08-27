"""
FRSM v3多层深度消融: 1层 vs 3层 vs 5层
基于v3 Residual设计，只加深度
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

class ScaleRecurrentBlock(nn.Module):
    def __init__(self, d_model, alpha_init=2.0):
        super().__init__()
        self.W_forget=nn.Linear(d_model*2,d_model)
        self.W_input=nn.Linear(d_model*2,d_model)
        self.W_cand=nn.Linear(d_model*2,d_model)
        nn.init.constant_(self.W_forget.bias,1.0)
        nn.init.constant_(self.W_input.bias,-2.0)
        self.register_buffer('alpha',torch.sigmoid(torch.tensor(alpha_init)))
        self.norm=nn.LayerNorm(d_model)
    def forward(self,h_prev,inp):
        c=torch.cat([h_prev,inp],-1)
        f=torch.sigmoid(self.W_forget(c)); i=torch.sigmoid(self.W_input(c))
        cand=f*h_prev+i*torch.tanh(self.W_cand(c))
        return self.norm(self.alpha*h_prev+(1-self.alpha)*cand)

class ResidualFRSM_Layer(nn.Module):
    """单层 v3 Residual FRSM"""
    def __init__(self, d_model, num_scales=4):
        super().__init__()
        self.H=d_model; self.ns=num_scales
        self.inp_norm=nn.LayerNorm(d_model)
        self.scales=nn.ModuleList([ScaleRecurrentBlock(d_model) for _ in range(num_scales)])
        self.scale_proj=nn.ModuleList([nn.Linear(d_model,d_model) for _ in range(num_scales)])
        self.fusion=nn.Linear(d_model*num_scales,d_model)
        self.fusion_norm=nn.LayerNorm(d_model)
    def forward(self, inp, h_prev, t_step):
        B,D=inp.shape
        if h_prev is None: h=[torch.zeros(B,D,device=inp.device) for _ in range(self.ns)]
        else: h=h_prev
        inp_n=self.inp_norm(inp)
        nh=[]
        for s in range(self.ns):
            if t_step%(2**s)==0: nh.append(self.scales[s](h[s],inp_n))
            else: nh.append(h[s])
        fused=self.fusion(torch.cat(nh,-1))
        fused=fused+sum(self.scale_proj[s](nh[s]) for s in range(self.ns))/self.ns
        out=self.fusion_norm(fused+inp_n)
        return out, nh

class DeepResidualFRSM(nn.Module):
    """多层堆叠 v3 Residual FRSM"""
    def __init__(self, vocab_size, d_model=256, num_scales=4, num_layers=1):
        super().__init__()
        self.H=d_model; self.ns=num_scales; self.nl=num_layers
        self.embed=nn.Embedding(vocab_size,d_model)
        self.input_proj=nn.Linear(d_model,d_model)
        self.layers=nn.ModuleList([ResidualFRSM_Layer(d_model,num_scales) for _ in range(num_layers)])
        self.head=nn.Linear(d_model,vocab_size)
    def forward(self,x,h_prev=None):
        B,T=x.shape
        if h_prev is None:
            states=[[torch.zeros(B,self.H,device=x.device) for _ in range(self.ns)] for _ in range(self.nl)]
        else:
            states=[[s.clone() for s in layer_s] for layer_s in h_prev]
        xe=self.embed(x); outs=[]
        for t in range(T):
            inp=self.input_proj(xe[:,t,:])
            for l in range(self.nl):
                out,states[l]=self.layers[l](inp,states[l],t)
                inp=out  # 下一层的输入 = 上一层的输出
            outs.append(self.head(out).unsqueeze(1))
        return torch.cat(outs,1),states

# 原始 v1 深度模型对比
class FRSM_Orig(nn.Module):
    def __init__(self, vs, d_model=256, ns=4):
        super().__init__()
        self.H=d_model; self.ns=ns
        self.embed=nn.Embedding(vs,d_model); self.inp=nn.Linear(d_model,d_model)
        self.Wf=nn.ModuleList([nn.Linear(d_model*2,d_model) for _ in range(ns)])
        self.Wi=nn.ModuleList([nn.Linear(d_model*2,d_model) for _ in range(ns)])
        self.Wc=nn.ModuleList([nn.Linear(d_model*2,d_model) for _ in range(ns)])
        for w in self.Wf: nn.init.constant_(w.bias,1.0)
        for w in self.Wi: nn.init.constant_(w.bias,-2.0)
        self.fusion=nn.Linear(d_model*ns,d_model); self.norm=nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,vs)
    def forward(self,x,hp=None):
        B,T=x.shape
        if hp is None: h=[torch.zeros(B,self.H,device=x.device) for _ in range(self.ns)]
        else: h=[s.clone() for s in hp]
        xe=self.embed(x); outs=[]
        for t in range(T):
            inp=self.inp(xe[:,t,:]); nh=[]
            for s in range(self.ns):
                if t%(2**s)==0:
                    c=torch.cat([h[s],inp],-1)
                    f=torch.sigmoid(self.Wf[s](c)); i=torch.sigmoid(self.Wi[s](c))
                    nh.append(f*h[s]+i*torch.tanh(self.Wc[s](c)))
                else: nh.append(h[s])
            h=nh; fused=self.norm(self.fusion(torch.cat(h,-1)))
            outs.append(self.head(fused).unsqueeze(1))
        return torch.cat(outs,1),h

# CopyFirst 版本
VOCAB_CF=32; END_CF=0; IGN_CF=1; H_CF=128
def make_cf(bs,nl):
    t=torch.randint(2,VOCAB_CF,(bs,)); n=torch.randint(2,VOCAB_CF,(bs,nl))
    e=torch.full((bs,1),END_CF,dtype=torch.long)
    x=torch.cat([t.unsqueeze(1),n,e],1); y=torch.full_like(x,IGN_CF); y[:,-1]=t
    return x,y

# ============================================================
print(f"{'='*70}")
print(f"  FRSM v3 Depth Ablation: 1 vs 2 vs 3 layers")
print(f"  LM Loss (500 steps) + CopyFirst (2500 steps)")
print(f"{'='*70}")

dataset=PretrainDataset("minimind_data/pretrain_t2t_mini.jsonl",voc,max_len=256,max_lines=2000)

def train_lm(model,name,steps=500):
    loader=DataLoader(dataset,batch_size=4,shuffle=True,collate_fn=PretrainDataset.collate_fn,drop_last=True)
    torch.manual_seed(42); model=model.to(device)
    n=sum(p.numel() for p in model.parameters())
    opt=AdamW(model.parameters(),lr=3e-4,weight_decay=0.01,betas=(0.9,0.95))
    def lr_s(opt,w,t):
        def f(s):
            if s<w: return s/max(1,w)
            p=(s-w)/max(1,t-w); return max(0.0,0.5*(1.0+math.cos(math.pi*p)))
        return torch.optim.lr_scheduler.LambdaLR(opt,f)
    sch=lr_s(opt,50,steps); model.train(); best=float('inf'); di=iter(loader)
    print(f"\n  [{name}] {n:,}p",flush=True)
    for step in range(steps):
        try: x,t=next(di)
        except: di=iter(loader); x,t=next(di)
        x,t=x.to(device),t.to(device)
        log,_=model(x); loss=F.cross_entropy(log.reshape(-1,vs),t.reshape(-1),ignore_index=0)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sch.step()
        if loss.item()<best: best=loss.item()
        if (step+1)%100==0: print(f"    step{step+1:4d} loss={loss.item():.4f} best={best:.4f}",flush=True)
    model.eval()
    with torch.no_grad():
        tl=0;tt=0
        for x,t in loader:
            x,t=x.to(device),t.to(device)
            log,_=model(x); l=F.cross_entropy(log.reshape(-1,vs),t.reshape(-1),ignore_index=0,reduction='sum')
            tl+=l.item();tt+=(t!=0).sum().item()
        el=tl/tt
    print(f"    => eval={el:.4f} ppl={math.exp(el):.2f}")
    del model; torch.cuda.empty_cache()
    return el,best

def train_cf(ModelClass,args,name,steps=2500):
    torch.manual_seed(42); model=ModelClass(*args).to(device)
    n=sum(p.numel() for p in model.parameters())
    opt=AdamW(model.parameters(),lr=1e-3,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps)
    model.train(); best=float('inf')
    print(f"\n  [{name}] {n:,}p",flush=True)
    for st in range(1,steps+1):
        x,y=make_cf(64,random.randint(4,64)); x,y=x.to(device),y.to(device)
        log,_=model(x); loss=F.cross_entropy(log[:,-1,:],y[:,-1],ignore_index=IGN_CF)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sch.step()
        if loss.item()<best: best=loss.item()
        if st%500==0: print(f"    step{st:5d} best={best:.5f}",flush=True)
    model.eval(); accs={}
    for d in [4,64,1024,4096,16384,32768,65536]:
        eb=64 if d<=4096 else 8; c=0;total=0
        for _ in range(4 if d<=4096 else 2):
            x,y=make_cf(eb,d); x,y=x.to(device),y.to(device)
            log,_=model(x); c+=(log[:,-1,:].argmax(-1)==y[:,-1]).sum().item(); total+=eb
        accs[d]=c/total*100
    del model; torch.cuda.empty_cache()
    return best,accs

# ============================================================
# LM 实验
# ============================================================
print(f"\n--- LM Loss ---")
lm_results={}
for nl in [1,2,3]:
    el,best=train_lm(DeepResidualFRSM(vs,256,4,nl),f"Resid-{nl}L")
    lm_results[nl]=(el,best)

# CopyFirst 实验
print(f"\n--- CopyFirst ---")
cf_results={}
for nl in [1,2,3]:
    best,accs=train_cf(DeepResidualFRSM,(VOCAB_CF,H_CF,4,nl),f"Resid-{nl}L")
    cf_results[nl]=(best,accs)

# 也加入 v1 Orig-4sc 和 Orig-2sc 做对照
el_o4,best_o4=train_lm(FRSM_Orig(vs,256,4),"Orig-4sc")
lm_results['o4']=(el_o4,best_o4)
el_o2,best_o2=train_lm(FRSM_Orig(vs,256,2),"Orig-2sc")
lm_results['o2']=(el_o2,best_o2)
best_o4_cf,accs_o4_cf=train_cf(FRSM_Orig,(VOCAB_CF,H_CF,4),"Orig-4sc")
cf_results['o4']=(best_o4_cf,accs_o4_cf)

# ============================================================
# 汇总
# ============================================================
print(f"\n{'='*70}")
print(f"  Final Results")
print(f"{'='*70}")
print(f"  {'Model':>14} | {'Params':>9} | {'LM Loss':>8} | {'CF best':>8} | {'CF@4K':>7} | {'CF@16K':>7} | {'CF@32K':>7} | {'CF@65K':>7}")
print(f"  "+"-"*78)
for nl,tag,par in [(1,"Resid-1L",sum(p.numel() for p in DeepResidualFRSM(vs,256,4,1).parameters())),
                    (2,"Resid-2L",sum(p.numel() for p in DeepResidualFRSM(vs,256,4,2).parameters())),
                    (3,"Resid-3L",sum(p.numel() for p in DeepResidualFRSM(vs,256,4,3).parameters())),
                    ('o4',"Orig-4sc",sum(p.numel() for p in FRSM_Orig(vs,256,4).parameters())),
                    ('o2',"Orig-2sc",sum(p.numel() for p in FRSM_Orig(vs,256,2).parameters()))]:
    el,best=lm_results[nl]
    if nl in cf_results:
        cf_best,cf_accs=cf_results[nl]
        cf4k=cf_accs.get(4096,0); cf16k=cf_accs.get(16384,0)
        cf32k=cf_accs.get(32768,0); cf65k=cf_accs.get(65536,0)
    else:
        cf_best='—'; cf4k=cf16k=cf32k=cf65k='—'
    print(f"  {tag:>14} | {par:>9,} | {el:8.4f} | {cf_best if isinstance(cf_best,str) else f'{cf_best:8.5f}'} | {cf4k if isinstance(cf4k,str) else f'{cf4k:6.1f}%'} | {cf16k if isinstance(cf16k,str) else f'{cf16k:6.1f}%'} | {cf32k if isinstance(cf32k,str) else f'{cf32k:6.1f}%'} | {cf65k if isinstance(cf65k,str) else f'{cf65k:6.1f}%'}")

print(f"\nDone.")
