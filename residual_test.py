"""
FRSM v3: 残差结构改进
核心改动: 尺度间残差连接 + 层内残差, 保持多尺度但避免梯度稀疏
对比: 原始FRSM(4sc) vs 残差FRSM(4sc) vs 最优FRSM(2sc)
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

# ============================================================
# 原始 FRSM (baseline)
# ============================================================
class FRSM_Orig(nn.Module):
    def __init__(self, d_model, num_scales=4):
        super().__init__()
        self.H=d_model; self.ns=num_scales
        self.embed=nn.Embedding(vs,d_model); self.inp=nn.Linear(d_model,d_model)
        self.Wf=nn.ModuleList([nn.Linear(d_model*2,d_model) for _ in range(num_scales)])
        self.Wi=nn.ModuleList([nn.Linear(d_model*2,d_model) for _ in range(num_scales)])
        self.Wc=nn.ModuleList([nn.Linear(d_model*2,d_model) for _ in range(num_scales)])
        for w in self.Wf: nn.init.constant_(w.bias,1.0)
        for w in self.Wi: nn.init.constant_(w.bias,-2.0)
        self.fusion=nn.Linear(d_model*num_scales,d_model)
        self.norm=nn.LayerNorm(d_model); self.head=nn.Linear(d_model,vs)
    def forward(self,x):
        B,T=x.shape
        h=[torch.zeros(B,self.H,device=x.device) for _ in range(self.ns)]
        xe=self.embed(x); outs=[]
        for t in range(T):
            inp=self.inp(xe[:,t,:]); nh=[]
            for s in range(self.ns):
                if t%(2**s)==0:
                    c=torch.cat([h[s],inp],-1)
                    f=torch.sigmoid(self.Wf[s](c)); i=torch.sigmoid(self.Wi[s](c))
                    nh.append(f*h[s]+i*torch.tanh(self.Wc[s](c)))
                else: nh.append(h[s])
            h=nh
            fused=self.norm(self.fusion(torch.cat(h,-1)))
            outs.append(self.head(fused).unsqueeze(1))
        return torch.cat(outs,1)

# ============================================================
# 残差 FRSM: 每个尺度有残差连接 + 输入旁路
# ============================================================
class FRSM_Residual(nn.Module):
    """
    改进点:
    1. 每个尺度内部残差: h_new = h_old + delta_h (而非完全替换)
    2. 输入旁路: 每步都将 inp 加到所有尺度的候选中
    3. 融合残差: fused = inp + fusion(concat(h))
    """
    def __init__(self, d_model, num_scales=4):
        super().__init__()
        self.H=d_model; self.ns=num_scales
        self.embed=nn.Embedding(vs,d_model); self.inp=nn.Linear(d_model,d_model)
        self.inp_norm=nn.LayerNorm(d_model)
        # 每个尺度: delta generator (输出增量而非新状态)
        self.Wf=nn.ModuleList([nn.Linear(d_model*2,d_model) for _ in range(num_scales)])
        self.Wi=nn.ModuleList([nn.Linear(d_model*2,d_model) for _ in range(num_scales)])
        self.Wc=nn.ModuleList([nn.Linear(d_model*2,d_model) for _ in range(num_scales)])
        for w in self.Wf: nn.init.constant_(w.bias,1.0)
        for w in self.Wi: nn.init.constant_(w.bias,-2.0)
        # 尺度间残差连接: 每个尺度也直接贡献到输出
        self.scale_proj=nn.ModuleList([nn.Linear(d_model,d_model) for _ in range(num_scales)])
        self.fusion=nn.Linear(d_model*num_scales,d_model)
        self.norm=nn.LayerNorm(d_model); self.head=nn.Linear(d_model,vs)

    def forward(self,x):
        B,T=x.shape
        h=[torch.zeros(B,self.H,device=x.device) for _ in range(self.ns)]
        xe=self.embed(x); outs=[]
        for t in range(T):
            inp=self.inp_norm(self.inp(xe[:,t,:]))
            nh=[]
            for s in range(self.ns):
                if t%(2**s)==0:
                    c=torch.cat([h[s],inp],-1)
                    f=torch.sigmoid(self.Wf[s](c)); i=torch.sigmoid(self.Wi[s](c))
                    delta=i*torch.tanh(self.Wc[s](c))
                    # 残差: 状态 = 旧状态 * forget_gate + delta * input_gate
                    # 关键改动: 加回旧状态的残差
                    nh.append(f*h[s]+delta + (1-f)*h[s]*(t>0))  # 残差保持
                else:
                    nh.append(h[s])
            h=nh
            # 融合 + 输入旁路残差
            cat=torch.cat(h,-1)
            fused=self.fusion(cat)
            # 每个尺度的独立贡献也加上
            for s in range(self.ns):
                fused=fused+self.scale_proj[s](h[s])*(1.0/self.ns)
            fused=self.norm(fused+inp)  # 输入残差
            outs.append(self.head(fused).unsqueeze(1))
        return torch.cat(outs,1)

# ============================================================
# 另一种残差: 简单的 h_new = h_old + gate * candidate
# ============================================================
class FRSM_AddResidual(nn.Module):
    """最简单的残差: h = h + gate * candidate, 无 forget gate"""
    def __init__(self, d_model, num_scales=4):
        super().__init__()
        self.H=d_model; self.ns=num_scales
        self.embed=nn.Embedding(vs,d_model); self.inp=nn.Linear(d_model,d_model)
        # 只用 input gate, 状态是累加式
        self.Wg=nn.ModuleList([nn.Linear(d_model*2,d_model) for _ in range(num_scales)])
        self.Wc=nn.ModuleList([nn.Linear(d_model*2,d_model) for _ in range(num_scales)])
        for w in self.Wg: nn.init.constant_(w.bias,-1.0)  # 默认不写
        # 尺度归一化 (替代 forget gate 的稳定性作用)
        self.scale_norm=nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_scales)])
        self.fusion=nn.Linear(d_model*num_scales,d_model)
        self.norm=nn.LayerNorm(d_model); self.head=nn.Linear(d_model,vs)

    def forward(self,x):
        B,T=x.shape
        h=[torch.zeros(B,self.H,device=x.device) for _ in range(self.ns)]
        xe=self.embed(x); outs=[]
        for t in range(T):
            inp=self.inp(xe[:,t,:]); nh=[]
            for s in range(self.ns):
                if t%(2**s)==0:
                    c=torch.cat([h[s],inp],-1)
                    g=torch.sigmoid(self.Wg[s](c))
                    delta=g*torch.tanh(self.Wc[s](c))
                    # 纯残差 + 归一化
                    h_new=self.scale_norm[s](h[s]+delta)
                    nh.append(h_new)
                else:
                    nh.append(h[s])
            h=nh
            fused=self.norm(self.fusion(torch.cat(h,-1)))
            outs.append(self.head(fused).unsqueeze(1))
        return torch.cat(outs,1)

# ============================================================
# 训练+评估
# ============================================================
dataset = PretrainDataset("minimind_data/pretrain_t2t_mini.jsonl", voc, max_len=256, max_lines=2000)

def get_lr(opt,w,t):
    def f(s):
        if s<w: return s/max(1,w)
        p=(s-w)/max(1,t-w); return max(0.0,0.5*(1.0+math.cos(math.pi*p)))
    return torch.optim.lr_scheduler.LambdaLR(opt,f)

VOCAB_CF=32; END_CF=0; IGN_CF=1; H_CF=128

def make_cf_batch(bs,nl):
    t=torch.randint(2,VOCAB_CF,(bs,))
    n=torch.randint(2,VOCAB_CF,(bs,nl))
    e=torch.full((bs,1),END_CF,dtype=torch.long)
    x=torch.cat([t.unsqueeze(1),n,e],1)
    y=torch.full_like(x,IGN_CF); y[:,-1]=t
    return x,y

# CopyFirst mini 模型 (用于长期依赖测试)
class CF_FRSM_Orig(nn.Module):
    def __init__(self):
        super().__init__()
        self.H=128; self.ns=4
        self.embed=nn.Embedding(VOCAB_CF,128); self.inp=nn.Linear(128,128)
        self.Wf=nn.ModuleList([nn.Linear(256,128) for _ in range(4)])
        self.Wi=nn.ModuleList([nn.Linear(256,128) for _ in range(4)])
        self.Wc=nn.ModuleList([nn.Linear(256,128) for _ in range(4)])
        for w in self.Wf: nn.init.constant_(w.bias,1.0)
        for w in self.Wi: nn.init.constant_(w.bias,-2.0)
        self.fusion=nn.Linear(512,128); self.ln=nn.LayerNorm(128)
        self.head=nn.Linear(128,VOCAB_CF)
    def forward(self,x,hp=None):
        B,T=x.shape
        if hp is None: h=[torch.zeros(B,128,device=device) for _ in range(4)]
        else: h=[s.clone() for s in hp]
        xe=self.embed(x); outs=[]
        for t in range(T):
            inp=self.inp(xe[:,t,:]); nh=[]
            for s in range(4):
                if t%(2**s)==0:
                    c=torch.cat([h[s],inp],-1)
                    f=torch.sigmoid(self.Wf[s](c)); i=torch.sigmoid(self.Wi[s](c))
                    nh.append(f*h[s]+i*torch.tanh(self.Wc[s](c)))
                else: nh.append(h[s])
            h=nh; fused=self.ln(self.fusion(torch.cat(h,-1)))
            outs.append(self.head(fused).unsqueeze(1))
        return torch.cat(outs,1),h

class CF_FRSM_Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.H=128; self.ns=4
        self.embed=nn.Embedding(VOCAB_CF,128); self.inp=nn.Linear(128,128)
        self.inp_norm=nn.LayerNorm(128)
        self.Wf=nn.ModuleList([nn.Linear(256,128) for _ in range(4)])
        self.Wi=nn.ModuleList([nn.Linear(256,128) for _ in range(4)])
        self.Wc=nn.ModuleList([nn.Linear(256,128) for _ in range(4)])
        for w in self.Wf: nn.init.constant_(w.bias,1.0)
        for w in self.Wi: nn.init.constant_(w.bias,-2.0)
        self.scale_proj=nn.ModuleList([nn.Linear(128,128) for _ in range(4)])
        self.fusion=nn.Linear(512,128); self.ln=nn.LayerNorm(128)
        self.head=nn.Linear(128,VOCAB_CF)
    def forward(self,x,hp=None):
        B,T=x.shape
        if hp is None: h=[torch.zeros(B,128,device=device) for _ in range(4)]
        else: h=[s.clone() for s in hp]
        xe=self.embed(x); outs=[]
        for t in range(T):
            inp=self.inp_norm(self.inp(xe[:,t,:])); nh=[]
            for s in range(4):
                if t%(2**s)==0:
                    c=torch.cat([h[s],inp],-1)
                    f=torch.sigmoid(self.Wf[s](c)); i=torch.sigmoid(self.Wi[s](c))
                    delta=i*torch.tanh(self.Wc[s](c))
                    nh.append(f*h[s]+delta+(1-f)*h[s]*(t>0))
                else: nh.append(h[s])
            h=nh
            cat=torch.cat(h,-1); fused=self.fusion(cat)
            for s in range(4): fused=fused+self.scale_proj[s](h[s])*0.25
            fused=self.ln(fused+inp)
            outs.append(self.head(fused).unsqueeze(1))
        return torch.cat(outs,1),h

class CF_FRSM_AddRes(nn.Module):
    def __init__(self):
        super().__init__()
        self.H=128; self.ns=4
        self.embed=nn.Embedding(VOCAB_CF,128); self.inp=nn.Linear(128,128)
        self.Wg=nn.ModuleList([nn.Linear(256,128) for _ in range(4)])
        self.Wc=nn.ModuleList([nn.Linear(256,128) for _ in range(4)])
        for w in self.Wg: nn.init.constant_(w.bias,-1.0)
        self.scale_norm=nn.ModuleList([nn.LayerNorm(128) for _ in range(4)])
        self.fusion=nn.Linear(512,128); self.ln=nn.LayerNorm(128)
        self.head=nn.Linear(128,VOCAB_CF)
    def forward(self,x,hp=None):
        B,T=x.shape
        if hp is None: h=[torch.zeros(B,128,device=device) for _ in range(4)]
        else: h=[s.clone() for s in hp]
        xe=self.embed(x); outs=[]
        for t in range(T):
            inp=self.inp(xe[:,t,:]); nh=[]
            for s in range(4):
                if t%(2**s)==0:
                    c=torch.cat([h[s],inp],-1)
                    g=torch.sigmoid(self.Wg[s](c)); delta=g*torch.tanh(self.Wc[s](c))
                    nh.append(self.scale_norm[s](h[s]+delta))
                else: nh.append(h[s])
            h=nh; fused=self.ln(self.fusion(torch.cat(h,-1)))
            outs.append(self.head(fused).unsqueeze(1))
        return torch.cat(outs,1),h

# ============================================================
# 实验1: LM Loss 对比
# ============================================================
print(f"{'='*70}")
print(f"  Part 1: LM Loss — Original vs Residual vs AddResidual")
print(f"{'='*70}")

lm_models = [
    ("Orig-4sc",    FRSM_Orig(256, 4)),
    ("Orig-2sc",    FRSM_Orig(256, 2)),
    ("Residual-4sc",FRSM_Residual(256, 4)),
    ("AddRes-4sc",  FRSM_AddResidual(256, 4)),
]

lm_results = {}
for name, model in lm_models:
    n = sum(p.numel() for p in model.parameters())
    loader = DataLoader(dataset, batch_size=4, shuffle=True,
                       collate_fn=PretrainDataset.collate_fn, drop_last=True)
    torch.manual_seed(42); model = model.to(device)
    opt = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9,0.95))
    sch = get_lr(opt, 50, 500)
    model.train(); step=0; best=float('inf'); t0=time.time()
    di=iter(loader)
    print(f"\n  [{name}] {n:,} params", flush=True)
    while step < 500:
        try: x,t=next(di)
        except: di=iter(loader); x,t=next(di)
        x,t=x.to(device),t.to(device)
        logits=model(x)
        loss=F.cross_entropy(logits.reshape(-1,vs),t.reshape(-1),ignore_index=0)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sch.step(); step+=1
        if loss.item()<best: best=loss.item()
        if step%100==0: print(f"    step{step:4d} loss={loss.item():.4f} best={best:.4f} {time.time()-t0:.0f}s",flush=True)
    model.eval()
    with torch.no_grad():
        tl=0;tt=0
        for x,t in loader:
            x,t=x.to(device),t.to(device)
            logits=model(x)
            l=F.cross_entropy(logits.reshape(-1,vs),t.reshape(-1),ignore_index=0,reduction='sum')
            tl+=l.item();tt+=(t!=0).sum().item()
        el=tl/tt; ep=math.exp(el) if el<20 else 99999
    lm_results[name]={'best':best,'eval':el,'ppl':ep,'params':n}
    print(f"    => eval={el:.4f} ppl={ep:.2f}",flush=True)
    del model; torch.cuda.empty_cache()

print(f"\n{'='*70}")
print(f"  LM Loss Results")
print(f"{'='*70}")
print(f"  {'Model':>14} | {'Params':>10} | {'Best':>8} | {'Eval':>8} | {'PPL':>8}")
print(f"  "+"-"*50)
for name in ["Orig-2sc","Orig-4sc","Residual-4sc","AddRes-4sc"]:
    r=lm_results[name]
    print(f"  {name:>14} | {r['params']:10,} | {r['best']:8.4f} | {r['eval']:8.4f} | {r['ppl']:8.2f}")

# ============================================================
# 实验2: CopyFirst 长期依赖对比
# ============================================================
print(f"\n{'='*70}")
print(f"  Part 2: CopyFirst — Original vs Residual vs AddResidual")
print(f"{'='*70}")

cf_models = [
    ("Orig-4sc",    CF_FRSM_Orig()),
    ("Residual-4sc",CF_FRSM_Residual()),
    ("AddRes-4sc",  CF_FRSM_AddRes()),
]

cf_results = {}
for name, model in cf_models:
    n=sum(p.numel() for p in model.parameters())
    torch.manual_seed(42); model=model.to(device)
    opt=AdamW(model.parameters(),lr=1e-3,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,2500)
    model.train(); best=float('inf'); t0=time.time()
    print(f"\n  [{name}] {n:,} params",flush=True)
    for st in range(1,2501):
        x,y=make_cf_batch(64,random.randint(4,64)); x,y=x.to(device),y.to(device)
        log,_=model(x); loss=F.cross_entropy(log[:,-1,:],y[:,-1],ignore_index=IGN_CF)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sch.step()
        if loss.item()<best: best=loss.item()
        if st%500==0: print(f"    step{st:5d} best={best:.4f} {time.time()-t0:.0f}s",flush=True)
    
    # Eval at distances
    model.eval(); accs={}
    for d in [4,64,256,1024,4096,8192,16384,32768,65536]:
        eb=64 if d<=4096 else 8; c=0;total=0
        for _ in range(4 if d<=4096 else 2):
            x,y=make_cf_batch(eb,d); x,y=x.to(device),y.to(device)
            log,_=model(x); c+=(log[:,-1,:].argmax(-1)==y[:,-1]).sum().item(); total+=eb
        accs[d]=c/total*100
    cf_results[name]={'best':best,'accs':accs,'params':n}
    print(f"    => best={best:.5f}",flush=True)
    del model; torch.cuda.empty_cache()

print(f"\n{'='*70}")
print(f"  CopyFirst Results")
print(f"{'='*70}")
dists=[4,64,256,1024,4096,8192,16384,32768,65536]
print(f"  {'Dist':>6} | " + " | ".join([f"{n:>14}" for n,_ in cf_models]))
print(f"  "+"-"*(8+16*len(cf_models)))
for d in dists:
    print(f"  {d:6d} | " + " | ".join([f"{cf_results[n]['accs'][d]:14.1f}" for n,_ in cf_models]))

# 汇总
print(f"\n  Summary:")
print(f"  {'Model':>14} | {'LM Loss':>8} | {'CF best':>8} | {'CF@131K':>8} | {'CF@65K':>8}")
print(f"  "+"-"*55)
for name,_ in cf_models:
    lm = lm_results.get(name,{}).get('eval','—')
    cf_best = cf_results[name]['best']
    far = sum(cf_results[name]['accs'][d] for d in [4096,8192,16384,32768])/4
    print(f"  {name:>14} | {lm if isinstance(lm,str) else f'{lm:8.4f}'} | {cf_best:8.5f} | {cf_results[name]['accs'].get(65536,0):7.1f}% | {far:7.1f}%")

print(f"\nDone.")
