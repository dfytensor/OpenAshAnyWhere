"""
FRSM v3 改进消融: 4个方向
1. α调优(0.5/0.7/0.88/0.95)
2. Gated Fusion (学习每尺度贡献权重)
3. Content Query (输入查询尺度状态)
4. Expansion Factor (2x/3x/4x)
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
    def __init__(self, d_model, alpha_init=2.0, expansion=2.0):
        super().__init__()
        hd = int(d_model * expansion)
        self.proj_h = nn.Linear(d_model, hd, bias=False)  # state → hd
        self.proj_inp = nn.Linear(d_model, hd, bias=False)  # input → hd
        self.W_forget = nn.Linear(hd + hd, hd)
        self.W_input  = nn.Linear(hd + hd, hd)
        self.W_cand   = nn.Linear(hd + hd, hd)
        self.proj_out = nn.Linear(hd, d_model)
        nn.init.constant_(self.W_forget.bias, 1.0)
        nn.init.constant_(self.W_input.bias, -2.0)
        self.register_buffer('alpha', torch.sigmoid(torch.tensor(alpha_init)))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_prev, inp):
        h_proj = self.proj_h(h_prev)   # (B, hd)
        i_proj = self.proj_inp(inp)    # (B, hd)
        c = torch.cat([h_proj, i_proj], -1)
        f = torch.sigmoid(self.W_forget(c))
        i = torch.sigmoid(self.W_input(c))
        cand_hd = f * h_proj + i * torch.tanh(self.W_cand(c))
        cand = self.proj_out(cand_hd)   # (B, d_model)
        return self.norm(self.alpha * h_prev + (1 - self.alpha) * cand)

class FRSM_Base(nn.Module):
    """基础 v3 Residual (基线)"""
    def __init__(self, vs, H=256, ns=4, alpha_init=2.0, expansion=2.0):
        super().__init__()
        self.H=H; self.ns=ns
        self.embed=nn.Embedding(vs,H); self.inp=nn.Linear(H,H); self.inp_norm=nn.LayerNorm(H)
        self.scales=nn.ModuleList([ScaleRecurrentBlock(H,alpha_init,expansion) for _ in range(ns)])
        self.scale_proj=nn.ModuleList([nn.Linear(H,H) for _ in range(ns)])
        self.fusion=nn.Linear(H*ns,H); self.fusion_norm=nn.LayerNorm(H)
        self.head=nn.Linear(H,vs)
    def forward(self,x,hp=None):
        B,T=x.shape
        if hp is None: h=[torch.zeros(B,self.H,device=x.device) for _ in range(self.ns)]
        else: h=[s.clone() for s in hp]
        xe=self.embed(x); outs=[]
        for t in range(T):
            inp=self.inp_norm(self.inp(xe[:,t,:])); nh=[]
            for s in range(self.ns):
                if t%(2**s)==0: nh.append(self.scales[s](h[s],inp))
                else: nh.append(h[s])
            h=nh
            fused=self.fusion(torch.cat(h,-1))
            fused=fused+sum(self.scale_proj[s](h[s]) for s in range(self.ns))/self.ns
            out=self.fusion_norm(fused+inp)
            outs.append(self.head(out).unsqueeze(1))
        return torch.cat(outs,1),h

class FRSM_GatedFusion(nn.Module):
    """改进1: 学每尺度贡献权重"""
    def __init__(self, vs, H=256, ns=4, alpha_init=2.0):
        super().__init__()
        self.H=H; self.ns=ns
        self.embed=nn.Embedding(vs,H); self.inp=nn.Linear(H,H); self.inp_norm=nn.LayerNorm(H)
        self.scales=nn.ModuleList([ScaleRecurrentBlock(H,alpha_init) for _ in range(ns)])
        # Gated fusion: 输入决定每尺度的读取权重
        self.gate_proj = nn.Linear(H, ns)  # input -> per-scale gate
        self.fusion = nn.Linear(H*ns, H); self.fusion_norm = nn.LayerNorm(H)
        self.head = nn.Linear(H, vs)

    def forward(self, x, hp=None):
        B,T=x.shape
        if hp is None: h=[torch.zeros(B,self.H,device=x.device) for _ in range(self.ns)]
        else: h=[s.clone() for s in hp]
        xe=self.embed(x); outs=[]
        for t in range(T):
            inp=self.inp_norm(self.inp(xe[:,t,:])); nh=[]
            for s in range(self.ns):
                if t%(2**s)==0: nh.append(self.scales[s](h[s],inp))
                else: nh.append(h[s])
            h=nh
            # Gated: 输入决定从每个尺度读多少
            gates = torch.softmax(self.gate_proj(inp), dim=-1)  # (B, ns)
            gated_h = [h[s] * gates[:, s:s+1] for s in range(self.ns)]
            fused = self.fusion(torch.cat(gated_h, -1))
            out = self.fusion_norm(fused + inp)
            outs.append(self.head(out).unsqueeze(1))
        return torch.cat(outs, 1), h

class FRSM_ContentQuery(nn.Module):
    """改进2: 输入查询尺度状态 (mini attention)"""
    def __init__(self, vs, H=256, ns=4, alpha_init=2.0):
        super().__init__()
        self.H=H; self.ns=ns
        self.embed=nn.Embedding(vs,H); self.inp=nn.Linear(H,H); self.inp_norm=nn.LayerNorm(H)
        self.scales=nn.ModuleList([ScaleRecurrentBlock(H,alpha_init) for _ in range(ns)])
        # Content query: Q=inp, K=h[s], V=h[s]
        self.W_q = nn.Linear(H, H)
        self.W_k = nn.Linear(H, H)
        self.W_v = nn.Linear(H, H)
        self.fusion = nn.Linear(H*ns, H)  # still keep fusion
        self.fusion_norm = nn.LayerNorm(H)
        self.head = nn.Linear(H, vs)

    def forward(self, x, hp=None):
        B,T=x.shape
        if hp is None: h=[torch.zeros(B,self.H,device=x.device) for _ in range(self.ns)]
        else: h=[s.clone() for s in hp]
        xe=self.embed(x); outs=[]
        for t in range(T):
            inp=self.inp_norm(self.inp(xe[:,t,:])); nh=[]
            for s in range(self.ns):
                if t%(2**s)==0: nh.append(self.scales[s](h[s],inp))
                else: nh.append(h[s])
            h=nh
            # Content query: Q·K for each scale
            q = self.W_q(inp)  # (B, H)
            scores = []
            values = []
            for s in range(self.ns):
                k = self.W_k(h[s])  # (B, H)
                v = self.W_v(h[s])  # (B, H)
                scores.append((q * k).sum(dim=-1, keepdim=True))  # (B, 1)
                values.append(v)
            attn = torch.softmax(torch.cat(scores, dim=-1), dim=-1)  # (B, ns)
            # Weighted combination
            read = sum(values[s] * attn[:, s:s+1] for s in range(self.ns))
            # Also keep fusion for rich features
            fused = self.fusion(torch.cat(h, -1))
            out = self.fusion_norm(fused + read + inp)
            outs.append(self.head(out).unsqueeze(1))
        return torch.cat(outs, 1), h

# ============================================================
dataset = PretrainDataset("minimind_data/pretrain_t2t_mini.jsonl", voc, max_len=256, max_lines=2000)

def train_one(model, name, steps=500):
    loader = DataLoader(dataset, batch_size=4, shuffle=True,
                       collate_fn=PretrainDataset.collate_fn, drop_last=True)
    torch.manual_seed(42); model = model.to(device)
    n = sum(p.numel() for p in model.parameters())
    opt = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
    def lrs(opt,w,t):
        def f(s):
            if s<w: return s/max(1,w)
            p=(s-w)/max(1,t-w); return max(0.0,0.5*(1.0+math.cos(math.pi*p)))
        return torch.optim.lr_scheduler.LambdaLR(opt,f)
    sch=lrs(opt,50,steps); model.train(); best=float('inf'); di=iter(loader)
    print(f"\n  [{name}] {n:,}p", flush=True)
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
    print(f"    => eval={el:.4f} ppl={math.exp(el):.2f}",flush=True)
    del model; torch.cuda.empty_cache()
    return el, best

# ============================================================
print(f"{'='*70}")
print(f"  FRSM v3 Improvement Directions")
print(f"{'='*70}")

results = {}

# 1. α 调优
print(f"\n--- 1. Alpha Tuning ---")
for a_logit in [0.0, 1.0, 2.0, 3.0]:  # sigmoid: 0.5, 0.73, 0.88, 0.95
    a = torch.sigmoid(torch.tensor(a_logit)).item()
    el, best = train_one(FRSM_Base(vs, 256, 4, alpha_init=a_logit), f"α={a:.2f}")
    results[f"α={a:.2f}"] = el

# 2. Gated Fusion
print(f"\n--- 2. Gated Fusion ---")
el, best = train_one(FRSM_GatedFusion(vs, 256, 4), "GatedFusion")
results["GatedFusion"] = el

# 3. Content Query
print(f"\n--- 3. Content Query ---")
el, best = train_one(FRSM_ContentQuery(vs, 256, 4), "ContentQuery")
results["ContentQuery"] = el

# 4. Expansion Factor
print(f"\n--- 4. Expansion Factor ---")
for exp in [3.0, 4.0]:
    el, best = train_one(FRSM_Base(vs, 256, 4, alpha_init=2.0, expansion=exp), f"exp={exp:.1f}x")
    results[f"exp={exp:.1f}x"] = el

# baseline
el, best = train_one(FRSM_Base(vs, 256, 4), "Baseline(α=0.88)")
results["Baseline"] = el

# ============================================================
print(f"\n{'='*70}")
print(f"  Summary")
print(f"{'='*70}")
print(f"  {'Variant':>20} | {'Eval Loss':>10} | {'PPL':>8} | {'Δ vs Baseline':>13}")
print(f"  "+"-"*55)
bl = results["Baseline"]
for k in ["Baseline", "α=0.50", "α=0.73", "α=0.88", "α=0.95", "GatedFusion", "ContentQuery", "exp=3.0x", "exp=4.0x"]:
    v = results[k]
    delta = v - bl
    print(f"  {k:>20} | {v:10.4f} | {math.exp(v):8.2f} | {delta:+13.4f}")

best_k = min(results, key=results.get)
print(f"\n  Best: {best_k} (loss={results[best_k]:.4f})")
print(f"\nDone.")
