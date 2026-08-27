"""FRSM v4 修复版"""
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
VOCAB_CF = 32; END_CF = 0; IGN_CF = 1; H_CF = 128

class FRSM_AdaptiveRes(nn.Module):
    def __init__(self, vocab_size, d_model, num_scales=4):
        super().__init__()
        self.H=d_model; self.ns=num_scales; self.vs=vocab_size
        self.embed=nn.Embedding(vocab_size,d_model)
        self.inp=nn.Linear(d_model,d_model); self.inp_norm=nn.LayerNorm(d_model)
        self.Wf=nn.ModuleList([nn.Linear(d_model*2,d_model) for _ in range(num_scales)])
        self.Wi=nn.ModuleList([nn.Linear(d_model*2,d_model) for _ in range(num_scales)])
        self.Wc=nn.ModuleList([nn.Linear(d_model*2,d_model) for _ in range(num_scales)])
        for w in self.Wf: nn.init.constant_(w.bias,1.0)
        for w in self.Wi: nn.init.constant_(w.bias,-2.0)
        # 自适应保留率: 每尺度每步动态计算 (依赖输入)
        self.W_alpha=nn.ModuleList([nn.Linear(d_model*2,1) for _ in range(num_scales)])
        for w in self.W_alpha: nn.init.constant_(w.bias,2.0)  # sigmoid(2)=0.88
        self.state_norm=nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_scales)])
        self.fusion=nn.Linear(d_model*num_scales,d_model); self.fusion_norm=nn.LayerNorm(d_model)
        self.W_beta=nn.Linear(d_model*2,1); nn.init.constant_(self.W_beta.bias,0.0)
        self.head=nn.Linear(d_model,vocab_size)

    def forward(self, x, hp=None):
        B,T=x.shape
        if hp is None: h=[torch.zeros(B,self.H,device=x.device) for _ in range(self.ns)]
        else: h=[s.clone() for s in hp]
        xe=self.embed(x); outs=[]
        for t in range(T):
            inp=self.inp_norm(self.inp(xe[:,t,:])); nh=[]
            for s in range(self.ns):
                if t%(2**s)==0:
                    c=torch.cat([h[s],inp],-1)
                    f=torch.sigmoid(self.Wf[s](c)); i=torch.sigmoid(self.Wi[s](c))
                    candidate=f*h[s]+i*torch.tanh(self.Wc[s](c))
                    alpha=torch.sigmoid(self.W_alpha[s](c)).squeeze(-1).unsqueeze(-1)  # (B,1)
                    h_new=self.state_norm[s](alpha*h[s]+(1-alpha)*candidate)
                    nh.append(h_new)
                else: nh.append(h[s])
            h=nh
            ssm_out=self.fusion_norm(self.fusion(torch.cat(h,-1)))
            cat=torch.cat([ssm_out,inp],-1)
            beta=torch.sigmoid(self.W_beta(cat)).squeeze(-1).unsqueeze(-1)
            out=beta*ssm_out+(1-beta)*inp
            outs.append(self.head(out).unsqueeze(1))
        return torch.cat(outs,1), h

class FRSM_Orig(nn.Module):
    def __init__(self, vocab_size, d_model, num_scales=4):
        super().__init__()
        self.H=d_model; self.ns=num_scales
        self.embed=nn.Embedding(vocab_size,d_model); self.inp=nn.Linear(d_model,d_model)
        self.Wf=nn.ModuleList([nn.Linear(d_model*2,d_model) for _ in range(num_scales)])
        self.Wi=nn.ModuleList([nn.Linear(d_model*2,d_model) for _ in range(num_scales)])
        self.Wc=nn.ModuleList([nn.Linear(d_model*2,d_model) for _ in range(num_scales)])
        for w in self.Wf: nn.init.constant_(w.bias,1.0)
        for w in self.Wi: nn.init.constant_(w.bias,-2.0)
        self.fusion=nn.Linear(d_model*num_scales,d_model); self.norm=nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,vocab_size)
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

# ============================================================
dataset = PretrainDataset("minimind_data/pretrain_t2t_mini.jsonl", voc, max_len=256, max_lines=2000)

def get_lr(opt,w,t):
    def f(s):
        if s<w: return s/max(1,w)
        p=(s-w)/max(1,t-w); return max(0.0,0.5*(1.0+math.cos(math.pi*p)))
    return torch.optim.lr_scheduler.LambdaLR(opt,f)

def make_cf(bs,nl):
    t=torch.randint(2,VOCAB_CF,(bs,)); n=torch.randint(2,VOCAB_CF,(bs,nl))
    e=torch.full((bs,1),END_CF,dtype=torch.long)
    x=torch.cat([t.unsqueeze(1),n,e],1); y=torch.full_like(x,IGN_CF); y[:,-1]=t
    return x,y

# ============================================================
# LM 实验
# ============================================================
print(f"{'='*70}")
print(f"  Part 1: LM Loss (500 steps)")
print(f"{'='*70}")

lm_results = {}
for name, ModelClass in [("Orig-4sc", FRSM_Orig), ("AdaptiveRes-4sc", FRSM_AdaptiveRes)]:
    loader = DataLoader(dataset, batch_size=4, shuffle=True,
                       collate_fn=PretrainDataset.collate_fn, drop_last=True)
    torch.manual_seed(42)
    model = ModelClass(vs, 256, 4).to(device)
    n = sum(p.numel() for p in model.parameters())
    opt = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9,0.95))
    sch = get_lr(opt, 50, 500)
    model.train(); best = float('inf'); t0 = time.time()
    data_iter = iter(loader)
    print(f"\n  [{name}] {n:,} params", flush=True)
    for step in range(500):
        try: x, t = next(data_iter)
        except StopIteration: data_iter = iter(loader); x, t = next(data_iter)
        x, t = x.to(device), t.to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vs), t.reshape(-1), ignore_index=0)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if loss.item() < best: best = loss.item()
        if (step+1) % 100 == 0:
            print(f"    step{step+1:4d} loss={loss.item():.4f} best={best:.4f} {time.time()-t0:.0f}s", flush=True)
    model.eval()
    with torch.no_grad():
        tl=0; tt=0
        for x, t in loader:
            x, t = x.to(device), t.to(device)
            logits, _ = model(x)
            l = F.cross_entropy(logits.reshape(-1, vs), t.reshape(-1), ignore_index=0, reduction='sum')
            tl += l.item(); tt += (t != 0).sum().item()
        el = tl/tt; ep = math.exp(el) if el < 20 else 99999
    lm_results[name] = (el, best)
    print(f"    => eval={el:.4f} ppl={ep:.2f}", flush=True)
    del model; torch.cuda.empty_cache()

# ============================================================
# CopyFirst 实验
# ============================================================
print(f"\n{'='*70}")
print(f"  Part 2: CopyFirst (2500 steps)")
print(f"{'='*70}")

cf_results = {}
for name, ModelClass in [("Orig-4sc", FRSM_Orig), ("AdaptiveRes-4sc", FRSM_AdaptiveRes)]:
    torch.manual_seed(42)
    model = ModelClass(VOCAB_CF, H_CF, 4).to(device)
    n = sum(p.numel() for p in model.parameters())
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 2500)
    model.train(); best = float('inf'); t0 = time.time()
    print(f"\n  [{name}] {n:,} params", flush=True)
    for st in range(1, 2501):
        x, y = make_cf(64, random.randint(4, 64)); x, y = x.to(device), y.to(device)
        log, _ = model(x); loss = F.cross_entropy(log[:, -1, :], y[:, -1], ignore_index=IGN_CF)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if loss.item() < best: best = loss.item()
        if st % 500 == 0:
            print(f"    step{st:5d} best={best:.5f} {time.time()-t0:.0f}s", flush=True)
    model.eval(); accs = {}
    for d in [4, 64, 1024, 4096, 16384, 32768, 65536]:
        eb = 64 if d <= 4096 else 8; c = 0; total = 0
        for _ in range(4 if d <= 4096 else 2):
            x, y = make_cf(eb, d); x, y = x.to(device), y.to(device)
            log, _ = model(x); c += (log[:, -1, :].argmax(-1) == y[:, -1]).sum().item(); total += eb
        accs[d] = c / total * 100
    cf_results[name] = (best, accs)
    print(f"    => best={best:.5f}", flush=True)
    del model; torch.cuda.empty_cache()

# ============================================================
# 汇总
# ============================================================
print(f"\n{'='*70}")
print(f"  Final Comparison")
print(f"{'='*70}")
print(f"  {'Model':>16} | {'LM Loss':>8} | {'CF best':>8} | {'CF@4K':>7} | {'CF@16K':>7} | {'CF@32K':>7} | {'CF@65K':>7}")
print(f"  " + "-" * 72)
for name in ["Orig-4sc", "AdaptiveRes-4sc"]:
    lm_el, lm_best = lm_results[name]
    cf_best, cf_accs = cf_results[name]
    print(f"  {name:>16} | {lm_el:8.4f} | {cf_best:8.5f} | {cf_accs.get(4096,0):6.1f}% | {cf_accs.get(16384,0):6.1f}% | {cf_accs.get(32768,0):6.1f}% | {cf_accs.get(65536,0):6.1f}%")

print(f"\nDone.")
