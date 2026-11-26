"""
FRSM 尺度数 vs Loss 下降关系
固定: d_model=256, 2000条数据, 500步, lr=3e-4
变量: num_scales ∈ {1, 2, 3, 4, 6, 8}
"""
import os, sys, time, math, torch
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

class FRSM(nn.Module):
    def __init__(self, vocab_size, d_model, num_scales=4):
        super().__init__()
        self.d_model=d_model; self.ns=num_scales
        self.embed=nn.Embedding(vocab_size,d_model)
        self.inp=nn.Linear(d_model,d_model)
        self.W_forget=nn.ModuleList([nn.Linear(d_model*2,d_model) for _ in range(num_scales)])
        self.W_input=nn.ModuleList([nn.Linear(d_model*2,d_model) for _ in range(num_scales)])
        self.W_cand=nn.ModuleList([nn.Linear(d_model*2,d_model) for _ in range(num_scales)])
        for w in self.W_forget: nn.init.constant_(w.bias,1.0)
        for w in self.W_input: nn.init.constant_(w.bias,-2.0)
        self.fusion=nn.Linear(d_model*num_scales,d_model)
        self.norm=nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,vocab_size)
    def forward(self,x):
        B,T=x.shape
        h=[torch.zeros(B,self.d_model,device=x.device) for _ in range(self.ns)]
        xe=self.embed(x); outs=[]
        for t in range(T):
            inp=self.inp(xe[:,t,:]); nh=[]
            for s in range(self.ns):
                if t%(2**s)==0:
                    c=torch.cat([h[s],inp],-1)
                    f=torch.sigmoid(self.W_forget[s](c)); i=torch.sigmoid(self.W_input[s](c))
                    nh.append(f*h[s]+i*torch.tanh(self.W_cand[s](c)))
                else: nh.append(h[s])
            h=nh
            fused=self.norm(self.fusion(torch.cat(h,-1)))
            outs.append(self.head(fused).unsqueeze(1))
        return torch.cat(outs,1)

dataset = PretrainDataset("minimind_data/pretrain_t2t_mini.jsonl", voc, max_len=256, max_lines=2000)
STEPS=500

def get_lr(opt,w,t):
    def f(s):
        if s<w: return s/max(1,w)
        p=(s-w)/max(1,t-w); return max(0.0,0.5*(1.0+math.cos(math.pi*p)))
    return torch.optim.lr_scheduler.LambdaLR(opt,f)

scale_configs = [1, 2, 3, 4, 6, 8]

print(f"{'='*70}")
print(f"  FRSM: num_scales vs Loss (d_model=256, 500 steps)")
print(f"{'='*70}")

results = {}
for ns in scale_configs:
    bs = max(2, 8 // ns) if ns > 2 else 8  # 调batch防OOM
    loader = DataLoader(dataset, batch_size=bs, shuffle=True,
                       collate_fn=PretrainDataset.collate_fn, drop_last=True)
    torch.manual_seed(42)
    model = FRSM(vs, 256, ns).to(device)
    n = sum(p.numel() for p in model.parameters())
    opt = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9,0.95))
    sch = get_lr(opt, 50, STEPS)

    model.train(); step=0; best=float('inf'); t0=time.time()
    hist=[]; data_iter=iter(loader)
    print(f"\n  scales={ns} | {n:,} params | bs={bs} | periods={[2**i for i in range(ns)]}", flush=True)

    while step < STEPS:
        try: x,t = next(data_iter)
        except StopIteration: data_iter=iter(loader); x,t=next(data_iter)
        x,t = x.to(device),t.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1,vs), t.reshape(-1), ignore_index=0)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sch.step()
        step+=1
        if loss.item()<best: best=loss.item()
        if step%50==0: hist.append((step, loss.item()))
        if step%100==0:
            print(f"    step{step:4d} loss={loss.item():.4f} best={best:.4f} {time.time()-t0:.0f}s", flush=True)

    # Eval
    model.eval()
    with torch.no_grad():
        tl=0; tt=0
        for x,t in loader:
            x,t=x.to(device),t.to(device)
            logits=model(x)
            l=F.cross_entropy(logits.reshape(-1,vs),t.reshape(-1),ignore_index=0,reduction='sum')
            tl+=l.item(); tt+=(t!=0).sum().item()
        el=tl/tt; ep=math.exp(el) if el<20 else 99999

    results[ns] = {'params':n, 'best':best, 'eval_loss':el, 'eval_ppl':ep, 'hist':hist, 'time':time.time()-t0}
    print(f"    => eval_loss={el:.4f} ppl={ep:.2f} time={time.time()-t0:.0f}s", flush=True)
    del model; torch.cuda.empty_cache()

# ============================================================
# 汇总
# ============================================================
print(f"\n{'='*70}")
print(f"  Summary: num_scales vs Loss")
print(f"{'='*70}")
print(f"  {'Scales':>7} | {'Periods':>20} | {'Params':>10} | {'Best':>8} | {'Eval':>8} | {'PPL':>8} | {'ΔLoss':>8}")
print(f"  "+"-"*80)
prev=None
for ns in scale_configs:
    r=results[ns]
    periods=str([2**i for i in range(ns)])
    dl=f"{r['eval_loss']-prev:+.4f}" if prev is not None else "—"
    print(f"  {ns:7d} | {periods:>20} | {r['params']:10,} | {r['best']:8.4f} | {r['eval_loss']:8.4f} | {r['eval_ppl']:8.2f} | {dl:>8}")
    prev=r['eval_loss']

# Loss 曲线对比
print(f"\n  Loss Curves (every 50 steps):")
header=f"  {'Step':>5}" + "".join([f" | S={ns:>2}" for ns in scale_configs])
print(header)
print(f"  "+"-"*(6+8*len(scale_configs)))
max_hist=max(len(results[ns]['hist']) for ns in scale_configs)
for i in range(max_hist):
    step=results[scale_configs[0]]['hist'][i][0]
    vals=[]
    for ns in scale_configs:
        if i<len(results[ns]['hist']):
            vals.append(f"{results[ns]['hist'][i][1]:6.3f}")
        else:
            vals.append(f"{'—':>6}")
    print(f"  {step:5d}" + "".join([f" | {v:>6}" for v in vals]))

# 边际收益
print(f"\n  Marginal Return (each additional scale):")
for i in range(1, len(scale_configs)):
    prev_ns=scale_configs[i-1]; cur_ns=scale_configs[i]
    prev_l=results[prev_ns]['eval_loss']; cur_l=results[cur_ns]['eval_loss']
    delta=prev_l-cur_l
    pct=delta/prev_l*100 if prev_l>0 else 0
    bar="█"*int(abs(pct)*5) if pct>0 else ""
    print(f"    {prev_ns}→{cur_ns} scales: Δ={delta:+.4f} ({pct:+.1f}%) {bar}")

# 最优尺度
best_ns=min(scale_configs, key=lambda ns: results[ns]['eval_loss'])
print(f"\n  Best: scales={best_ns} (eval_loss={results[best_ns]['eval_loss']:.4f})")
print(f"\nDone.")
