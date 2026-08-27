"""
FRSM 规模 vs Loss 极限 快速实验
固定: 2000条数据, 500步, lr=3e-4, bs=4
变量: d_model ∈ {64, 128, 256, 512}, num_scales=4
"""
import os, sys, time, math, torch, json
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
        self.critical_reg_coeff=0.01
    
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

# 固定数据
dataset = PretrainDataset("minimind_data/pretrain_t2t_mini.jsonl", voc, max_len=256, max_lines=2000)
print(f"Dataset: {len(dataset)} samples, seq_len<=256", flush=True)

def get_lr(opt, warmup, total):
    def f(s):
        if s<warmup: return s/max(1,warmup)
        p=(s-warmup)/max(1,total-warmup)
        return max(0.0,0.5*(1.0+math.cos(math.pi*p)))
    return torch.optim.lr_scheduler.LambdaLR(opt,f)

STEPS = 500
configs = [
    (64,  4, 8),   # d_model, scales, batch
    (128, 4, 8),
    (256, 4, 4),
    (512, 4, 2),
]

print(f"\n{'='*60}")
print(f"  FRSM Scaling: d_model vs Loss (500 steps, 2K data)")
print(f"{'='*60}")

results = []
for d_model, ns, bs in configs:
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, 
                       collate_fn=PretrainDataset.collate_fn, drop_last=True)
    
    torch.manual_seed(42)
    model = FRSM(vs, d_model, ns).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    
    opt = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9,0.95))
    sch = get_lr(opt, 50, STEPS)
    
    model.train()
    step=0; best=float('inf'); t0=time.time()
    data_iter=iter(loader)
    
    print(f"\n  d_model={d_model:4d} | {n_params:>10,} params | bs={bs}", flush=True)
    
    while step < STEPS:
        try: x,t = next(data_iter)
        except StopIteration: data_iter=iter(loader); x,t=next(data_iter)
        x,t = x.to(device), t.to(device)
        
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1,vs), t.reshape(-1), ignore_index=0)
        
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sch.step()
        
        step+=1
        if loss.item()<best: best=loss.item()
        
        if step%100==0:
            print(f"    step{step:4d} loss={loss.item():.4f} best={best:.4f} {time.time()-t0:.0f}s", flush=True)
    
    # 最终 eval
    model.eval()
    with torch.no_grad():
        total_l=0; total_t=0
        for x,t in loader:
            x,t=x.to(device),t.to(device)
            logits=model(x)
            l=F.cross_entropy(logits.reshape(-1,vs),t.reshape(-1),ignore_index=0,reduction='sum')
            total_l+=l.item(); total_t+=(t!=0).sum().item()
        eval_loss=total_l/total_t
        eval_ppl=math.exp(eval_loss) if eval_loss<20 else 99999
    
    elapsed=time.time()-t0
    results.append((d_model, n_params, best, eval_loss, eval_ppl, elapsed))
    print(f"    => eval_loss={eval_loss:.4f} eval_ppl={eval_ppl:.2f} time={elapsed:.0f}s", flush=True)
    del model; torch.cuda.empty_cache()

# 汇总
print(f"\n{'='*60}")
print(f"  Summary: FRSM Scaling Law")
print(f"{'='*60}")
print(f"  {'d_model':>8} | {'Params':>10} | {'Best Train':>10} | {'Eval Loss':>10} | {'Eval PPL':>10} | {'Time(s)':>8} | {'ΔLoss/Δparams':>14}")
print(f"  "+"-"*82)
prev_loss=None; prev_params=None
for dm,np,bt,el,ep,ti in results:
    dl=""
    if prev_loss is not None and prev_params>0:
        delta_l=el-prev_loss
        delta_p=np-prev_params
        rate=delta_l/delta_p*1e6  # loss per million params
        dl=f"{rate:+.3f}/M"
    print(f"  {dm:8d} | {np:10,} | {bt:10.4f} | {el:10.4f} | {ep:10.2f} | {ti:8.0f} | {dl:>14}")
    prev_loss=el; prev_params=np

# Scaling law 拟合
import math as m
params_log=[m.log10(r[1]) for r in results]
loss_vals=[r[3] for r in results]
if len(results)>=3:
    # 线性回归: loss = a * log10(params) + b
    n=len(results)
    sx=sum(params_log); sy=sum(loss_vals)
    sxx=sum(x*x for x in params_log); sxy=sum(x*y for x,y in zip(params_log,loss_vals))
    a=(n*sxy-sx*sy)/(n*sxx-sx*sx)
    b=(sy-a*sx)/n
    print(f"\n  Scaling law: loss ≈ {a:.3f} × log10(params) + {b:.3f}")
    
    # 外推
    for target_p in [50e6, 100e6, 500e6]:
        pred = a * m.log10(target_p) + b
        print(f"    {target_p/1e6:.0f}M params → predicted loss ≈ {pred:.2f} (PPL≈{m.exp(pred):.1f})")

print(f"\nDone.")
