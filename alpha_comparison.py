"""α=0.50 vs α=0.88 CopyFirst + LM 完整对比"""
import os, sys, math, torch, random, time
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
VOCAB_CF=32; END_CF=0; IGN_CF=1; H_CF=128

class ScaleBlock(nn.Module):
    def __init__(self, d_model, alpha_init=2.0):
        super().__init__()
        hd = int(d_model * 2)
        self.proj_h = nn.Linear(d_model, hd, bias=False)
        self.proj_inp = nn.Linear(d_model, hd, bias=False)
        self.Wf = nn.Linear(hd*2, hd); self.Wi = nn.Linear(hd*2, hd); self.Wc = nn.Linear(hd*2, hd)
        nn.init.constant_(self.Wf.bias, 1.0); nn.init.constant_(self.Wi.bias, -2.0)
        self.proj_out = nn.Linear(hd, d_model)
        self.register_buffer('alpha', torch.sigmoid(torch.tensor(alpha_init)))
        self.norm = nn.LayerNorm(d_model)
    def forward(self, h_prev, inp):
        hp=self.proj_h(h_prev); ip=self.proj_inp(inp); c=torch.cat([hp,ip],-1)
        f=torch.sigmoid(self.Wf(c)); i=torch.sigmoid(self.Wi(c))
        cand_hd=f*hp + i*torch.tanh(self.Wc(c))
        cand=self.proj_out(cand_hd)
        return self.norm(self.alpha*h_prev+(1-self.alpha)*cand)

class V3FRSM(nn.Module):
    def __init__(self, vs, H=256, ns=4, alpha_init=2.0):
        super().__init__()
        self.H=H; self.ns=ns
        self.embed=nn.Embedding(vs,H); self.inp=nn.Linear(H,H); self.inp_norm=nn.LayerNorm(H)
        self.scales=nn.ModuleList([ScaleBlock(H,alpha_init) for _ in range(ns)])
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

dataset=PretrainDataset("minimind_data/pretrain_t2t_mini.jsonl",voc,max_len=256,max_lines=2000)

def make_cf(bs,nl):
    t=torch.randint(2,VOCAB_CF,(bs,)); n=torch.randint(2,VOCAB_CF,(bs,nl))
    e=torch.full((bs,1),END_CF,dtype=torch.long)
    x=torch.cat([t.unsqueeze(1),n,e],1); y=torch.full_like(x,IGN_CF); y[:,-1]=t
    return x,y

def train_lm(model,name,steps=500):
    loader=DataLoader(dataset,batch_size=4,shuffle=True,collate_fn=PretrainDataset.collate_fn,drop_last=True)
    torch.manual_seed(42); model=model.to(device)
    opt=AdamW(model.parameters(),lr=3e-4,weight_decay=0.01,betas=(0.9,0.95))
    def lrs(o,w,t):
        def f(s):
            if s<w: return s/max(1,w)
            p=(s-w)/max(1,t-w); return max(0.0,0.5*(1.0+math.cos(math.pi*p)))
        return torch.optim.lr_scheduler.LambdaLR(o,f)
    sch=lrs(opt,50,steps); model.train(); best=float('inf'); di=iter(loader)
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
    model.train(); best=float('inf'); t0=time.time()
    for st in range(1,steps+1):
        x,y=make_cf(64,random.randint(4,64)); x,y=x.to(device),y.to(device)
        log,_=model(x); loss=F.cross_entropy(log[:,-1,:],y[:,-1],ignore_index=IGN_CF)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sch.step()
        if loss.item()<best: best=loss.item()
        if st%500==0: print(f"    step{st:5d} best={best:.5f} {time.time()-t0:.0f}s",flush=True)
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
print(f"{'='*70}")
print(f"  α=0.50 vs α=0.88: LM + CopyFirst 完整对比")
print(f"{'='*70}")

results={}
for alpha_init, a_val in [(0.0, 0.50), (2.0, 0.88)]:
    # LM
    el,lm_best=train_lm(V3FRSM(vs,256,4,alpha_init),f"α={a_val}")
    # CopyFirst
    cf_best,accs=train_cf(V3FRSM(VOCAB_CF,H_CF,4,alpha_init),f"α={a_val}")
    results[a_val]={'lm':el,'lm_best':lm_best,'cf_best':cf_best,'accs':accs}

# 汇总
print(f"\n{'='*70}")
print(f"  Final: α=0.50 vs α=0.88")
print(f"{'='*70}")
print(f"  {'Metric':>15} | {'α=0.50':>12} | {'α=0.88':>12} | {'v1 Orig':>12}")
print(f"  "+"-"*55)
print(f"  {'LM Loss':>15} | {results[0.50]['lm']:12.4f} | {results[0.88]['lm']:12.4f} | {'5.70':>12}")
print(f"  {'LM PPL':>15} | {math.exp(results[0.50]['lm']):12.2f} | {math.exp(results[0.88]['lm']):12.2f} | {'299':>12}")
print(f"  {'CF best_loss':>15} | {results[0.50]['cf_best']:12.5f} | {results[0.88]['cf_best']:12.5f} | {'0.00027':>12}")
for d in [4, 1024, 4096, 16384, 32768, 65536]:
    a50=results[0.50]['accs'].get(d,0)
    a88=results[0.88]['accs'].get(d,0)
    o4={'4':100,'1024':100,'4096':100,'16384':94,'32768':75,'65536':6}.get(str(d),'—')
    print(f"  {'CF@{:<5}'.format(d):>15} | {f'{a50:.1f}%':>12} | {f'{a88:.1f}%':>12} | {f'{o4}':>12}")

print(f"\nDone.")
