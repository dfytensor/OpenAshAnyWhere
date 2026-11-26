"""单独跑 CopyFirst 评估"""
import os, sys, math, torch, random
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, 'F:/OpenASH2605')
device = torch.device("cuda")
VOCAB_CF=32; END_CF=0; IGN_CF=1; H_CF=128

# 模型定义 (同 adaptive_test.py)
class FRSM_AdaptiveRes(nn.Module):
    def __init__(self, vocab_size, d_model, num_scales=4):
        super().__init__()
        self.H=d_model; self.ns=num_scales
        self.embed=nn.Embedding(vocab_size,d_model)
        self.inp=nn.Linear(d_model,d_model); self.inp_norm=nn.LayerNorm(d_model)
        self.Wf=nn.ModuleList([nn.Linear(d_model*2,d_model) for _ in range(num_scales)])
        self.Wi=nn.ModuleList([nn.Linear(d_model*2,d_model) for _ in range(num_scales)])
        self.Wc=nn.ModuleList([nn.Linear(d_model*2,d_model) for _ in range(num_scales)])
        for w in self.Wf: nn.init.constant_(w.bias,1.0)
        for w in self.Wi: nn.init.constant_(w.bias,-2.0)
        self.W_alpha=nn.ModuleList([nn.Linear(d_model*2,1) for _ in range(num_scales)])
        for w in self.W_alpha: nn.init.constant_(w.bias,2.0)
        self.state_norm=nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_scales)])
        self.fusion=nn.Linear(d_model*num_scales,d_model); self.fusion_norm=nn.LayerNorm(d_model)
        self.W_beta=nn.Linear(d_model*2,1); nn.init.constant_(self.W_beta.bias,0.0)
        self.head=nn.Linear(d_model,vocab_size)
    def forward(self,x,hp=None):
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
                    alpha=torch.sigmoid(self.W_alpha[s](c)).squeeze(-1).unsqueeze(-1)
                    h_new=self.state_norm[s](alpha*h[s]+(1-alpha)*candidate)
                    nh.append(h_new)
                else: nh.append(h[s])
            h=nh
            ssm_out=self.fusion_norm(self.fusion(torch.cat(h,-1)))
            cat=torch.cat([ssm_out,inp],-1)
            beta=torch.sigmoid(self.W_beta(cat)).squeeze(-1).unsqueeze(-1)
            out=beta*ssm_out+(1-beta)*inp
            outs.append(self.head(out).unsqueeze(1))
        return torch.cat(outs,1),h

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

def make_cf(bs,nl):
    t=torch.randint(2,VOCAB_CF,(bs,)); n=torch.randint(2,VOCAB_CF,(bs,nl))
    e=torch.full((bs,1),END_CF,dtype=torch.long)
    x=torch.cat([t.unsqueeze(1),n,e],1); y=torch.full_like(x,IGN_CF); y[:,-1]=t
    return x,y

def train_and_eval(ModelClass, name):
    torch.manual_seed(42)
    model=ModelClass(VOCAB_CF,H_CF,4).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,2500)
    model.train(); best=float('inf')
    for st in range(1,2501):
        x,y=make_cf(64,random.randint(4,64)); x,y=x.to(device),y.to(device)
        log,_=model(x); loss=F.cross_entropy(log[:,-1,:],y[:,-1],ignore_index=IGN_CF)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sch.step()
        if loss.item()<best: best=loss.item()
    print(f"  {name}: best={best:.5f}",flush=True)
    model.eval(); accs={}
    for d in [4,64,1024,4096,16384,32768,65536]:
        eb=64 if d<=4096 else 8; c=0; total=0
        for _ in range(4 if d<=4096 else 2):
            x,y=make_cf(eb,d); x,y=x.to(device),y.to(device)
            log,_=model(x); c+=(log[:,-1,:].argmax(-1)==y[:,-1]).sum().item(); total+=eb
        accs[d]=c/total*100
    return best,accs

print("CopyFirst Eval:")
o_best,o_accs=train_and_eval(FRSM_Orig,"Orig-4sc")
a_best,a_accs=train_and_eval(FRSM_AdaptiveRes,"AdaptiveRes-4sc")

print(f"\n{'Dist':>6} | {'Orig':>8} | {'Adaptive':>8}")
print("-"*30)
for d in [4,64,1024,4096,16384,32768,65536]:
    print(f"{d:6d} | {o_accs[d]:7.1f}% | {a_accs[d]:7.1f}%")

# 也包含之前残差实验的数据
print(f"\nFull comparison (including v3 Residual):")
print(f"{'Model':>16} | {'CF best':>8} | {'CF@16K':>7} | {'CF@32K':>7} | {'CF@65K':>7}")
print("-"*55)
print(f"{'Orig-4sc':>16} | {o_best:8.5f} | {o_accs.get(16384,0):6.1f}% | {o_accs.get(32768,0):6.1f}% | {o_accs.get(65536,0):6.1f}%")
print(f"{'AdaptiveRes-4sc':>16} | {a_best:8.5f} | {a_accs.get(16384,0):6.1f}% | {a_accs.get(32768,0):6.1f}% | {a_accs.get(65536,0):6.1f}%")
print(f"{'v3-Residual-4sc':>16} | {'0.00022':>8} | {'87.5':>6}% | {'75.0':>6}% | {'68.8':>6}%  (from prev)")
print(f"{'LM Orig-4sc':>16} | {'5.70':>8} |   —    |   —    |   —     (LM loss)")
print(f"{'LM AdaptiveRes':>16} | {'6.00':>8} |   —    |   —    |   —     (LM loss)")
print(f"{'LM v3-Residual':>16} | {'6.05':>8} |   —    |   —    |   —     (LM loss)")
