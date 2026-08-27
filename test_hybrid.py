"""HybridFRSM 测试: CopyFirst + LM"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
device = torch.device("cuda")

# ============================================================
# HybridFRSM adapted for LM (add embed + output head)
# ============================================================
class SlowScaleCell(nn.Module):
    def __init__(self, num_slow, d_model):
        super().__init__()
        self.num_slow = num_slow; self.d_model = d_model
        self.W_forget = nn.Parameter(torch.empty(num_slow, d_model, 2*d_model))
        self.b_forget = nn.Parameter(torch.empty(num_slow, d_model))
        self.W_input  = nn.Parameter(torch.empty(num_slow, d_model, 2*d_model))
        self.b_input  = nn.Parameter(torch.empty(num_slow, d_model))
        self.W_cand   = nn.Parameter(torch.empty(num_slow, d_model, 2*d_model))
        self.b_cand   = nn.Parameter(torch.empty(num_slow, d_model))
        d_hidden = max(d_model//4, 1)
        self.gate_W1 = nn.Parameter(torch.empty(num_slow, d_hidden, 2*d_model))
        self.gate_b1 = nn.Parameter(torch.empty(num_slow, d_hidden))
        self.gate_W2 = nn.Parameter(torch.empty(num_slow, 1, d_hidden))
        self.gate_b2 = nn.Parameter(torch.empty(num_slow, 1))
        self._init_weights()
    def _init_weights(self):
        for p in [self.W_forget,self.W_input,self.W_cand,self.gate_W1,self.gate_W2]:
            for s in range(self.num_slow): nn.init.kaiming_uniform_(p[s], a=math.sqrt(5))
        for p in [self.b_forget,self.b_input,self.b_cand,self.gate_b1,self.gate_b2]: nn.init.zeros_(p)
        nn.init.constant_(self.b_forget, 1.0); nn.init.constant_(self.b_input, -2.0)
    def forward(self, x_t, h_prev):
        B,S = x_t.size(0),self.num_slow
        x_exp = x_t.unsqueeze(1).expand(-1,S,-1)
        gate_in = torch.cat([h_prev, x_exp], dim=-1)
        f = torch.sigmoid(torch.einsum('bnj,nij->bni', gate_in, self.W_forget) + self.b_forget)
        i = torch.sigmoid(torch.einsum('bnj,nij->bni', gate_in, self.W_input) + self.b_input)
        cand = torch.tanh(torch.einsum('bnj,nij->bni', gate_in, self.W_cand) + self.b_cand)
        candidate = f * h_prev + i * cand
        h1 = F.gelu(torch.einsum('bnj,nij->bni', gate_in, self.gate_W1) + self.gate_b1)
        alpha = torch.sigmoid(torch.einsum('bni,noi->bno', h1, self.gate_W2) + self.gate_b2)
        return alpha * candidate + (1-alpha) * h_prev

class HybridFRSM_LM(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_fast=3, num_slow=1, slow_freq=8):
        super().__init__()
        self.d_model=d_model; self.nf=num_fast; self.ns=num_slow; self.K=slow_freq
        self.embed = nn.Embedding(vocab_size, d_model)
        self.input_proj = nn.Linear(d_model, d_model)
        self.fast_proj = nn.Linear(d_model, num_fast*4*d_model)
        self.slow_cell = SlowScaleCell(num_slow, d_model)
        total = num_fast + num_slow
        self.fusion = nn.Linear(total*d_model, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.hard_infer = True
    def forward(self, x, h_prev=None):
        B,T = x.shape
        xe = self.input_proj(self.embed(x))  # (B,T,D)
        NF,NS,D,K = self.nf,self.ns,self.d_model,self.K
        # Fast: parallel scan
        fg = self.fast_proj(xe).reshape(B,T,NF,4,D)
        alpha=torch.sigmoid(fg[...,0,:]); f_f=torch.sigmoid(fg[...,1,:])
        i_f=torch.sigmoid(fg[...,2,:]); c_f=torch.tanh(fg[...,3,:])
        A=alpha*f_f+(1-alpha); B_f=alpha*i_f*c_f
        hf=[torch.zeros(B,NF,D,device=x.device) for _ in range(T)]
        h0=torch.zeros(B,NF,D,device=x.device)
        hf_cur=h0
        for t in range(T): hf_cur=A[:,t]*hf_cur+B_f[:,t]; hf[t]=hf_cur
        H_fast=torch.stack(hf,dim=1)
        # Slow:分段常数
        hs=torch.zeros(B,NS,D,device=x.device)
        H_slow=torch.zeros(B,T,NS,D,device=x.device,dtype=xe.dtype)
        prev=0
        for t in range(0,T,K):
            hs=self.slow_cell(xe[:,t,:],hs); H_slow[:,prev:t+1]=hs.unsqueeze(1); prev=t+1
        if prev<T: H_slow[:,prev:]=hs.unsqueeze(1)
        # Fusion
        H_all=torch.cat([H_fast,H_slow],dim=2).reshape(B,T,-1)
        fused=self.fusion_norm(self.fusion(H_all))
        return self.output_proj(fused)

# ============================================================
# CopyFirst test
# ============================================================
VCF=32; ECF=0; ICF=1; HCF=128
def mf(bs,nl):
    t=torch.randint(2,VCF,(bs,)); n=torch.randint(2,VCF,(bs,nl))
    e=torch.full((bs,1),ECF,dtype=torch.long)
    x=torch.cat([t.unsqueeze(1),n,e],1); y=torch.full_like(x,ICF); y[:,-1]=t
    return x,y

for name, model in [("HybridFRSM", HybridFRSM_LM(VCF,HCF,3,1,8))]:
    n=sum(p.numel() for p in model.parameters())
    torch.manual_seed(42); model=model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,2500)
    model.train(); best=float('inf'); t0=time.time()
    print(f"[{name}] {n:,} params",flush=True)
    for st in range(1,2501):
        x,y=mf(64,random.randint(4,64)); x,y=x.to(device),y.to(device)
        log=model(x); loss=F.cross_entropy(log[:,-1,:],y[:,-1],ignore_index=ICF)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step()
        if loss.item()<best: best=loss.item()
        if st%500==0: print(f"  step{st:5d} best={best:.5f} {time.time()-t0:.0f}s",flush=True)
    print(f"  best={best:.5f}",flush=True)
    model.eval()
    print(f"  {'Dist':>6} | Acc",flush=True)
    for d in [4,64,256,1024,4096,8192,16384,32768,65536]:
        eb=64 if d<=4096 else 8; c=0;total=0
        for _ in range(4 if d<=4096 else 2):
            x,y=mf(eb,d); x,y=x.to(device),y.to(device)
            log=model(x); c+=(log[:,-1,:].argmax(-1)==y[:,-1]).sum().item(); total+=eb
        print(f"  {d:6d} | {c/total*100:.1f}%",flush=True)
    del model; torch.cuda.empty_cache()

# Also test V6a Fast for direct comparison
from frsm_v6a_fast import FRSM_V6_Fast
print(f"\n[V6a-Fast] comparison",flush=True)
torch.manual_seed(42); m2=FRSM_V6_Fast(VCF,HCF,4).to(device)
n2=sum(p.numel() for p in m2.parameters())
opt2=torch.optim.AdamW(m2.parameters(),lr=1e-3,weight_decay=0.01)
sch2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,2500)
m2.train(); best2=float('inf'); t0=time.time()
print(f"  {n2:,} params",flush=True)
for st in range(1,2501):
    x,y=mf(64,random.randint(4,64)); x,y=x.to(device),y.to(device)
    log=m2(x); loss=F.cross_entropy(log[:,-1,:],y[:,-1],ignore_index=ICF)
    opt2.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(m2.parameters(),1.0); opt2.step(); sch2.step()
    if loss.item()<best2: best2=loss.item()
    if st%500==0: print(f"  step{st:5d} best={best2:.5f} {time.time()-t0:.0f}s",flush=True)
print(f"  best={best2:.5f}",flush=True)
m2.eval()
print(f"  {'Dist':>6} | Acc",flush=True)
for d in [4,64,256,1024,4096,8192,16384,32768,65536]:
    eb=64 if d<=4096 else 8; c=0;total=0
    for _ in range(4 if d<=4096 else 2):
        x,y=mf(eb,d); x,y=x.to(device),y.to(device)
        log=m2(x); c+=(log[:,-1,:].argmax(-1)==y[:,-1]).sum().item(); total+=eb
    print(f"  {d:6d} | {c/total*100:.1f}%",flush=True)

print("\nDone.")
