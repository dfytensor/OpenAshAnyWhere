"""
5架构 CopyFirst 对比: FRSM vs OpenASH vs WDLM vs Transformer vs LP-SSM
统一参数量, 统一训练, 统一评估
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
from einops import rearrange

device = torch.device("cuda")
VOCAB = 32; END = 0; IGNORE = 1; H = 128

# ============================================================
# 1. FRSM (2-scale gated)
# ============================================================
class MiniFRSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.H=H; self.ns=2
        self.embed=nn.Embedding(VOCAB,H); self.inp=nn.Linear(H,H)
        self.W_forget=nn.ModuleList([nn.Linear(H*2,H) for _ in range(2)])
        self.W_input=nn.ModuleList([nn.Linear(H*2,H) for _ in range(2)])
        self.W_cand=nn.ModuleList([nn.Linear(H*2,H) for _ in range(2)])
        for w in self.W_forget: nn.init.constant_(w.bias, 1.0)
        for w in self.W_input: nn.init.constant_(w.bias, -2.0)
        self.fusion=nn.Linear(H*2,H); self.ln=nn.LayerNorm(H)
        self.head=nn.Linear(H,VOCAB)
    def forward(self, x, hp=None):
        B,T=x.shape
        if hp is None: h=[torch.zeros(B,H,device=device) for _ in range(self.ns)]
        else: h=[s.clone() for s in hp]
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
            fused=self.ln(self.fusion(torch.cat(h,-1)))
            outs.append(self.head(fused).unsqueeze(1))
        return torch.cat(outs,1), h

# ============================================================
# 2. OpenASH (cummax)
# ============================================================
class MiniOpenASH(nn.Module):
    def __init__(self):
        super().__init__()
        self.H=H; self.heads=4; self.dh=H//self.heads
        self.embed=nn.Embedding(VOCAB,H); self.proj=nn.Linear(H,4*H,bias=False)
        self.gen_out=nn.Linear(5*H,H)
        self.a1=nn.Parameter(torch.tensor(0.5)); self.a2=nn.Parameter(torch.tensor(0.5)); self.a3=nn.Parameter(torch.tensor(0.5))
        self.ln=nn.LayerNorm(H); self.head=nn.Linear(H,VOCAB,bias=False)
    def forward(self,x,state=None):
        B,T=x.shape; h=self.embed(x)
        o=self.proj(h).view(B,T,4,self.heads,self.dh)
        a,b,c,d=[t.permute(0,3,1,2) for t in o.unbind(2)]
        if state is None: e,_=torch.cummax(c,2); sn=e[:,:,-1:,:]
        else: e,_=torch.cummax(torch.cat([state,c],2),2); e=e[:,:,1:,:]; sn=e[:,:,-1:,:]
        t1=a*b; t2=self.a1*b+self.a2*d; t3=a*(self.a3*e+d); t4=b*(c+e); t5=c*e
        cb=torch.cat([t1,t2,t3,t4,t5],-1).permute(0,2,1,3).reshape(B,T,-1)
        return self.head(self.ln(self.gen_out(cb))), sn

# ============================================================
# 3. WDLM-Neural
# ============================================================
class MiniWDLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.H=H
        self.embed=nn.Embedding(VOCAB,H)
        self.rot=nn.Linear(H,H,bias=False); self.amp=nn.Linear(H,H,bias=False); self.gate=nn.Linear(H,H,bias=False)
        self.cum_proj=nn.Linear(H,4*H); self.gen_out=nn.Linear(5*H,H)
        self.a1=nn.Parameter(torch.tensor(0.5)); self.a2=nn.Parameter(torch.tensor(0.5)); self.a3=nn.Parameter(torch.tensor(0.5))
        self.ln=nn.LayerNorm(H); self.head=nn.Linear(H,VOCAB,bias=False)
    def forward(self,x,state=None):
        B,T=x.shape; psi=self.embed(x)
        psi=psi*self.rot(psi)+torch.sigmoid(self.gate(psi))*self.amp(psi)+psi
        a,b,c,d=self.cum_proj(psi).chunk(4,-1)
        if state is None: e,_=torch.cummax(c,1); sn=e[:,-1:,:]
        else: e,_=torch.cummax(torch.cat([state,c],1),1); e=e[:,1:,:]; sn=e[:,-1:,:]
        t1=a*b; t2=self.a1*b+self.a2*d; t3=a*(self.a3*e+d); t4=b*(c+e); t5=c*e
        return self.head(self.ln(self.gen_out(torch.cat([t1,t2,t3,t4,t5],-1)))), sn

# ============================================================
# 4. Transformer
# ============================================================
class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(VOCAB,H)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(H,4,H*4,0.0,batch_first=True) for _ in range(2)])
        self.head=nn.Linear(H,VOCAB)
    def forward(self,x,hp=None):
        T=x.size(1); m=nn.Transformer.generate_square_subsequent_mask(T,device=device)
        return self.head(self.layers[1](self.layers[0](self.embed(x)*math.sqrt(H),src_mask=m),src_mask=m)),None

# ============================================================
# 5. LP-SSM (用户提供)
# ============================================================
class LogPeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, f_min=0.1, f_max=10.0, eps=1e-6):
        super().__init__()
        half_dim = d_model // 2
        log_f = torch.linspace(math.log(f_min), math.log(f_max), half_dim)
        self.register_buffer("freq", torch.exp(log_f))
        self.eps = eps
    def forward(self, positions):
        if positions.dim() == 1: positions = positions.unsqueeze(0)
        pos = positions.float().unsqueeze(-1)
        log_pos = torch.log(pos + self.eps)
        angles = log_pos * self.freq
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

class LogPeriodicDiagonalSSM(nn.Module):
    def __init__(self, d_model, d_state=32, num_scales=2, alpha=0.5,
                 base_omega=2.0, omega_spread=2.0, delta_init=1.0):
        super().__init__()
        assert d_model % num_scales == 0
        assert d_state % 2 == 0
        self.d_model=d_model; self.d_state=d_state; self.num_scales=num_scales
        self.head_dim=d_model//num_scales
        half_state=d_state//2
        k=torch.arange(1,half_state+1,dtype=torch.float32)
        log_k=torch.log(k)
        omegas=base_omega*(omega_spread**torch.arange(num_scales,dtype=torch.float32))
        lambda_pos=-alpha+1j*(omegas.unsqueeze(1)*log_k.unsqueeze(0))
        lambda_neg=-alpha-1j*(omegas.unsqueeze(1)*log_k.unsqueeze(0))
        Lambda=torch.zeros(num_scales,d_state,dtype=torch.complex64)
        Lambda[:,0::2]=lambda_pos; Lambda[:,1::2]=lambda_neg
        self.register_buffer("Lambda",Lambda)
        self.B=nn.Parameter(torch.randn(num_scales,d_state,self.head_dim,dtype=torch.cfloat)*0.1)
        self.C=nn.Parameter(torch.randn(num_scales,self.head_dim,d_state,dtype=torch.cfloat)*0.1)
        self.delta_log=nn.Parameter(torch.full((num_scales,),math.log(delta_init)))
    
    def _get_disc(self, s):
        delta=torch.exp(self.delta_log[s])
        Lambda=self.Lambda[s]
        Lambda_bar=torch.exp(delta*Lambda)
        B_bar_coef=(Lambda_bar-1.0)/Lambda
        return Lambda_bar, B_bar_coef.unsqueeze(-1)*self.B[s], self.C[s]
    
    def forward(self, x):
        B,L,_=x.shape
        x_scales=rearrange(x,'b l (s h)->b l s h',s=self.num_scales,h=self.head_dim)
        y_scales=[]
        for s in range(self.num_scales):
            Lambda_bar,B_bar,C_s=self._get_disc(s)
            steps=torch.arange(L,device=x.device)
            Lambda_bar_pow=Lambda_bar.unsqueeze(0)**steps.unsqueeze(1).to(torch.complex64)
            h=torch.einsum('id,ld,dj->lij', C_s, Lambda_bar_pow, B_bar)
            x_s=x_scales[:,:,s,:]
            x_fft=torch.fft.fft(x_s,n=2*L,dim=1)
            h_fft=torch.fft.fft(h,n=2*L,dim=0)
            y_fft=torch.einsum('fij,bfj->bfi', h_fft, x_fft)
            y_s=torch.fft.ifft(y_fft,n=2*L,dim=1)[:,:L,:].real
            y_scales.append(y_s.real)
        y=torch.stack(y_scales,dim=2)
        return rearrange(y,'b l s h->b l (s h)')

class LogPeriodicGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, A_g=0.2, omega_g=3.0, phi_g=0.0, eps=1e-6):
        super().__init__()
        d_ff=d_ff or d_model*4
        self.fc1=nn.Linear(d_model,d_ff,bias=False); self.fc2=nn.Linear(d_model,d_ff,bias=False)
        self.out=nn.Linear(d_ff,d_model,bias=False)
        self.A_g=A_g; self.omega_g=omega_g; self.phi_g=phi_g; self.eps=eps
    def forward(self, x, seq_pos=None):
        if seq_pos is None:
            L=x.shape[1] if x.dim()==3 else 1
            seq_pos=torch.arange(1,L+1,device=x.device,dtype=x.dtype)
        log_pos=torch.log(seq_pos+self.eps)
        bias=self.A_g*torch.cos(self.omega_g*log_pos+self.phi_g)
        # bias shape: (L,) -> broadcast to (B, L, 1) for elementwise with fc2 output
        if x.dim()==3:
            bias=bias.unsqueeze(0).unsqueeze(-1)  # (1, L, 1)
        g=F.silu(self.fc1(x))*torch.sigmoid(self.fc2(x)+bias)
        return self.out(g)

class LPSSMBlock(nn.Module):
    def __init__(self, d_model, d_state=32, num_scales=2, alpha=0.5,
                 base_omega=2.0, omega_spread=2.0, dropout=0.0):
        super().__init__()
        self.norm1=nn.LayerNorm(d_model)
        self.ssm=LogPeriodicDiagonalSSM(d_model,d_state,num_scales,alpha,base_omega,omega_spread)
        self.norm2=nn.LayerNorm(d_model)
        self.glu=LogPeriodicGLU(d_model,d_model*4)
        self.dropout=nn.Dropout(dropout)
    def forward(self, x, seq_pos=None):
        residual=x; x_norm=self.norm1(x); x_ssm=self.ssm(x_norm)
        x=residual+self.dropout(x_ssm)
        residual=x; x_norm=self.norm2(x); x_glu=self.glu(x_norm,seq_pos)
        return residual+self.dropout(x_glu)

class MiniLPSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(VOCAB,H)
        self.pos_enc=LogPeriodicPositionalEncoding(H)
        self.layers=nn.ModuleList([LPSSMBlock(H,d_state=32,num_scales=2) for _ in range(2)])
        self.norm_out=nn.LayerNorm(H); self.head=nn.Linear(H,VOCAB,bias=False)
    def forward(self, x, hp=None):
        B,L=x.shape; xe=self.embed(x)
        pos=torch.arange(1,L+1,device=x.device).unsqueeze(0).expand(B,-1)
        x=xe+self.pos_enc(pos)
        seq_pos=torch.arange(1,L+1,dtype=torch.float,device=x.device)
        for layer in self.layers: x=layer(x,seq_pos)
        return self.head(self.norm_out(x)), None

# ============================================================
# Data
# ============================================================
def make_batch(bs, nl):
    t=torch.randint(2,VOCAB,(bs,))
    n=torch.randint(2,VOCAB,(bs,nl))
    e=torch.full((bs,1),END,dtype=torch.long)
    x=torch.cat([t.unsqueeze(1),n,e],1)
    y=torch.full_like(x,IGNORE); y[:,-1]=t
    return x,y

def train_one(model, name, steps=2500):
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps)
    best=float('inf'); t0=time.time()
    for st in range(1,steps+1):
        x,y=make_batch(64,random.randint(4,64)); x,y=x.to(device),y.to(device)
        log,_=model(x); loss=F.cross_entropy(log[:,-1,:],y[:,-1],ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sch.step()
        if loss.item()<best: best=loss.item()
        if st%500==0: print(f"  {name:>12} step{st:5d} loss={loss.item():.4f} best={best:.4f} {time.time()-t0:.0f}s",flush=True)
    return best

@torch.no_grad()
def eval_one(model, name, dists, bs=64):
    model.eval(); r={}
    for d in dists:
        # Transformer 用极小 batch
        if name == "Transformer" and d >= 1024:
            eb = 2
        elif name == "Transformer" and d >= 256:
            eb = 8
        elif d <= 4096:
            eb = bs
        else:
            eb = 8
        c=0; total=0
        n_batch = 4 if d <= 4096 else 2
        for _ in range(n_batch):
            x,y=make_batch(eb,d); x,y=x.to(device),y.to(device)
            log,_=model(x); c+=(log[:,-1,:].argmax(-1)==y[:,-1]).sum().item(); total+=eb
        r[d]=c/total*100
    return r

# ============================================================
print("="*70)
print("  CopyFirst: 5 Architectures @ ~250K params")
print("="*70)

models=[
    ("FRSM",       MiniFRSM()),
    ("OpenASH",    MiniOpenASH()),
    ("WDLM-N",     MiniWDLM()),
    ("Transformer",MiniTransformer()),
    ("LP-SSM",     MiniLPSSM()),
]
for n,m in models:
    p=sum(p.numel() for p in m.parameters())
    print(f"  {n:>12}: {p:,} params"); m.to(device)

print(f"\n  Training (2500 steps, noise 4-64)...")
best={}
for n,m in models: best[n]=train_one(m,n)

dists=[4,64,256,1024,4096,8192,16384]
print(f"\n{'='*70}")
print(f"  {'Dist':>6} | " + " | ".join([f"{n:>7}" for n,_ in models]))
print(f"  "+"-"*(8+9*len(models)))
res={}
for n,m in models: res[n]=eval_one(m,n,dists)
for d in dists:
    print(f"  {d:6d} | " + " | ".join([f"{res[n][d]:7.1f}" for n,_ in models]))

print(f"\n  Summary (best_loss | far-field avg 4K-16K):")
rank=[]
for n,_ in models:
    far=sum(res[n][d] for d in [4096,8192,16384])/3
    rank.append((n,far,best[n]))
rank.sort(key=lambda x:x[1],reverse=True)
for n,far,bl in rank:
    mk="OK" if bl<0.1 else ("partial" if bl<1.0 else "FAIL")
    print(f"  {n:>12}: far={far:5.1f}%  loss={bl:.4f}  [{mk}]")
print(f"\nDone.")
