"""
快速评估: 对已训练模型做 CopyFirst 评估
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random

device = torch.device("cuda")
VOCAB = 32; END = 0; IGNORE = 1; H = 128

# ========== 模型 (同前) ==========
class MiniFRSM2(nn.Module):
    def __init__(self):
        super().__init__()
        self.H = H; self.ns = 2
        self.embed = nn.Embedding(VOCAB, H); self.inp = nn.Linear(H, H)
        self.W_forget = nn.ModuleList([nn.Linear(H*2, H) for _ in range(2)])
        self.W_input = nn.ModuleList([nn.Linear(H*2, H) for _ in range(2)])
        self.W_cand = nn.ModuleList([nn.Linear(H*2, H) for _ in range(2)])
        for w in self.W_forget: nn.init.constant_(w.bias, 1.0)
        for w in self.W_input: nn.init.constant_(w.bias, -2.0)
        self.fusion = nn.Linear(H*2, H); self.ln = nn.LayerNorm(H)
        self.head = nn.Linear(H, VOCAB)
    def forward(self, x, h_prev=None):
        B, T = x.shape
        if h_prev is None: h = [torch.zeros(B, H, device=device) for _ in range(self.ns)]
        else: h = [hs.clone() for hs in h_prev]
        x_e = self.embed(x); outs = []
        for t in range(T):
            inp = self.inp(x_e[:,t,:])
            nh = []
            for s in range(self.ns):
                if t % (2**s) == 0:
                    c = torch.cat([h[s], inp], -1)
                    f = torch.sigmoid(self.W_forget[s](c))
                    i = torch.sigmoid(self.W_input[s](c))
                    nh.append(f*h[s] + i*torch.tanh(self.W_cand[s](c)))
                else: nh.append(h[s])
            h = nh
            fused = self.ln(self.fusion(torch.cat(h, -1)))
            outs.append(self.head(fused).unsqueeze(1))
        return torch.cat(outs, 1), h

# 简化 cummax 模型
CACHE = {}
for cls_name, cls in [
    ("OpenASH", None), ("WDLM-Neural", None), ("WDLM-Real", None),
    ("Transformer", None), ("LSTM", None), ("GRU", None)
]:
    # 从之前完整的 bench_all_arch.py 导入
    pass

# ============================================================
class MiniOpenASH(nn.Module):
    def __init__(self):
        super().__init__()
        self.H = H; self.heads = 4; self.dh = H//self.heads
        self.embed = nn.Embedding(VOCAB, H); self.proj = nn.Linear(H, 4*H, bias=False)
        self.gen_out = nn.Linear(5*H, H)
        self.a1 = nn.Parameter(torch.tensor(0.5)); self.a2 = nn.Parameter(torch.tensor(0.5)); self.a3 = nn.Parameter(torch.tensor(0.5))
        self.ln = nn.LayerNorm(H); self.head = nn.Linear(H, VOCAB, bias=False)
        self.model_flag = "train"
    def forward(self, x, state=None):
        B, T = x.shape; h = self.embed(x)
        o = self.proj(h).view(B,T,4,self.heads,self.dh)
        a,b,c,d = [t.permute(0,3,1,2) for t in o.unbind(2)]
        if state is None: e,_ = torch.cummax(c,2); sn = e[:,:, -1:,:]
        else: e,_ = torch.cummax(torch.cat([state,c],2),2); e=e[:,:,1:,:]; sn=e[:,:,-1:,:]
        t1=a*b; t2=self.a1*b+self.a2*d; t3=a*(self.a3*e+d); t4=b*(c+e); t5=c*e
        cb=torch.cat([t1,t2,t3,t4,t5],-1).permute(0,2,1,3).reshape(B,T,-1)
        return self.head(self.ln(self.gen_out(cb))), sn

class MiniWDLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.H = H
        self.embed = nn.Embedding(VOCAB, H)
        self.rot = nn.Linear(H,H,bias=False); self.amp = nn.Linear(H,H,bias=False); self.gate = nn.Linear(H,H,bias=False)
        self.cum_proj = nn.Linear(H,4*H); self.gen_out = nn.Linear(5*H,H)
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

class MiniWDLMReal(nn.Module):
    def __init__(self):
        super().__init__()
        self.H=H
        self.embed=nn.Embedding(VOCAB,H)
        self.evo_k=nn.Linear(H,H,bias=False); self.evo_g=nn.Linear(H,H,bias=False)
        self.dt=nn.Parameter(torch.tensor(0.1))
        self.cum_proj=nn.Linear(H,4*H); self.gen_out=nn.Linear(5*H,H)
        self.a1=nn.Parameter(torch.tensor(0.5)); self.a2=nn.Parameter(torch.tensor(0.5)); self.a3=nn.Parameter(torch.tensor(0.5))
        self.ln=nn.LayerNorm(H); self.head=nn.Linear(H,VOCAB,bias=False)
    def forward(self,x,state=None):
        B,T=x.shape; psi=self.embed(x)
        g=self.evo_g(psi); psi=psi+self.dt*self.evo_k(psi)*(torch.sin(g)+torch.cos(g))*0.5
        a,b,c,d=self.cum_proj(psi).chunk(4,-1)
        if state is None: e,_=torch.cummax(c,1); sn=e[:,-1:,:]
        else: e,_=torch.cummax(torch.cat([state,c],1),1); e=e[:,1:,:]; sn=e[:,-1:,:]
        t1=a*b; t2=self.a1*b+self.a2*d; t3=a*(self.a3*e+d); t4=b*(c+e); t5=c*e
        return self.head(self.ln(self.gen_out(torch.cat([t1,t2,t3,t4,t5],-1)))), sn

class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(VOCAB,H)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(H,4,H*4,0.0,batch_first=True) for _ in range(2)])
        self.head=nn.Linear(H,VOCAB)
    def forward(self,x,hp=None):
        T=x.size(1); m=nn.Transformer.generate_square_subsequent_mask(T,device=device)
        h=self.layers[1](self.layers[0](self.embed(x)*math.sqrt(H),src_mask=m),src_mask=m)
        return self.head(h),None

class MiniLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(VOCAB,H); self.lstm=nn.LSTM(H,H,2,batch_first=True); self.head=nn.Linear(H,VOCAB)
    def forward(self,x,hp=None): return self.head(self.lstm(self.embed(x))[0]),None

class MiniGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(VOCAB,H); self.gru=nn.GRU(H,H,2,batch_first=True); self.head=nn.Linear(H,VOCAB)
    def forward(self,x,hp=None): return self.head(self.gru(self.embed(x))[0]),None

# ============================================================
def make_batch(bs,nl):
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
    best=float('inf')
    for st in range(1,steps+1):
        x,y=make_batch(64,random.randint(4,64)); x,y=x.to(device),y.to(device)
        log,_=model(x); loss=F.cross_entropy(log[:,-1,:],y[:,-1],ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sch.step()
        if loss.item()<best: best=loss.item()
        if st%500==0: print(f"  {name:>16s} step{st:5d} loss={loss.item():.4f} best={best:.4f}",flush=True)
    return best

@torch.no_grad()
def eval_one(model, dists, bs=128):
    model.eval(); r={}
    for d in dists:
        c=0
        for _ in range(8):
            x,y=make_batch(bs,d); x,y=x.to(device),y.to(device)
            log,_=model(x); c+=(log[:,-1,:].argmax(-1)==y[:,-1]).sum().item()
        r[d]=c/(8*bs)*100
    return r

# ============================================================
print("="*70)
print("  CopyFirst: 7 Architectures @ ~250K params")
print("="*70)

models=[("FRSM2",MiniFRSM2()),("OpenASH",MiniOpenASH()),("WDLM-N",MiniWDLM()),
        ("WDLM-R",MiniWDLMReal()),("Transf",MiniTransformer()),("LSTM",MiniLSTM()),("GRU",MiniGRU())]

for n,m in models:
    print(f"  {n:>10}: {sum(p.numel() for p in m.parameters()):,} p"); m.to(device)

torch.manual_seed(42)
print("\nTraining (2500 steps, noise 4-64)...")
best={}
for n,m in models: best[n]=train_one(m,n)

# 评估: 关键距离点 (避免 Transformer O(n²) 超时)
dists=[4,64,256,1024,4096,8192,16384]
bs_eval=128
print(f"\n{'='*70}")
print(f"  {'Dist':>6} | " + " | ".join([f"{n:>7}" for n,_ in models]))
print("  "+"-"*(9+9*len(models)))
res={}
for n,m in models: 
    # 对 Transformer 用更小的 batch
    eb = bs_eval if "Transf" not in n else 32
    res[n]=eval_one(m,dists,bs=eb)
for d in dists:
    row=f"  {d:6d} | " + " | ".join([f"{res[n][d]:7.1f}" for n,_ in models])
    print(row)

# 极限距离单独测试 (只用小 batch)
print(f"\n  Extreme (bs=16):")
extreme=[32768,65536]
for d in extreme:
    row=f"  {d:6d} |"
    for n,m in models:
        eb=16 if "Transf" in n else 32  # Transformer 16更小
        try:
            acc=eval_one(m,[d],bs=eb)[d]
            row+=f" {acc:7.1f}"
        except:
            row+=f" {'OOM':>7}"
    print(row)

# 排名
print(f"\n  Generalization (avg 512-16384 | convergence):")
all_dist_acc = {}
for n,m in models: all_dist_acc[n]=res[n]
rank=[]
for n,_ in models:
    far_vals=[all_dist_acc[n][d] for d in [512,1024,2048,4096,8192,16384] if d in all_dist_acc[n]]
    far=sum(far_vals)/len(far_vals) if far_vals else 0
    rank.append((n,far,best[n]))
rank.sort(key=lambda x:x[1],reverse=True)
for n,far,bl in rank:
    mark="✓✓" if bl<0.01 else ("✓" if bl<0.1 else ("△" if bl<1.0 else "✗"))
    print(f"  {n:>10}: far={far:5.1f}%  best_loss={bl:.4f}  [{mark}]")

print("\nDone.")
