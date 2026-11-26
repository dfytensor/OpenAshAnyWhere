"""
V6: Content-Gated Update Frequency
核心: 不用固定周期 2^s，让模型自己学什么时候更新
每尺度: update_prob = sigmoid(Linear(h + inp)) → 伯努利采样/软阈值
LM + CopyFirst 对比 V1
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
device = torch.device("cuda"); V=32; E=0; I=1; H=128

class RawBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.Wf = nn.Linear(d*2, d); self.Wi = nn.Linear(d*2, d); self.Wc = nn.Linear(d*2, d)
        nn.init.constant_(self.Wf.bias, 1.0); nn.init.constant_(self.Wi.bias, -2.0)
    def forward(self, h, inp):
        c = torch.cat([h, inp], -1); f = torch.sigmoid(self.Wf(c)); i = torch.sigmoid(self.Wi(c))
        return f * h + i * torch.tanh(self.Wc(c))

# ============================================================
# V1 baseline
# ============================================================
class V1(nn.Module):
    def __init__(self, vs, H=128, ns=4):
        super().__init__(); self.H=H; self.ns=ns
        self.e=nn.Embedding(vs,H); self.ii=nn.Linear(H,H)
        self.sc=nn.ModuleList([RawBlock(H) for _ in range(ns)])
        self.fu=nn.Linear(H*ns,H); self.fn=nn.LayerNorm(H); self.h=nn.Linear(H,vs)
    def forward(self,x,hp=None):
        B,T=x.shape
        if hp is None: h=[torch.zeros(B,self.H,device=x.device) for _ in range(self.ns)]
        else: h=[c.clone() for c in hp]
        xe=self.e(x); o=[]
        for t in range(T):
            inp=self.ii(xe[:,t,:]); nh=[]
            for s in range(self.ns):
                if t%(2**s)==0: nh.append(self.sc[s](h[s],inp))
                else: nh.append(h[s])
            h=nh; fu=self.fn(self.fu(torch.cat(h,-1)))
            o.append(self.h(fu).unsqueeze(1))
        return torch.cat(o,1),h

# ============================================================
# V6a: Content-gated update (软决策)
# ============================================================
class V6_ContentGate(nn.Module):
    """
    每尺度: update_strength = sigmoid(W * [h; inp]) ∈ (0,1)
    h_new = update_strength * candidate + (1 - update_strength) * h_old
    训练时用软决策，推理时>0.5则更新
    """
    def __init__(self, vs, H=128, ns=4):
        super().__init__(); self.H=H; self.ns=ns
        self.e=nn.Embedding(vs,H); self.ii=nn.Linear(H,H)
        self.sc=nn.ModuleList([RawBlock(H) for _ in range(ns)])
        # Content gate per scale: decides update strength
        self.gates=nn.ModuleList([nn.Sequential(nn.Linear(H*2,H//4),nn.GELU(),nn.Linear(H//4,1),nn.Sigmoid()) for _ in range(ns)])
        self.fu=nn.Linear(H*ns,H); self.fn=nn.LayerNorm(H); self.h=nn.Linear(H,vs)

    def forward(self,x,hp=None):
        B,T=x.shape
        if hp is None: h=[torch.zeros(B,self.H,device=x.device) for _ in range(self.ns)]
        else: h=[c.clone() for c in hp]
        xe=self.e(x); o=[]
        for t in range(T):
            inp=self.ii(xe[:,t,:]); nh=[]
            for s in range(self.ns):
                candidate=self.sc[s](h[s],inp)
                gt=torch.cat([h[s],inp],-1)
                update_str=self.gates[s](gt)  # (B,1)
                # 硬阈值 (训练用直通估计)
                if self.training:
                    # 软混合: 平滑更新
                    nh.append(update_str*candidate + (1-update_str)*h[s])
                else:
                    # 硬决策
                    update=update_str>0.5
                    nh.append(torch.where(update,candidate,h[s]))
            h=nh; fu=self.fn(self.fu(torch.cat(h,-1)))
            o.append(self.h(fu).unsqueeze(1))
        return torch.cat(o,1),h

# ============================================================
# V6b: Content-gated + Residual (v3继承)
# ============================================================
class V6_ContentResidual(nn.Module):
    """v6a + α残差: update_str 控制写入, α控制保留"""
    def __init__(self, vs, H=128, ns=4, alpha=0.7):
        super().__init__(); self.H=H; self.ns=ns
        self.e=nn.Embedding(vs,H); self.ii=nn.Linear(H,H)
        self.sc=nn.ModuleList([RawBlock(H) for _ in range(ns)])
        self.gates=nn.ModuleList([nn.Sequential(nn.Linear(H*2,H//4),nn.GELU(),nn.Linear(H//4,1),nn.Sigmoid()) for _ in range(ns)])
        self.register_buffer('alpha',torch.tensor(alpha))
        self.fu=nn.Linear(H*ns,H); self.fn=nn.LayerNorm(H); self.h=nn.Linear(H,vs)

    def forward(self,x,hp=None):
        B,T=x.shape
        if hp is None: h=[torch.zeros(B,self.H,device=x.device) for _ in range(self.ns)]
        else: h=[c.clone() for c in hp]
        xe=self.e(x); o=[]
        for t in range(T):
            inp=self.ii(xe[:,t,:]); nh=[]
            for s in range(self.ns):
                candidate=self.sc[s](h[s],inp)
                gt=torch.cat([h[s],inp],-1)
                update_str=self.gates[s](gt)  # 0→不可写入, 1→全写入
                # 双重控制: α保留 + update控制写入量
                nh.append(self.alpha*h[s] + (1-self.alpha)*(update_str*candidate + (1-update_str)*h[s]))
            h=nh; fu=self.fn(self.fu(torch.cat(h,-1)))
            o.append(self.h(fu).unsqueeze(1))
        return torch.cat(o,1),h

# ============================================================
def mf(bs,nl):
    t=torch.randint(2,V,(bs,)); n=torch.randint(2,V,(bs,nl))
    e=torch.full((bs,1),E,dtype=torch.long)
    x=torch.cat([t.unsqueeze(1),n,e],1); y=torch.full_like(x,I); y[:,-1]=t
    return x,y

for name, Model in [("V1", V1), ("V6a-ContentGate", V6_ContentGate), ("V6b-ContentResid", V6_ContentResidual)]:
    torch.manual_seed(42)
    m = Model(V).to(device)
    n = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 2500)
    m.train(); best = float('inf'); t0 = time.time()
    for st in range(1, 2501):
        x, y = mf(64, random.randint(4, 64)); x, y = x.to(device), y.to(device)
        log, _ = m(x); loss = F.cross_entropy(log[:, -1, :], y[:, -1], ignore_index=I)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        if loss.item() < best: best = loss.item()
        if st % 500 == 0: print(f"  [{name}] step{st:5d} best={best:.5f} {time.time()-t0:.0f}s", flush=True)
    print(f"  [{name}] best={best:.5f} params={n:,}", flush=True)
    m.eval()
    for d in [4, 64, 256, 1024, 4096, 8192, 16384, 32768, 65536]:
        eb = 64 if d <= 4096 else 8; c = 0; total = 0
        for _ in range(4 if d <= 4096 else 2):
            x, y = mf(eb, d); x, y = x.to(device), y.to(device)
            log, _ = m(x); c += (log[:, -1, :].argmax(-1) == y[:, -1]).sum().item(); total += eb
        print(f"  {d:5d} | {c/total*100:.1f}%", flush=True)
    del m; torch.cuda.empty_cache()

print("\nDone.")
