"""V6a loop vs fast 速度对比"""
import torch, torch.nn as nn, torch.nn.functional as F, time

device = torch.device("cuda")
V=32; H=128; B=4; T=384

# === V6a Loop ===
class V6a_Loop(nn.Module):
    def __init__(self):
        super().__init__(); self.H=H; self.ns=4
        self.e=nn.Embedding(V,H); self.ii=nn.Linear(H,H)
        self.sc=nn.ModuleList([nn.Linear(H*2,H) for _ in range(4)])
        self.gates=nn.ModuleList([nn.Sequential(nn.Linear(H*2,H//4),nn.GELU(),nn.Linear(H//4,1),nn.Sigmoid()) for _ in range(4)])
        self.fu=nn.Linear(H*4,H); self.fn=nn.LayerNorm(H); self.h=nn.Linear(H,V)
        self.sc_f=nn.ModuleList([nn.Linear(H*2,H) for _ in range(4)])
        self.sc_c=nn.ModuleList([nn.Linear(H*2,H) for _ in range(4)])
        for w in self.sc_f: nn.init.constant_(w.bias,1.0)
        for w in self.sc: nn.init.constant_(w.bias,-2.0)
    def forward(self,x):
        B,T=x.shape; h=[torch.zeros(B,H,device=x.device) for _ in range(4)]
        xe=self.e(x); o=[]
        for t in range(T):
            inp=self.ii(xe[:,t,:]); nh=[]
            for s in range(4):
                cand=torch.tanh(self.sc_c[s](torch.cat([h[s],inp],-1)))
                f=torch.sigmoid(self.sc_f[s](torch.cat([h[s],inp],-1)))
                i=torch.sigmoid(self.sc[s](torch.cat([h[s],inp],-1)))
                cand_val=f*h[s]+i*cand
                g=self.gates[s](torch.cat([h[s],inp],-1))
                nh.append(g*cand_val+(1-g)*h[s])
            h=nh; fu=self.fn(self.fu(torch.cat(h,-1)))
            o.append(self.h(fu).unsqueeze(1))
        return torch.cat(o,1)

# === V6a Fast ===
from frsm_v6a_fast import FRSM_V6_Fast

# Pre-load once
x = torch.randint(0, V, (B, T), device=device)
tgt = torch.randint(0, V, (B, T), device=device)
ITER = 50

for name, m in [("Loop", V6a_Loop()), ("Fast", FRSM_V6_Fast(V, H, 4))]:
    m = m.to(device).train()
    n = sum(p.numel() for p in m.parameters())
    
    # Warmup
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(5):
        log = m(x); loss = F.cross_entropy(log.reshape(-1, V), tgt.reshape(-1), ignore_index=1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize()
    
    # Benchmark
    t0 = time.time()
    for _ in range(ITER):
        log = m(x); loss = F.cross_entropy(log.reshape(-1, V), tgt.reshape(-1), ignore_index=1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    
    tok_per_s = B * T * ITER / elapsed
    print(f"  {name:>6} ({n:,}p): {elapsed:.1f}s total, {elapsed/ITER*1000:.0f}ms/step, {tok_per_s:.0f} tok/s")
    del m; torch.cuda.empty_cache()

# 前向 Only
print("\n  Forward only:")
for name, m in [("Loop", V6a_Loop()), ("Fast", FRSM_V6_Fast(V, H, 4))]:
    m = m.to(device).eval()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(100):
        _ = m(x)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    tok_per_s = B * T * 100 / elapsed
    print(f"  {name:>6}: {tok_per_s:.0f} tok/s")
    del m; torch.cuda.empty_cache()
