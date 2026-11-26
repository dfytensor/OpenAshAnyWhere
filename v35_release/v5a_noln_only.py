"""V5a-NoLN 单独 CopyFirst"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
device = torch.device("cuda"); V=32; E=0; I=1; H=128

class RawBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.Wf = nn.Linear(d*2, d); self.Wi = nn.Linear(d*2, d); self.Wc = nn.Linear(d*2, d)
        nn.init.constant_(self.Wf.bias, 1.0); nn.init.constant_(self.Wi.bias, -2.0)
    def forward(self, h, inp):
        c = torch.cat([h, inp], -1)
        f = torch.sigmoid(self.Wf(c)); i = torch.sigmoid(self.Wi(c))
        return f * h + i * torch.tanh(self.Wc(c))

class V5a_NoLN(nn.Module):
    def __init__(self, vs, H=128):
        super().__init__()
        self.H = H; self.embed = nn.Embedding(vs, H); self.input_proj = nn.Linear(H, H)
        self.lm_scales = nn.ModuleList([RawBlock(H) for _ in range(2)])
        self.lm_fusion = nn.Linear(H*2, H); self.lm_fn = nn.LayerNorm(H)
        self.mem_scales = nn.ModuleList([RawBlock(H) for _ in range(4)])
        self.mem_fusion = nn.Linear(H*4, H); self.mem_fn = nn.LayerNorm(H)
        self.gate = nn.Sequential(nn.Linear(H, H//4), nn.GELU(), nn.Linear(H//4, 1), nn.Sigmoid())
        self.out_norm = nn.LayerNorm(H); self.head = nn.Linear(H, vs)

    def forward(self, x, hp=None):
        B, T = x.shape
        lm_h = [torch.zeros(B, self.H, device=x.device) for _ in range(2)]
        mem_h = [torch.zeros(B, self.H, device=x.device) for _ in range(4)]
        xe = self.embed(x); outs = []
        for t in range(T):
            inp = self.input_proj(xe[:, t, :])
            ln = []; mn = []
            for s in range(2):
                if t % (2**s) == 0: ln.append(self.lm_scales[s](lm_h[s], inp))
                else: ln.append(lm_h[s])
            lm_h = ln; lo = self.lm_fn(self.lm_fusion(torch.cat(lm_h, -1)))
            for s in range(4):
                if t % (2**s) == 0: mn.append(self.mem_scales[s](mem_h[s], inp))
                else: mn.append(mem_h[s])
            mem_h = mn; mo = self.mem_fn(self.mem_fusion(torch.cat(mem_h, -1)))
            g = self.gate(inp)
            out = self.out_norm(g * lo + (1 - g) * mo + inp)
            outs.append(self.head(out).unsqueeze(1))
        return torch.cat(outs, 1), lm_h + mem_h

def mf(bs, nl):
    t = torch.randint(2, V, (bs,)); n = torch.randint(2, V, (bs, nl))
    e = torch.full((bs, 1), E, dtype=torch.long)
    x = torch.cat([t.unsqueeze(1), n, e], 1); y = torch.full_like(x, I); y[:, -1] = t
    return x, y

torch.manual_seed(42)
m = V5a_NoLN(V).to(device); n = sum(p.numel() for p in m.parameters())
print(f"V5a-NoLN: {n:,} params")
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 2500)
m.train(); best = float('inf'); t0 = time.time()
for st in range(1, 2501):
    x, y = mf(64, random.randint(4, 64)); x, y = x.to(device), y.to(device)
    log, _ = m(x); loss = F.cross_entropy(log[:, -1, :], y[:, -1], ignore_index=I)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
    if loss.item() < best: best = loss.item()
    if st % 500 == 0: print(f"step{st:5d} best={best:.5f} {time.time()-t0:.0f}s", flush=True)
print(f"best={best:.5f} {time.time()-t0:.0f}s", flush=True)
m.eval()
for d in [4, 64, 256, 1024, 4096, 8192, 16384, 32768, 65536]:
    eb = 64 if d <= 4096 else 8; c = 0; total = 0
    for _ in range(4 if d <= 4096 else 2):
        x, y = mf(eb, d); x, y = x.to(device), y.to(device)
        log, _ = m(x); c += (log[:, -1, :].argmax(-1) == y[:, -1]).sum().item(); total += eb
    print(f"{d:5d} | {c/total*100:.1f}%", flush=True)
print("Done.")
