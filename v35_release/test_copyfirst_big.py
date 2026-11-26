"""FRSM CopyFirst: big model test — already trained, just eval"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time

device = torch.device("cuda")
VOCAB = 32; END = 0; IGNORE = 1

# Models (same as before, only needed for loading)
class CopyFirstFRSM(nn.Module):
    def __init__(self, d_model=256, num_scales=4):
        super().__init__()
        self.d_model = d_model; self.num_scales = num_scales
        self.embed = nn.Embedding(VOCAB, d_model)
        self.input_proj = nn.Linear(d_model, d_model)
        self.W_forget = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(num_scales)])
        self.W_input  = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(num_scales)])
        self.W_cand   = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(num_scales)])
        for w in self.W_forget: nn.init.constant_(w.bias, 1.0)
        for w in self.W_input:  nn.init.constant_(w.bias, -2.0)
        self.scale_fusion = nn.Linear(d_model * num_scales, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB)
    def forward(self, x, h_prev=None):
        B, T = x.shape
        if h_prev is None: h = [torch.zeros(B, self.d_model, device=device) for _ in range(self.num_scales)]
        else: h = [hs.clone() for hs in h_prev]
        x_emb = self.embed(x); outs = []
        for t in range(T):
            inp = self.input_proj(x_emb[:, t, :])
            nh = []
            for s in range(self.num_scales):
                if t % (2**s) == 0:
                    c = torch.cat([h[s], inp], -1)
                    f = torch.sigmoid(self.W_forget[s](c)); i = torch.sigmoid(self.W_input[s](c))
                    nh.append(f * h[s] + i * torch.tanh(self.W_cand[s](c)))
                else: nh.append(h[s])
            h = nh
            fused = self.fusion_norm(self.scale_fusion(torch.cat(h, -1)))
            outs.append(self.head(fused).unsqueeze(1))
        return torch.cat(outs, 1), h

class MiniFRSM2(nn.Module):
    def __init__(self):
        super().__init__()
        H = 128; ns = 2
        self.H = H; self.ns = ns
        self.embed = nn.Embedding(VOCAB, H); self.inp = nn.Linear(H, H)
        self.W_forget = nn.ModuleList([nn.Linear(H*2, H) for _ in range(ns)])
        self.W_input = nn.ModuleList([nn.Linear(H*2, H) for _ in range(ns)])
        self.W_cand = nn.ModuleList([nn.Linear(H*2, H) for _ in range(ns)])
        for w in self.W_forget: nn.init.constant_(w.bias, 1.0)
        for w in self.W_input: nn.init.constant_(w.bias, -2.0)
        self.fusion = nn.Linear(H*2, H); self.ln = nn.LayerNorm(H)
        self.head = nn.Linear(H, VOCAB)
    def forward(self, x, h_prev=None):
        B, T = x.shape
        if h_prev is None: h = [torch.zeros(B, self.H, device=device) for _ in range(self.ns)]
        else: h = [hs.clone() for hs in h_prev]
        x_e = self.embed(x); outs = []
        for t in range(T):
            inp = self.inp(x_e[:,t,:])
            nh = []
            for s in range(self.ns):
                if t % (2**s) == 0:
                    c = torch.cat([h[s], inp], -1)
                    f = torch.sigmoid(self.W_forget[s](c)); i = torch.sigmoid(self.W_input[s](c))
                    nh.append(f*h[s] + i*torch.tanh(self.W_cand[s](c)))
                else: nh.append(h[s])
            h = nh
            fused = self.ln(self.fusion(torch.cat(h, -1)))
            outs.append(self.head(fused).unsqueeze(1))
        return torch.cat(outs, 1), h

def make_batch(bs, noise_len):
    targets = torch.randint(2, VOCAB, (bs,))
    noise = torch.randint(2, VOCAB, (bs, noise_len))
    end = torch.full((bs, 1), END, dtype=torch.long)
    x = torch.cat([targets.unsqueeze(1), noise, end], 1)
    y = torch.full_like(x, IGNORE); y[:, -1] = targets
    return x, y

def train_and_eval(model, name, steps=2000, max_noise=64, bs=32):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best = float('inf')
    t0 = time.time()
    for st in range(1, steps + 1):
        nl = random.randint(4, max_noise)
        x, y = make_batch(bs, nl); x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, -1, :], y[:, -1], ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if loss.item() < best: best = loss.item()
    print(f"  {name:>18s}: best_loss={best:.5f}  time={time.time()-t0:.0f}s", flush=True)
    
    # 评估: 关键距离
    model.eval()
    test_dists = [4, 64, 512, 2048, 8192, 16384, 32768, 65536]
    results = {}
    for d in test_dists:
        eb = 64 if d <= 8192 else 16  # 大距离用小batch
        n_batch = 2 if d <= 32768 else 1
        c = 0; total = eb * n_batch
        for _ in range(n_batch):
            x, y = make_batch(eb, d); x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            c += (logits[:, -1, :].argmax(-1) == y[:, -1]).sum().item()
        results[d] = c / total * 100
    return best, results

# ============================================================
print("=" * 70)
print("  CopyFirst: FRSM scale comparison")
print("=" * 70)

torch.manual_seed(42)

models = [
    ("FRSM-14.7M(4sc)", CopyFirstFRSM(d_model=256, num_scales=4).to(device), 32),
    ("FRSM-512K(2sc)",  CopyFirstFRSM(d_model=256, num_scales=2).to(device), 32),
    ("FRSM-255K(2sc)",  MiniFRSM2().to(device), 64),
]
for name, m, _ in models:
    print(f"  {name}: {sum(p.numel() for p in m.parameters()):,} params")

results = {}
for name, m, bs in models:
    bl, acc = train_and_eval(m, name, steps=2000, bs=bs)
    results[name] = (bl, acc)

# 汇总
dists = [4, 64, 512, 2048, 8192, 16384, 32768, 65536]
print(f"\n{'='*70}")
print(f"  {'Dist':>7} | " + " | ".join([f"{n:>15}" for n, _, _ in models]))
print(f"  " + "-" * (9 + 17 * len(models)))
for d in dists:
    vals = [f"{results[n][1].get(d, 0):15.1f}" for n, _, _ in models]
    print(f"  {d:7,d} | " + " | ".join(vals))

print(f"\n  Convergence + Far-field summary:")
for name, m, _ in models:
    bl, acc = results[name]
    near = sum(acc[d] for d in [4,64]) / 2
    far  = sum(acc[d] for d in [8192,16384,32768,65536]) / 4
    n = sum(p.numel() for p in m.parameters())
    print(f"    {name:>18s} ({n:>9,}p) loss={bl:.5f}  near={near:.1f}%  far={far:.1f}%  gap={far-near:+.1f}%")
print(f"\nDone.")
