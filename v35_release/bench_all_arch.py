"""
7架构对比 V2: 使用训练好的模型直接评估 (跳过训练)
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random

device = torch.device("cuda")
VOCAB = 32; END = 0; IGNORE = 1
H = 128

# ========== 模型定义 (同上，略简) ==========
class MiniFRSM2(nn.Module):  # 2-scale 版本 (之前表现最好的)
    def __init__(self):
        super().__init__()
        self.H = H; self.ns = 2
        self.embed = nn.Embedding(VOCAB, H); self.inp = nn.Linear(H, H)
        self.W_forget = nn.ModuleList([nn.Linear(H*2, H) for _ in range(2)])
        self.W_input = nn.ModuleList([nn.Linear(H*2, H) for _ in range(2)])
        self.W_cand = nn.ModuleList([nn.Linear(H*2, H) for _ in range(2)])
        for w in self.W_forget:
            nn.init.constant_(w.bias, 1.0)
        for w in self.W_input:
            nn.init.constant_(w.bias, -2.0)
        self.fusion = nn.Linear(H*2, H); self.ln = nn.LayerNorm(H)
        self.head = nn.Linear(H, VOCAB)

    def forward(self, x, h_prev=None):
        B, T = x.shape
        if h_prev is None:
            h = [torch.zeros(B, self.H, device=device) for _ in range(self.ns)]
        else:
            h = [hs.clone() for hs in h_prev]
        x_e = self.embed(x); outs = []
        for t in range(T):
            inp = self.inp(x_e[:, t, :])
            nh = []
            for s in range(self.ns):
                if t % (2**s) == 0:
                    c = torch.cat([h[s], inp], dim=-1)
                    f = torch.sigmoid(self.W_forget[s](c))
                    i = torch.sigmoid(self.W_input[s](c))
                    cand = torch.tanh(self.W_cand[s](c))
                    nh.append(f * h[s] + i * cand)
                else:
                    nh.append(h[s])
            h = nh
            fused = self.ln(self.fusion(torch.cat(h, dim=-1)))
            outs.append(self.head(fused).unsqueeze(1))
        return torch.cat(outs, 1), h

class MiniOpenASH(nn.Module):
    def __init__(self):
        super().__init__()
        self.H = H; self.heads = 4; self.dh = H // self.heads
        self.embed = nn.Embedding(VOCAB, H)
        self.proj = nn.Linear(H, 4*H, bias=False)
        self.gen_out = nn.Linear(5*H, H)
        self.a1 = nn.Parameter(torch.tensor(0.5))
        self.a2 = nn.Parameter(torch.tensor(0.5))
        self.a3 = nn.Parameter(torch.tensor(0.5))
        self.ln = nn.LayerNorm(H)
        self.head = nn.Linear(H, VOCAB, bias=False)
        self.model_flag = "train"

    def forward(self, x, state=None):
        B, T = x.shape
        h = self.embed(x)
        o = self.proj(h).view(B, T, 4, self.heads, self.dh)
        a, b, c, d = o.unbind(2)
        a = a.permute(0, 3, 1, 2); b = b.permute(0, 3, 1, 2)
        c = c.permute(0, 3, 1, 2); d = d.permute(0, 3, 1, 2)

        if state is None:
            e, _ = torch.cummax(c, dim=2); state_new = e[:, :, -1:, :]
        else:
            e, _ = torch.cummax(torch.cat([state, c], dim=2), dim=2)
            e = e[:, :, 1:, :] if self.model_flag == "train" else e[:, :, -1:, :]
            state_new = e[:, :, -1:, :]

        t1 = a * b; t2 = self.a1 * b + self.a2 * d
        t3 = a * (self.a3 * e + d); t4 = b * (c + e); t5 = c * e
        combined = torch.cat([t1, t2, t3, t4, t5], dim=-1)
        out = self.gen_out(combined.permute(0, 2, 1, 3).reshape(B, T, -1))
        return self.head(self.ln(out)), state_new

class MiniWDLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.H = H
        self.embed = nn.Embedding(VOCAB, H)
        self.rot = nn.Linear(H, H, bias=False); self.amp = nn.Linear(H, H, bias=False)
        self.gate = nn.Linear(H, H, bias=False)
        self.cum_proj = nn.Linear(H, 4*H); self.gen_out = nn.Linear(5*H, H)
        self.a1 = nn.Parameter(torch.tensor(0.5)); self.a2 = nn.Parameter(torch.tensor(0.5))
        self.a3 = nn.Parameter(torch.tensor(0.5))
        self.ln = nn.LayerNorm(H); self.head = nn.Linear(H, VOCAB, bias=False)

    def forward(self, x, state=None):
        B, T = x.shape
        psi = self.embed(x)
        psi = psi * self.rot(psi) + torch.sigmoid(self.gate(psi)) * self.amp(psi) + psi
        o = self.cum_proj(psi); a, b, c, d = o.chunk(4, -1)
        if state is None: e, _ = torch.cummax(c, dim=1); state_new = e[:, -1:, :]
        else: e, _ = torch.cummax(torch.cat([state, c], dim=1), dim=1); e = e[:, 1:, :]; state_new = e[:, -1:, :]
        t1 = a*b; t2 = self.a1*b + self.a2*d
        t3 = a*(self.a3*e + d); t4 = b*(c + e); t5 = c*e
        return self.head(self.ln(self.gen_out(torch.cat([t1,t2,t3,t4,t5], -1)))), state_new

class MiniWDLMReal(nn.Module):
    def __init__(self):
        super().__init__()
        self.H = H
        self.embed = nn.Embedding(VOCAB, H)
        self.evo_k = nn.Linear(H, H, bias=False); self.evo_g = nn.Linear(H, H, bias=False)
        self.dt = nn.Parameter(torch.tensor(0.1))
        self.cum_proj = nn.Linear(H, 4*H); self.gen_out = nn.Linear(5*H, H)
        self.a1 = nn.Parameter(torch.tensor(0.5)); self.a2 = nn.Parameter(torch.tensor(0.5))
        self.a3 = nn.Parameter(torch.tensor(0.5))
        self.ln = nn.LayerNorm(H); self.head = nn.Linear(H, VOCAB, bias=False)

    def forward(self, x, state=None):
        B, T = x.shape
        psi = self.embed(x)
        g = self.evo_g(psi); psi = psi + self.dt * self.evo_k(psi) * (torch.sin(g) + torch.cos(g)) * 0.5
        o = self.cum_proj(psi); a, b, c, d = o.chunk(4, -1)
        if state is None: e, _ = torch.cummax(c, dim=1); state_new = e[:, -1:, :]
        else: e, _ = torch.cummax(torch.cat([state, c], dim=1), dim=1); e = e[:, 1:, :]; state_new = e[:, -1:, :]
        t1 = a*b; t2 = self.a1*b + self.a2*d
        t3 = a*(self.a3*e + d); t4 = b*(c + e); t5 = c*e
        return self.head(self.ln(self.gen_out(torch.cat([t1,t2,t3,t4,t5], -1)))), state_new

class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, H)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(H, 4, H*4, 0.0, batch_first=True) for _ in range(2)])
        self.head = nn.Linear(H, VOCAB)
    def forward(self, x, h_prev=None):
        T = x.size(1); mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        return self.head(self.layers[1](self.layers[0](self.embed(x)*math.sqrt(H), src_mask=mask), src_mask=mask)), None

class MiniLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, H)
        self.lstm = nn.LSTM(H, H, 2, batch_first=True); self.head = nn.Linear(H, VOCAB)
    def forward(self, x, h_prev=None):
        return self.head(self.lstm(self.embed(x))[0]), None

class MiniGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, H)
        self.gru = nn.GRU(H, H, 2, batch_first=True); self.head = nn.Linear(H, VOCAB)
    def forward(self, x, h_prev=None):
        return self.head(self.gru(self.embed(x))[0]), None

# ============================================================
def make_batch(bs, noise_len):
    targets = torch.randint(2, VOCAB, (bs,))
    noise = torch.randint(2, VOCAB, (bs, noise_len))
    end = torch.full((bs, 1), END, dtype=torch.long)
    x = torch.cat([targets.unsqueeze(1), noise, end], dim=1)
    y = torch.full_like(x, IGNORE); y[:, -1] = targets
    return x, y

def train_one(model, name, steps=3000, max_noise=64, bs=64):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best = float('inf')
    for step in range(1, steps + 1):
        x, y = make_batch(bs, random.randint(4, max_noise)); x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, -1, :], y[:, -1], ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if loss.item() < best: best = loss.item()
        if step % 500 == 0:
            print(f"  {name:>16s} step{step:5d} loss={loss.item():.4f} best={best:.4f}", flush=True)
    return best

@torch.no_grad()
def eval_one(model, distances, bs=128):
    model.eval()
    r = {}
    for dist in distances:
        c, t = 0, 0
        for _ in range(10):
            x, y = make_batch(bs, dist); x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            c += (logits[:, -1, :].argmax(-1) == y[:, -1]).sum().item(); t += bs
        r[dist] = c / t * 100
    return r

# ============================================================
print("=" * 75)
print("  CopyFirst: 8 Architectures @ ~250K params")
print("  FRSM(2-scale) vs cummax-based vs attention vs recurrent")
print("=" * 75)

torch.manual_seed(42)

all_models = [
    ("FRSM(2-scale)", MiniFRSM2()),
    ("OpenASH",       MiniOpenASH()),
    ("WDLM-Neural",   MiniWDLM()),
    ("WDLM-Real",     MiniWDLMReal()),
    ("Transformer",   MiniTransformer()),
    ("LSTM",          MiniLSTM()),
    ("GRU",           MiniGRU()),
    # ("FRSM(1-scale)", MiniFRSM()),  # removed - use 2-scale only
]

for name, m in all_models:
    print(f"  {name:>16s}: {sum(p.numel() for p in m.parameters()):,} params")
    m.to(device)

# 训练
print(f"\n  Training (3000 steps, noise 4-64)...")
best_losses = {}
for name, m in all_models:
    bl = train_one(m, name)
    best_losses[name] = bl

# 评估
test_dists = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
results = {}
for name, m in all_models:
    results[name] = eval_one(m, test_dists)

# 汇总表
print(f"\n{'='*75}")
print(f"  {'Dist':>6} | " + " | ".join([f"{n:>13}" for n, _ in all_models]))
print(f"  " + "-" * (8 + 15 * len(all_models)))
for d in test_dists:
    row = f"  {d:6d} | " + " | ".join([f"{results[n].get(d, 0):13.1f}" for n, _ in all_models])
    print(row)

# 泛化得分
print(f"\n  Generalization Score (avg acc @ 4K-131K):")
scores = []
for name, _ in all_models:
    far = sum(results[name].get(d, 0) for d in [4096, 8192, 16384, 32768, 65536, 131072]) / 6
    in_d = sum(results[name].get(d, 0) for d in [4, 8, 16, 32, 64]) / 5
    scores.append((name, far, in_d, best_losses[name]))
scores.sort(key=lambda x: x[1], reverse=True)
for name, far, in_d, bl in scores:
    lvl = "✓" if bl < 0.1 else ("△" if bl < 1.0 else "✗")
    print(f"  {name:>16s}  far={far:5.1f}%  in={in_d:5.1f}%  best_loss={bl:.4f}  [{lvl}]")

print(f"\nDone.")
