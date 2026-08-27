"""
极小规模长期依赖 V3: 修复 FRSM gating + 正确激活
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random

device = torch.device("cuda")

# ============================================================
class MiniFRSM_V3(nn.Module):
    """带门控的 FRSM - gating 可以学会"不覆盖"重要信息"""
    def __init__(self, vocab_size=32, d_model=128, num_scales=2):
        super().__init__()
        self.d_model, self.num_scales = d_model, num_scales
        self.embed = nn.Embedding(vocab_size, d_model)
        self.input_proj = nn.Linear(d_model, d_model)
        
        # 每个尺度: 输入门 + 遗忘门 + 候选
        self.W_inp = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(num_scales)])
        self.W_forget = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(num_scales)])
        self.W_cand = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(num_scales)])
        
        self.fusion = nn.Linear(d_model * num_scales, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        # 初始化: forget gate 偏向 1 (记住), input gate 偏向 0 (不写入)
        for w in self.W_forget:
            nn.init.constant_(w.bias, 1.0)
        for w in self.W_inp:
            nn.init.constant_(w.bias, -2.0)

    def forward(self, x, h_prev=None):
        B, T = x.shape
        if h_prev is None:
            h = [torch.zeros(B, self.d_model, device=x.device) for _ in range(self.num_scales)]
        else:
            h = [hs.clone() for hs in h_prev]
        
        x_emb = self.embed(x)
        outs = []
        for t in range(T):
            inp = self.input_proj(x_emb[:, t, :])
            nh = []
            for s in range(self.num_scales):
                period = 2 ** s
                if t % period == 0:
                    combined = torch.cat([h[s], inp], dim=-1)
                    f_gate = torch.sigmoid(self.W_forget[s](combined))
                    i_gate = torch.sigmoid(self.W_inp[s](combined))
                    cand = torch.tanh(self.W_cand[s](combined))
                    nh.append(f_gate * h[s] + i_gate * cand)
                else:
                    nh.append(h[s])
            h = nh
            fused = self.ln(self.fusion(torch.cat(h, dim=-1)))
            outs.append(self.head(fused).unsqueeze(1))
        return torch.cat(outs, dim=1), h

class MiniLSTM(nn.Module):
    def __init__(self, vocab_size=32, d_model=128, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers, batch_first=True)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x, h_prev=None):
        return self.head(self.lstm(self.embed(x))[0]), None

# ============================================================
VOCAB = 32
END_TOKEN = 0; IGNORE = 1

def make_batch(bs, noise_len):
    targets = torch.randint(2, VOCAB, (bs,))
    noise = torch.randint(2, VOCAB, (bs, noise_len))
    end = torch.full((bs, 1), END_TOKEN, dtype=torch.long)
    x = torch.cat([targets.unsqueeze(1), noise, end], dim=1)
    y = torch.full_like(x, IGNORE)
    y[:, -1] = targets
    return x, y

def train_model(model, name, steps=5000, max_noise=64, bs=64):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hist = []
    print(f"\n  [{name}] Training...", flush=True)
    
    for step in range(1, steps + 1):
        nl = random.randint(4, max_noise)
        x, y = make_batch(bs, nl)
        x, y = x.to(device), y.to(device)
        
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, -1, :], y[:, -1], ignore_index=IGNORE)
        
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        hist.append(loss.item())
        
        if step % 500 == 0:
            avg = sum(hist[-200:])/len(hist[-200:])
            print(f"    step {step:5d}  loss={avg:.4f}", flush=True)
    return hist

@torch.no_grad()
def eval_acc(model, distances, bs=128):
    model.eval()
    results = {}
    for dist in distances:
        c, t = 0, 0
        for _ in range(10):
            x, y = make_batch(bs, dist)
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            p = logits[:, -1, :].argmax(dim=-1)
            c += (p == y[:, -1]).sum().item(); t += bs
        results[dist] = c / t * 100
    return results

# ============================================================
print("=" * 65)
print("  CopyFirst Task (Gated FRSM vs LSTM)")
print("=" * 65)

frsm = MiniFRSM_V3(VOCAB, 128, num_scales=2).to(device)
lstm = MiniLSTM(VOCAB, 128, num_layers=2).to(device)
print(f"  FRSM: {sum(p.numel() for p in frsm.parameters()):,}  LSTM: {sum(p.numel() for p in lstm.parameters()):,}")

train_model(frsm, "FRSM", steps=5000, max_noise=64)
train_model(lstm, "LSTM", steps=5000, max_noise=64)

test_dists = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
f_acc = eval_acc(frsm, test_dists)
l_acc = eval_acc(lstm, test_dists)

print(f"\n{'='*65}")
print(f"  {'Dist':>5} | {'FRSM':>8} | {'LSTM':>8} | {'Winner':>8}")
print(f"  " + "-" * 40)
for d in test_dists:
    f, l = f_acc[d], l_acc[d]
    w = "FRSM" if f > l + 3 else ("LSTM" if l > f + 3 else "TIE")
    print(f"  {d:5d} | {f:7.1f}% | {l:7.1f}% | {w:>8}")

in_f = sum(f_acc[d] for d in [4,8,16,32,64]) / 5
out_f = sum(f_acc[d] for d in test_dists if d > 64) / max(1, sum(1 for d in test_dists if d > 64))
in_l = sum(l_acc[d] for d in [4,8,16,32,64]) / 5
out_l = sum(l_acc[d] for d in test_dists if d > 64) / max(1, sum(1 for d in test_dists if d > 64))

print(f"\n  FRSM in-dist(4-64): {in_f:.1f}%  →  out-dist(128-8192): {out_f:.1f}%  (gap: {out_f-in_f:+.1f}%)")
print(f"  LSTM  in-dist(4-64): {in_l:.1f}%  →  out-dist(128-8192): {out_l:.1f}%  (gap: {out_l-in_l:+.1f}%)")

# 极限测试: Push to failure
print(f"\n  Extreme push test:")
for d in [16384, 32768, 65536, 131072]:
    try:
        fe = eval_acc(frsm, [d], bs=32)[d]
        le = eval_acc(lstm, [d], bs=32)[d]
        print(f"  {d:6d} | {fe:7.1f}% (FRSM) | {le:7.1f}% (LSTM)")
    except Exception as e:
        print(f"  {d:6d} | OOM or error: {e}")

print(f"\nDone.")
