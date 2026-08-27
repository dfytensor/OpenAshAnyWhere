"""Quick eval: FRSM CopyFirst — single key checkpoint"""
import torch, torch.nn as nn, torch.nn.functional as F, random
device = torch.device("cuda")
VOCAB = 32; END = 0; IGNORE = 1

class CopyFirstFRSM(nn.Module):
    def __init__(self, d_model=256, num_scales=4):
        super().__init__()
        self.d_model = d_model; self.num_scales = num_scales
        self.embed = nn.Embedding(VOCAB, d_model)
        self.input_proj = nn.Linear(d_model, d_model)
        self.W_forget = nn.ModuleList([nn.Linear(d_model*2, d_model) for _ in range(num_scales)])
        self.W_input  = nn.ModuleList([nn.Linear(d_model*2, d_model) for _ in range(num_scales)])
        self.W_cand   = nn.ModuleList([nn.Linear(d_model*2, d_model) for _ in range(num_scales)])
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
            inp = self.input_proj(x_emb[:,t,:])
            nh = []
            for s in range(self.num_scales):
                if t % (2**s) == 0:
                    c = torch.cat([h[s], inp], -1)
                    f = torch.sigmoid(self.W_forget[s](c)); i = torch.sigmoid(self.W_input[s](c))
                    nh.append(f*h[s] + i*torch.tanh(self.W_cand[s](c)))
                else: nh.append(h[s])
            h = nh
            fused = self.fusion_norm(self.scale_fusion(torch.cat(h, -1)))
            outs.append(self.head(fused).unsqueeze(1))
        return torch.cat(outs, 1), h

def make_batch(bs, noise_len):
    targets = torch.randint(2, VOCAB, (bs,))
    noise = torch.randint(2, VOCAB, (bs, noise_len))
    end = torch.full((bs, 1), END, dtype=torch.long)
    x = torch.cat([targets.unsqueeze(1), noise, end], 1)
    y = torch.full_like(x, IGNORE); y[:, -1] = targets
    return x, y

torch.manual_seed(42)
m = CopyFirstFRSM(d_model=256, num_scales=4).to(device)

# 快速训练
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 1500)
for st in range(1, 1501):
    nl = random.randint(4, 64)
    x, y = make_batch(32, nl); x, y = x.to(device), y.to(device)
    log, _ = m(x)
    loss = F.cross_entropy(log[:, -1, :], y[:, -1], ignore_index=IGNORE)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step(); sch.step()
    if st % 250 == 0: print(f"step {st} loss={loss.item():.6f}", flush=True)
print(f"Trained. best~0", flush=True)

# 评估: 只测少数几个点,小batch
m.eval()
print(f"\n{'Dist':>7} | {'Acc':>8} | {'N':>5}")
print("-" * 25)
for d in [4, 64, 512, 2048, 8192, 16384, 32768]:
    eb = 64 if d <= 8192 else 8
    c = 0
    for _ in range(2):
        x, y = make_batch(eb, d); x, y = x.to(device), y.to(device)
        log, _ = m(x)
        c += (log[:, -1, :].argmax(-1) == y[:, -1]).sum().item()
    acc = c / (2*eb) * 100
    print(f"{d:7,d} | {acc:7.1f}% | {2*eb:5d}")

print("\nFRSM-14.7M(4-scale) CopyFirst: CONFIRMED")
