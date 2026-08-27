"""CopyAnywhere: 目标在任意位置 P, END 时按查询年龄 q=L-P 检索.

机制: OpenASH v1 cummax + Max-Age 寄存器 (通道地址=写入时刻的age)
  - 随龄抗写: 通道越老, 覆盖所需超出量越大 (保护任意位置的已写入通道)
  - 高斯查询读出: gate = exp(-((A-q)/sigma)^2), 挑出 age=q 的通道 -> M*gate -> 查找
"""
import torch, torch.nn as nn, torch.nn.functional as F
import random, time

device = torch.device("cuda")
VOCAB = 32; END = 0; IGNORE = 1
BETA = 8.0


class OpenASHAgeAny(nn.Module):
    def __init__(self, d=512, beta=BETA, m_max=2.0, gamma_a=0.5, theta_a=2.0):
        super().__init__()
        self.d = d
        self.beta = beta
        self.m_max = m_max          # 抗写余量 (写后 2 步内进入不可覆盖)
        self.gamma_a = gamma_a
        self.theta_a = theta_a
        self.embed = nn.Embedding(VOCAB, d)
        self.Wk = nn.Linear(d, d)
        self.sigma = 3.0   # 查询带宽 (训练中退火, 见 train_and_eval)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)

    def forward(self, x, q, h_prev=None):
        """x: [B, T]; q: [B] 查询年龄 (要检索的通道 age)."""
        B, T = x.shape
        if h_prev is None:
            M = torch.zeros(B, self.d, device=device)
            A = torch.zeros(B, self.d, device=device)
        else:
            M, A = h_prev
        x_e = self.embed(x)
        K = self.Wk(x_e)
        outs = []
        for t in range(T):
            k = K[:, t]
            margin = self.m_max * torch.sigmoid(self.gamma_a * (A - self.theta_a))
            upd = k > M + margin                            # 用更新前的 M 判断
            s = torch.sigmoid(self.beta * (k - M - margin))
            M_soft = M + s * (k - M)
            M_hard = torch.maximum(M, k)
            M = M_hard + (M_soft - M_hard).detach()
            A_soft = (A + 1.0) * (1 - s)
            A_hard = torch.where(upd, torch.zeros_like(A), A + 1.0)
            A = A_hard + (A_soft - A_hard).detach()
        # 高斯查询读出 (只在最后一位用)
        qv = q.view(-1, 1)
        gate = torch.exp(-((A - qv) / max(self.sigma, 0.5)) ** 2)
        sig = self.ln(M * gate)
        logits = self.head(sig)
        return logits, (M, A)


def make_batch(bs, total_len, gap):
    """目标放在距 END 恰好 gap 步的位置 P = total_len-1-gap (绝对位置随 total_len 变化).
    序列: [噪声... target在P... 噪声] END. 查询 q = gap."""
    targets = torch.randint(2, VOCAB, (bs,))
    P = total_len - 1 - gap
    assert P >= 0
    x = torch.randint(2, VOCAB, (bs, total_len))
    for i in range(bs):
        x[i, P] = targets[i]
    x = torch.cat([x, torch.full((bs, 1), END, dtype=torch.long)], 1)
    q = torch.full((bs,), float(gap))
    return x, targets, q


def train_and_eval(model, name, steps=3000, max_len=128, bs=32):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best = float('inf')
    t0 = time.time()
    for st in range(1, steps + 1):
        model.sigma = 3.0 - (3.0 - 0.5) * (st / steps)   # 带宽退火: 粗 -> 精
        L = random.randint(8, max_len)
        gap = random.randint(4, L - 1)
        x, y, q = make_batch(bs, L, gap)
        x, y, q = x.to(device), y.to(device), q.to(device)
        logits, _ = model(x, q)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if loss.item() < best: best = loss.item()
    print(f"  {name}: best_loss={best:.5f}  time={time.time()-t0:.0f}s", flush=True)

    model.eval()
    test_gaps = [4, 64, 512, 2048, 8192, 16384, 32768, 65536]
    results = {}
    for g in test_gaps:
        eb = 64 if g <= 8192 else 16
        n_batch = 2 if g <= 32768 else 1
        c = 0; total = eb * n_batch
        for _ in range(n_batch):
            L = g + 1 + random.randint(0, 64)     # 绝对位置随机
            x, y, q = make_batch(eb, L, g)
            x, y, q = x.to(device), y.to(device), q.to(device)
            logits, _ = model(x, q)
            c += (logits.argmax(-1) == y).sum().item()
        results[g] = c / total * 100
    return best, results


if __name__ == "__main__":
    torch.manual_seed(42)
    m = OpenASHAgeAny(d=512).to(device)
    print(f"OpenASHAgeAny: {sum(p.numel() for p in m.parameters()):,} params")
    bl, acc = train_and_eval(m, "AgeAny", steps=3000, max_len=128, bs=32)
    gaps = [4, 64, 512, 2048, 8192, 16384, 32768, 65536]
    print(f"\n{'Gap':>7} | Acc%")
    print("-" * 24)
    for g in gaps:
        print(f"{g:7,d} | {acc[g]:.1f}")
    near = sum(acc[g] for g in [4, 64]) / 2
    far = sum(acc[g] for g in [8192, 16384, 32768, 65536]) / 4
    print(f"\nbest_loss={bl:.5f}  near={near:.1f}%  far={far:.1f}%  gap={far-near:+.1f}%")
