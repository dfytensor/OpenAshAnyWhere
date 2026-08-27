"""CopyFirst 新机制 v2: OpenASH v1 (cummax) + Max-Age 追踪器 + 结构化读出.

机制: M = 硬 cummax (STE 软梯度); A = 通道 age (当前最大值存活步数)。
读出: age 门控挑出第一个 token 幸存的高龄通道 -> 与全部 token 键做关联查找。
"""
import torch, torch.nn as nn, torch.nn.functional as F
import random, time

device = torch.device("cuda")
VOCAB = 32; END = 0; IGNORE = 1
BETA = 8.0


class OpenASHAge2(nn.Module):
    def __init__(self, d=512, beta=BETA, amp=2.0):
        super().__init__()
        self.d = d
        self.beta = beta
        self.amp = amp                    # 首写倍增: 第一个 token 的键放大写入
        self.embed = nn.Embedding(VOCAB, d)
        self.Wk = nn.Linear(d, d)
        self.gamma = nn.Parameter(torch.tensor(0.1))   # age 门控锐度
        self.theta = nn.Parameter(torch.tensor(8.0))   # age 门控阈值
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)

    def forward(self, x, h_prev=None):
        B, T = x.shape
        if h_prev is None:
            M = torch.zeros(B, self.d, device=device)
            A = torch.zeros(B, self.d, device=device)
        else:
            M, A = h_prev
        x_e = self.embed(x)
        K = self.Wk(x_e)                    # [B, T, d]
        outs = []
        for t in range(T):
            k = K[:, t]
            if t == 0:
                k = k * (1.0 + self.amp)    # 首写倍增 (结构性生存保障)
            s = torch.sigmoid(self.beta * (k - M))
            M_soft = M + s * (k - M)
            M_hard = torch.maximum(M, k)
            M = M_hard + (M_soft - M_hard).detach()
            A_soft = (A + 1.0) * (1 - s)
            A_hard = torch.where(k > M_hard, torch.zeros_like(A), A + 1.0)
            A = A_hard + (A_soft - A_hard).detach()
            # 结构化读出: age 门控 + 关联查找
            gate = torch.sigmoid(self.gamma * (A - self.theta))   # 高龄通道 -> 1
            sig = self.ln(M * gate)
            outs.append(self.head(sig).unsqueeze(1))
        return torch.cat(outs, 1), (M, A)


def make_batch(bs, noise_len):
    targets = torch.randint(2, VOCAB, (bs,))
    noise = torch.randint(2, VOCAB, (bs, noise_len))
    end = torch.full((bs, 1), END, dtype=torch.long)
    x = torch.cat([targets.unsqueeze(1), noise, end], 1)
    y = torch.full_like(x, IGNORE)
    y[:, -1] = targets
    return x, y


def train_and_eval(model, name, steps=2000, max_noise=128, bs=32):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best = float('inf')
    t0 = time.time()
    for st in range(1, steps + 1):
        nl = random.randint(4, max_noise)
        x, y = make_batch(bs, nl)
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, -1, :], y[:, -1], ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if loss.item() < best: best = loss.item()
    print(f"  {name}: best_loss={best:.5f}  time={time.time()-t0:.0f}s", flush=True)

    model.eval()
    test_dists = [4, 64, 512, 2048, 8192, 16384, 32768, 65536]
    results = {}
    for d in test_dists:
        eb = 64 if d <= 8192 else 16
        n_batch = 2 if d <= 32768 else 1
        c = 0; total = eb * n_batch
        for _ in range(n_batch):
            x, y = make_batch(eb, d)
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            c += (logits[:, -1, :].argmax(-1) == y[:, -1]).sum().item()
        results[d] = c / total * 100
    return best, results


if __name__ == "__main__":
    torch.manual_seed(42)
    print("OpenASH v1 基线: best_loss=1.6 (cummax 失败)")
    print("=" * 60)
    m = OpenASHAge2(d=512).to(device)
    print(f"OpenASHAge2: {sum(p.numel() for p in m.parameters()):,} params")
    bl, acc = train_and_eval(m, "OpenASHAge2", steps=2000, max_noise=128, bs=32)
    dists = [4, 64, 512, 2048, 8192, 16384, 32768, 65536]
    print(f"\n{'Dist':>7} | Acc%")
    print("-" * 24)
    for d in dists:
        print(f"{d:7,d} | {acc[d]:.1f}")
    near = sum(acc[d] for d in [4, 64]) / 2
    far = sum(acc[d] for d in [8192, 16384, 32768, 65536]) / 4
    print(f"\nbest_loss={bl:.5f}  near={near:.1f}%  far={far:.1f}%  gap={far-near:+.1f}%")
