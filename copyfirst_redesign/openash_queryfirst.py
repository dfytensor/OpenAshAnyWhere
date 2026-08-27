"""CopyAnywhere-QueryFirst: 查询前置, 目标任意位置.

序列: [QUERY 位置P] [噪声... 目标在P ... 噪声] [END]  -> 输出位置P的token.
机制 (OpenASH cummax 家族, 无禁术):
  1. 时钟寄存器: 记录当前位置
  2. 查询门控放大写: t==P 时, 该 token 的键 ×(1+amp) 写入 cummax -> 结构性不可覆盖
  3. 幅度门控读出: END 时 M > theta 的通道 = 被查询 token 的签名 -> 关联查找
"""
import torch, torch.nn as nn, torch.nn.functional as F
import random, time

device = torch.device("cuda")
VOCAB = 32; END = 0; IGNORE = 1; QUERY = 1
BETA = 8.0


class OpenASHQueryFirst(nn.Module):
    def __init__(self, d=512, beta=BETA, amp=2.0):
        super().__init__()
        self.d = d
        self.beta = beta
        self.amp = amp
        self.embed = nn.Embedding(VOCAB, d)
        self.Wk = nn.Linear(d, d)
        self.gamma = nn.Parameter(torch.tensor(0.5))   # 幅度门控锐度
        self.theta = nn.Parameter(torch.tensor(1.0))   # 幅度门控阈值
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)

    def forward(self, x, P, h_prev=None):
        """x: [B,T] (位置0为QUERY, 位置1..P为噪声, 位置P为目标); P: [B] 目标位置."""
        B, T = x.shape
        if h_prev is None:
            M = torch.zeros(B, self.d, device=device)
        else:
            M = h_prev
        x_e = self.embed(x)
        K = self.Wk(x_e)
        outs = []
        for t in range(1, T):                      # 从位置1开始 (0是QUERY)
            k = K[:, t]
            is_query_pos = (t == P).float().view(-1, 1)
            k = k * (1.0 + self.amp * is_query_pos)  # 查询位置放大写
            s = torch.sigmoid(self.beta * (k - M))
            M_soft = M + s * (k - M)
            M_hard = torch.maximum(M, k)
            M = M_hard + (M_soft - M_hard).detach()
        gate = torch.sigmoid(self.gamma * (M - self.theta))   # 幅度门控: 放大通道 -> 1
        sig = self.ln(M * gate)
        logits = self.head(sig)
        return logits, M


def make_batch(bs, total_len):
    """位置0=QUERY(1), 位置1=噪声, 目标在随机 P∈[1, total_len-1], 末尾 END.
    实际内容从位置2开始 (位置1 当噪声用)."""
    targets = torch.randint(2, VOCAB, (bs,))
    P = torch.randint(1, total_len - 1, (bs,))
    x = torch.randint(2, VOCAB, (bs, total_len + 1))
    x[:, 0] = QUERY
    for i in range(bs):
        x[i, P[i]] = targets[i]
    x[:, total_len] = END
    return x, targets, P


def train_and_eval(model, name, steps=3000, max_len=128, bs=32):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best = float('inf')
    t0 = time.time()
    for st in range(1, steps + 1):
        L = random.randint(8, max_len)
        x, y, P = make_batch(bs, L)
        x, y, P = x.to(device), y.to(device), P.to(device)
        logits, _ = model(x, P)
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
        L = g + 1 + random.randint(0, 64)
        eb = 64 if g <= 8192 else 16
        n_batch = 2 if g <= 32768 else 1
        c = 0; total = eb * n_batch
        for _ in range(n_batch):
            x, y, P = make_batch(eb, L)
            P = torch.full((eb,), L - 1 - g, dtype=torch.long)   # 目标距 END 恰 g 步
            for i in range(eb):
                x[i, P[i]] = y[i]
            x, y, P = x.to(device), y.to(device), P.to(device)
            logits, _ = model(x, P)
            c += (logits.argmax(-1) == y).sum().item()
        results[g] = c / total * 100
    return best, results


if __name__ == "__main__":
    torch.manual_seed(42)
    m = OpenASHQueryFirst(d=512).to(device)
    print(f"OpenASHQueryFirst: {sum(p.numel() for p in m.parameters()):,} params")
    bl, acc = train_and_eval(m, "QueryFirst", steps=3000, max_len=128, bs=32)
    gaps = [4, 64, 512, 2048, 8192, 16384, 32768, 65536]
    for g in gaps:
        print(f"{g:7,d} | {acc[g]:.1f}")
    near = sum(acc[g] for g in [4, 64]) / 2
    far = sum(acc[g] for g in [8192, 16384, 32768, 65536]) / 4
    print(f"\nbest_loss={bl:.5f}  near={near:.1f}%  far={far:.1f}%  gap={far-near:+.1f}%")
