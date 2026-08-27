"""ARC-LLM: Adaptive Rank-Controlled LLM.
基于 K-Net 论文的 3 条结论: (R1) 有效秩作复杂度, (R2) 闭环秩调控器, (R3) 惊讶门控秩路由.

 backbone = 小 transformer (causal), 每层 FFN 换成 SGR-FFN:
   full-rank 路径 + low-rank 路径, 由每 token 惊讶门控 s_t 动态混合.
 复杂度 C = Σ_l stable_rank(W_l) (不是权重范数).
"""
import torch, torch.nn as nn, torch.nn.functional as F, math


@torch.no_grad()
def stable_rank(W):
    """stable rank = ‖W‖_F² / ‖W‖_2². 对 2D 权重; 越小=越低秩(越压缩)."""
    if W.dim() != 2: return 0.0
    Wf = W.detach().float()
    fro = (Wf ** 2).sum().item()
    spec = torch.linalg.svdvals(Wf)[0].item() ** 2
    return fro / (spec + 1e-12)


@torch.no_grad()
def model_stable_rank(m):
    """模型所有 2D 权重的平均 stable rank."""
    rs = [stable_rank(p) for p in m.parameters() if p.dim() == 2]
    return sum(rs) / max(len(rs), 1)


class SGRFFN(nn.Module):
    """Surprise-Gated Rank FFN: full-rank ⊕ low-rank, per-token 门控混合.
    use_sgr=False 时退化为标准 full-rank FFN (消融用)."""
    def __init__(self, d, d_ffn, r_low, use_sgr=True):
        super().__init__()
        self.use_sgr = use_sgr
        self.act = nn.SiLU()
        self.full_up = nn.Linear(d, d_ffn, bias=False)
        self.full_down = nn.Linear(d_ffn, d, bias=False)
        self.low_down = nn.Linear(d, r_low, bias=False)
        self.low_up = nn.Linear(r_low, d, bias=False)
        nn.init.normal_(self.low_down.weight, std=0.02)
        nn.init.normal_(self.low_up.weight, std=0.02)
        self.gate = nn.Linear(d, 1, bias=True)         # 惊讶门控
        nn.init.constant_(self.gate.bias, 0.0)

    def forward(self, x):
        full = self.full_down(self.act(self.full_up(x)))
        low = self.low_up(self.low_down(x))
        if not self.use_sgr:
            return full, None
        s = torch.sigmoid(self.gate(x))                 # (B,T,1) 惊讶度
        y = s * full + (1.0 - s) * low
        return y, s


class ARCBlock(nn.Module):
    """Pre-norm transformer block: causal MHA + SGR-FFN."""
    def __init__(self, d, n_heads, d_ffn, r_low, use_sgr=True):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.ffn = SGRFFN(d, d_ffn, r_low, use_sgr)
        self.d = d; self.h = n_heads; self.hd = d // n_heads

    def attn(self, x):
        B, T, d = x.shape; H, hd = self.h, self.hd
        q = self.q(x).view(B, T, H, hd).transpose(1, 2)
        k = self.k(x).view(B, T, H, hd).transpose(1, 2)
        v = self.v(x).view(B, T, H, hd).transpose(1, 2)
        a = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        a = a + mask
        a = F.softmax(a, dim=-1)
        out = (a @ v).transpose(1, 2).reshape(B, T, d)
        return self.o(out), a

    def forward(self, x):
        h = self.n1(x)
        ao, _ = self.attn(h)
        x = x + ao
        f, s = self.ffn(self.n2(x))
        x = x + f
        return x, s


class ARC_LLM(nn.Module):
    def __init__(self, vocab, d=128, n_heads=4, n_layers=3, d_ffn=512, r_low=32,
                 use_sgr=True, max_len=64):
        super().__init__()
        self.use_sgr = use_sgr
        self.em = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([ARCBlock(d, n_heads, d_ffn, r_low, use_sgr)
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        B, T = x.shape
        h = self.em(x) + self.pos(torch.arange(T, device=x.device))
        gates = []
        for blk in self.blocks:
            h, s = blk(h)
            if s is not None: gates.append(float(s.mean().detach()))
        h = self.norm(h)
        return self.head(h), gates

    def gate_mean(self, x):
        """带梯度的全部门控均值 (供 CLR 压有效秩用)."""
        h = self.em(x) + self.pos(torch.arange(x.size(1), device=x.device))
        s_sum = None
        for blk in self.blocks:
            h_in = blk.n1(h); ao, _ = blk.attn(h_in); h = h + ao
            x_in = blk.n2(h)
            s = torch.sigmoid(blk.ffn.gate(x_in))      # (B,T,1) 带梯度
            s_sum = s.mean() if s_sum is None else s_sum + s.mean()
            f, _ = blk.ffn(x_in); h = h + f
        return s_sum / len(self.blocks)

    @torch.no_grad()
    def gate_per_token(self, x):
        """每 token 的门控均值(跨层), (B, T). 用于真实 LM 上的 gate-惊讶度相关性分析."""
        h = self.em(x) + self.pos(torch.arange(x.size(1), device=x.device))
        acc = None
        for blk in self.blocks:
            h_in = blk.n1(h); ao, _ = blk.attn(h_in); h = h + ao
            x_in = blk.n2(h)
            if blk.ffn.use_sgr:
                s = torch.sigmoid(blk.ffn.gate(x_in)).squeeze(-1)   # (B,T)
                acc = s if acc is None else acc + s
            f, _ = blk.ffn(x_in); h = h + f
        n = max(sum(blk.ffn.use_sgr for blk in self.blocks), 1)
        return (acc / n) if acc is not None else torch.zeros(x.shape, device=x.device)


if __name__ == '__main__':
    m = ARC_LLM(vocab=115, use_sgr=True)
    n = sum(p.numel() for p in m.parameters())
    print(f'ARC-LLM params={n:,}  stable_rank(init)={model_stable_rank(m):.2f}')
    x = torch.randint(0, 115, (4, 3))
    logits, gates = m(x)
    print(f'forward: logits={logits.shape}  gates(layer means)={[round(g,3) for g in gates]}')
    print('OK')
