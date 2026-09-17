# Rubik-GLA 核心模型实现
# S_t = exp(skew(W_g x_t)) @ S_{t-1} + k_t^T v_t ;  o_t = q_t S_t
# 基线: GLA (逐元素门控) / GRU / Transformer
import math
import torch
import torch.nn as nn


class RubikLayer(nn.Module):
    """非交换矩阵门控: G_t = exp(skew(W x_t)) in SO(dh), 状态 (B,H,dh,dh).
    decay=True: S_t = λ⊙(G_t S_{t-1}) + kv^T, λ 逐头逐通道标量 (非交换性保留)."""
    def __init__(self, d, H=4, decay=False):
        super().__init__()
        self.H, self.dh = H, d // H
        self.decay = decay
        self.ln_in = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.skew = nn.Linear(d, H * self.dh * self.dh)
        nn.init.normal_(self.skew.weight, 0.0, 0.01)   # A 小 => G ≈ I 起步
        nn.init.zeros_(self.skew.bias)
        if decay:
            self.lam = nn.Parameter(torch.full((H, self.dh, 1), -1.0))  # sigmoid(-1)≈0.27

    def forward(self, x, state=None):
        B, L, _ = x.shape
        H, dh = self.H, self.dh
        x = self.ln_in(x)
        q = self.q(x).view(B, L, H, dh)
        k = self.k(x).view(B, L, H, dh)
        v = self.v(x).view(B, L, H, dh)
        M = self.skew(x).view(B, L, H, dh, dh)
        A = M - M.transpose(-1, -2)                     # 反对称
        G = torch.matrix_exp(A)                         # (B,L,H,dh,dh) in SO(dh)
        if state is None:
            state = x.new_zeros(B, H, dh, dh)
        lam = torch.sigmoid(self.lam).unsqueeze(0) if self.decay else None
        outs = []
        for t in range(L):
            write = k[:, t].view(B, H, dh, 1) * v[:, t].view(B, H, 1, dh)
            state = G[:, t] @ state
            if self.decay:
                state = lam * state
            state = state + write
            o = (q[:, t].view(B, H, 1, dh) @ state).squeeze(-2)   # (B,H,dh)
            outs.append(o.reshape(B, H * dh))
        return torch.stack(outs, 1), state


class GLALayer(nn.Module):
    """逐元素(可交换)门控基线: S_t = g ⊙ S_{t-1} + k v^T."""
    def __init__(self, d, H=4):
        super().__init__()
        self.H, self.dh = H, d // H
        self.ln_in = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.gate = nn.Linear(d, H * self.dh)
        nn.init.zeros_(self.gate.weight)
        nn.init.ones_(self.gate.bias)                   # sigmoid(1)≈0.73 温和衰减

    def forward(self, x, state=None):
        B, L, _ = x.shape
        H, dh = self.H, self.dh
        x = self.ln_in(x)
        q = self.q(x).view(B, L, H, dh)
        k = self.k(x).view(B, L, H, dh)
        v = self.v(x).view(B, L, H, dh)
        g = torch.sigmoid(self.gate(x)).view(B, L, H, dh, 1)
        if state is None:
            state = x.new_zeros(B, H, dh, dh)
        outs = []
        for t in range(L):
            write = k[:, t].view(B, H, dh, 1) * v[:, t].view(B, H, 1, dh)
            state = g[:, t] * state + write
            o = (q[:, t].view(B, H, 1, dh) @ state).squeeze(-2)
            outs.append(o.reshape(B, H * dh))
        return torch.stack(outs, 1), state


class LM(nn.Module):
    """统一外壳: emb -> 4 层 -> ln -> head. kind: rubik/gla/gru/tf."""
    def __init__(self, V, d=128, kind="rubik", layers=4, H=4, max_len=512):
        super().__init__()
        self.kind = kind
        self.emb = nn.Embedding(V, d)
        if kind == "rubik":
            self.stack = nn.ModuleList([RubikLayer(d, H, decay=True) for _ in range(layers)])
        elif kind == "gla":
            self.stack = nn.ModuleList([GLALayer(d, H) for _ in range(layers)])
        elif kind == "gru":
            self.gru = nn.GRU(d, d, num_layers=layers, batch_first=True)
        elif kind == "tf":
            self.pos = nn.Embedding(max_len, d)
            layer = nn.TransformerEncoderLayer(d, nhead=H, dim_feedforward=4 * d,
                                               batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, num_layers=layers)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, ids):
        B, L = ids.shape
        x = self.emb(ids)
        if self.kind in ("rubik", "gla"):
            for layer in self.stack:
                x, _ = layer(x)
        elif self.kind == "gru":
            x, _ = self.gru(x)
        else:
            pos = self.pos(torch.arange(L, device=ids.device)).unsqueeze(0)
            x = x + pos
            mask = torch.triu(torch.ones(L, L, device=ids.device, dtype=torch.bool), 1)
            x = self.tf(x, mask=mask)
        return self.head(self.ln(x))
