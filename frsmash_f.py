"""
FRSMASH-F — Fast层(线性递推) 替代 MaxStateSuper(cummax) + 慢记忆

对比 FRSMASH:
  FRSMASH:   OpenASH cummax 骨干 + 慢记忆  (cummax单调递增, 强LM但无法遗忘)
  FRSMASH-F: Fast线性递推骨干 + 慢记忆     (线性衰减, 可遗忘, 推理更简单)

速度优势: 线性递推用 cumprod+cumsum 全并行, 无 cummax 的排序开销
推理优势: 每层只需一个 (B,D) 状态向量, cummax 需要每头一个 (B,heads,d_head)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeedForward(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.ffn1 = nn.Linear(hidden_size, hidden_size)
        self.ffn2 = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.ffn2(self.ffn1(x) * self.relu(self.gate(x)))


class FastLayer(nn.Module):
    """线性递推层 — 替代 MaxStateSuper
    
    h_t = A_t * h_{t-1} + B_t  (A,B 只依赖输入, 可并行扫描)
    
    训练: cumprod + cumsum 全并行 O(T)
    推理: 单步 h = A*h + B, O(1)
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # 4个门: alpha(写入强度), forget, input, candidate
        self.gate_proj = nn.Linear(d_model, 4 * d_model)
        self.ffn = FeedForward(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def _parallel_scan(self, A, B):
        """并行求解 h_t = A_t*h_{t-1} + B_t
        A, B: (B, T, D)
        返回: (B, T, D)
        """
        A_safe = A.clamp(min=1e-4, max=1.0)
        A_cumprod = torch.cumprod(A_safe, dim=1)
        B_scaled = B / A_safe
        cumsum_B = torch.cumsum(B_scaled, dim=1)
        return A_cumprod * cumsum_B

    def forward(self, x):
        """训练: 全序列并行
        x: (B, T, D) → 返回 (B, T, D)
        """
        B, T, D = x.shape
        g = self.gate_proj(x).reshape(B, T, 4, D)
        alpha = torch.sigmoid(g[:, :, 0, :])
        f = torch.sigmoid(g[:, :, 1, :])
        i = torch.sigmoid(g[:, :, 2, :])
        cand = torch.tanh(g[:, :, 3, :])
        A = alpha * f + (1 - alpha)
        B_coeff = alpha * i * cand
        H = self._parallel_scan(A, B_coeff)  # (B, T, D)
        # FFN + 残差 + norm
        out = self.norm(self.alpha * self.ffn(H) + (1 - self.alpha) * x)
        return out

    @torch.no_grad()
    def step(self, x_t, h_prev):
        """推理: 单步 O(1)
        x_t: (B, D), h_prev: (B, D) → (out, h_new)
        """
        B, D = x_t.shape
        g = self.gate_proj(x_t).reshape(B, 4, D)
        alpha = torch.sigmoid(g[:, 0, :])
        f = torch.sigmoid(g[:, 1, :])
        i = torch.sigmoid(g[:, 2, :])
        cand = torch.tanh(g[:, 3, :])
        A = alpha * f + (1 - alpha)
        B = alpha * i * cand
        h_new = A * h_prev + B
        out = self.norm(self.alpha * self.ffn(h_new) + (1 - self.alpha) * x_t)
        return out, h_new


class SlowMemoryCell(nn.Module):
    """内容门控慢记忆 — 选择性写入"""
    def __init__(self, d_model):
        super().__init__()
        d = d_model
        self.W_forget = nn.Linear(d * 2, d)
        self.W_input = nn.Linear(d * 2, d)
        self.W_cand = nn.Linear(d * 2, d)
        nn.init.constant_(self.W_forget.bias, 1.0)
        nn.init.constant_(self.W_input.bias, -2.0)
        dh = max(d // 4, 1)
        self.gate = nn.Sequential(
            nn.Linear(d * 2, dh), nn.GELU(),
            nn.Linear(dh, 1), nn.Sigmoid()
        )

    def forward(self, x_t, h_prev):
        c = torch.cat([h_prev, x_t], dim=-1)
        f = torch.sigmoid(self.W_forget(c))
        i = torch.sigmoid(self.W_input(c))
        cand = f * h_prev + i * torch.tanh(self.W_cand(c))
        alpha = self.gate(c).squeeze(-1).unsqueeze(-1)
        return alpha * cand + (1 - alpha) * h_prev


class FRSMASH_F(nn.Module):
    """
    FRSMASH-F = Fast线性递推骨干 + 1慢记忆

    对比:
      FRSMASH:   MaxStateSuper(cummax) 骨干 — cummax单调递增, 强LM
      FRSMASH-F: FastLayer(线性递推) 骨干 — 线性衰减, 可遗忘, 更快

    参数:
        voc_size:     词表大小
        hidden_size:  隐藏维度
        num_layers:   Fast层数 (逻辑深度)
        K:            慢尺度更新周期 (默认 8)
    """
    def __init__(self, voc_size, hidden_size, num_layers, K=8):
        super().__init__()
        self.D = hidden_size
        self.K = K
        self.num_layers = num_layers

        self.em = nn.Embedding(voc_size, hidden_size, padding_idx=0)

        # Fast 骨干 (替代 OpenASH)
        self.fast_layers = nn.ModuleList([
            FastLayer(hidden_size) for _ in range(num_layers)
        ])
        self.backbone_norm = nn.LayerNorm(hidden_size)

        # 慢尺度记忆
        self.mem_input_proj = nn.Linear(hidden_size, hidden_size)
        self.slow_cell = SlowMemoryCell(hidden_size)

        # 融合
        self.mem_proj = nn.Linear(hidden_size, hidden_size)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        self.fusion_norm = nn.LayerNorm(hidden_size)

        self.head = nn.Linear(hidden_size, voc_size, bias=False)

    def forward(self, x):
        """训练: 全序列并行
        x: (B, T) token ids → (B, T, voc_size)
        """
        B, T = x.shape
        D = self.D
        x_emb = self.em(x)

        # Fast 骨干 (全并行, 残差堆叠)
        h = x_emb
        for layer in self.fast_layers:
            h = layer(h) + h  # 残差
        x_backbone = self.backbone_norm(h)

        # 慢尺度记忆 (每K步更新)
        inp_seq = self.mem_input_proj(x_emb)
        h_slow = torch.zeros(B, D, device=x.device)
        H_slow = torch.zeros(B, T, D, device=x.device)
        prev = 0
        for t in range(0, T, self.K):
            h_slow = self.slow_cell(inp_seq[:, t], h_slow)
            H_slow[:, prev:t + 1] = h_slow.unsqueeze(1)
            prev = t + 1
        if prev < T:
            H_slow[:, prev:] = h_slow.unsqueeze(1)
        x_mem = self.mem_proj(H_slow)

        # 门控融合
        cat = torch.cat([x_backbone, x_mem], dim=-1)
        gate = self.fusion_gate(cat)
        fused = self.fusion_norm(gate * x_backbone + (1 - gate) * x_mem + x_emb)

        return self.head(fused)

    @torch.no_grad()
    def generate_step(self, token_id, fast_states, h_slow):
        """推理: 单步 O(1)
        token_id: (B, 1)
        fast_states: list of (B, D), 每层一个
        h_slow: (B, D)
        返回: logits, new_fast_states, new_h_slow
        """
        x = self.em(token_id)  # (B, 1, D)
        x_t = x[:, 0]  # (B, D)

        # Fast 骨干逐层
        h = x_t
        new_states = []
        for i, layer in enumerate(self.fast_layers):
            out, s = layer.step(h, fast_states[i])
            h = out + h  # 残差
            new_states.append(s)
        x_backbone = self.backbone_norm(h)

        # 慢尺度
        inp = self.mem_input_proj(x_t)
        h_slow_new = self.slow_cell(inp, h_slow)
        x_mem = self.mem_proj(h_slow_new)

        # 融合
        cat = torch.cat([x_backbone, x_mem], dim=-1)
        gate = self.fusion_gate(cat)
        fused = self.fusion_norm(gate * x_backbone + (1 - gate) * x_mem + x_t)
        logits = self.head(fused)

        return logits, new_states, h_slow_new


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V = 23005

    # 速度对比
    from frsmash import FRSMASH as FRSMASH_ASH

    for H, L, B in [(512, 4, 88), (512, 6, 64), (640, 4, 64)]:
        # FRSMASH-F (Fast层)
        m_f = FRSMASH_F(V, H, L, K=8).to(device)
        p_f = sum(pp.numel() for pp in m_f.parameters()) / 1e6
        m_f.train()
        x = torch.randint(1, V, (B, 384), device=device)
        t = torch.randint(1, V, (B, 384), device=device)
        opt = torch.optim.AdamW(m_f.parameters(), lr=1e-4)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        loss = F.cross_entropy(m_f(x).reshape(-1, V), t.reshape(-1), ignore_index=0)
        loss.backward(); opt.step()
        torch.cuda.synchronize()
        tok_f = B * 384 / (time.time() - t0)
        mem_f = torch.cuda.max_memory_allocated() / 1e9

        # FRSMASH (OpenASH)
        m_a = FRSMASH_ASH(V, H, 8, L, K=8).to(device)
        p_a = sum(pp.numel() for pp in m_a.parameters()) / 1e6
        m_a.train()
        opt2 = torch.optim.AdamW(m_a.parameters(), lr=1e-4)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        loss2 = F.cross_entropy(m_a(x).reshape(-1, V), t.reshape(-1), ignore_index=0)
        loss2.backward(); opt2.step()
        torch.cuda.synchronize()
        tok_a = B * 384 / (time.time() - t0)
        mem_a = torch.cuda.max_memory_allocated() / 1e9

        print(f"H={H} L={L} B={B}:")
        print(f"  FRSMASH-F:  {p_f:.0f}M  {tok_f:>6.0f} tok/s  {mem_f:.1f}GB")
        print(f"  FRSMASH-A:  {p_a:.0f}M  {tok_a:>6.0f} tok/s  {mem_a:.1f}GB")
        print(f"  F/A speedup: {tok_f/tok_a:.2f}x")
        del m_f, m_a, opt, opt2; torch.cuda.empty_cache()

    # 推理对比
    print("\n推理速度:")
    m_f = FRSMASH_F(V, 512, 4, K=8).to(device)
    m_f.eval()
    fs = [torch.zeros(1, 512, device=device) for _ in range(4)]
    hs = torch.zeros(1, 512, device=device)
    tok = torch.tensor([[42]], device=device)
    for _ in range(20):
        lg, fs, hs = m_f.generate_step(tok, fs, hs)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(1000):
        lg, fs, hs = m_f.generate_step(tok, fs, hs)
    torch.cuda.synchronize()
    print(f"  FRSMASH-F: {1000/(time.time()-t0):.0f} tok/s ({(time.time()-t0)/1000*1000:.2f} ms/tok)")

    m_a = FRSMASH_ASH(V, 512, 8, 4, K=8).to(device)
    m_a.eval()
    states = [None] * 4
    hs2 = torch.zeros(1, 512, device=device)
    for _ in range(20):
        lg, states, hs2 = m_a.generate_step(tok, states, hs2)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(1000):
        lg, states, hs2 = m_a.generate_step(tok, states, hs2)
    torch.cuda.synchronize()
    print(f"  FRSMASH-A: {1000/(time.time()-t0):.0f} tok/s ({(time.time()-t0)/1000*1000:.2f} ms/tok)")
