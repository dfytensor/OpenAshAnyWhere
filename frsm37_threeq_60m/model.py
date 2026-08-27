"""
FRSMASH v3.7 (DirectAdd) — 自包含版, 适配 OpenASHVoc 词表.
源: github.com/dfytensor/frsmash3.7  (v3.6 base + DirectAdd fusion)

架构: 多槽 SSM 骨干 + 线性 SlowMemory + fla multi-head GLA recall, DirectAdd 融合.
    fused = norm(x_ash + x_mem + x_emb) + x_recall   (无 gate, 强制各路贡献)

运行需: PyTorch+CUDA, fla(flash-linear-attention), triton. Windows 需 PYTHONUTF8=1.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.ops.hgrn import chunk_hgrn
from fla.ops.gla import chunk_gla


# ============================================================
# GLA Recall — fla multi-head gated linear attention (content addressing)
# ============================================================
class GlaRecall(nn.Module):
    def __init__(self, d, heads=8, d_h=64):
        super().__init__()
        self.heads, self.d_h = heads, d_h
        self.q_proj = nn.Linear(d, heads * d_h, bias=False)
        self.k_proj = nn.Linear(d, heads * d_h, bias=False)
        self.v_proj = nn.Linear(d, heads * d_h, bias=False)
        self.g_proj = nn.Linear(d, heads * d_h, bias=True)
        self.out_proj = nn.Linear(heads * d_h, d, bias=False)
        nn.init.constant_(self.g_proj.bias, 8.0)

    def forward(self, x, initial_state=None, return_state=False):
        B, T, d = x.shape
        H, K = self.heads, self.d_h
        q = self.q_proj(x).view(B, T, H, K)
        k = self.k_proj(x).view(B, T, H, K)
        v = self.v_proj(x).view(B, T, H, K)
        g = F.logsigmoid(self.g_proj(x).float()).view(B, T, H, K)
        out, st = chunk_gla(q, k, v, g, initial_state=initial_state, output_final_state=return_state)
        out = self.out_proj(out.view(B, T, H * K))
        return (out, st) if return_state else out

    @torch.no_grad()
    def step(self, x_t, S_prev):
        B = x_t.size(0)
        H, K = self.heads, self.d_h
        q = self.q_proj(x_t).view(B, H, K)
        k = self.k_proj(x_t).view(B, H, K)
        v = self.v_proj(x_t).view(B, H, K)
        g = torch.exp(F.logsigmoid(self.g_proj(x_t)).view(B, H, K).float())
        if S_prev is None:
            S_prev = torch.zeros(B, H, K, K, device=x_t.device, dtype=torch.float32)
        S_new = g.unsqueeze(-1) * S_prev + torch.einsum('bhk,bhj->bhkj', k.float(), v.float())
        o = torch.einsum('bhk,bhkj->bhj', q.float(), S_new)
        return self.out_proj(o.view(B, H * K)), S_new


# ============================================================
# Multi-Slot F-layer — 多槽门控递归 (fla HGRN scan)
# ============================================================
class MultiSlotFLayer(nn.Module):
    def __init__(self, dim_size, heads, n_slots=4):
        super().__init__()
        self.heads = heads
        self.d_head = dim_size // heads
        self.n_slots = n_slots
        self.d_sub = dim_size // n_slots
        assert dim_size % n_slots == 0
        assert dim_size % heads == 0

        self.combined = nn.Linear(dim_size, 4 * dim_size, bias=False)
        self.slot_proj = nn.Linear(dim_size, 4 * dim_size, bias=False)
        self.gen_gate = nn.Sequential(
            nn.Linear(heads * 5 * self.d_head, dim_size, bias=True),
            nn.GELU(),
            nn.Linear(dim_size, dim_size, bias=True),
        )
        self.gen_norm = nn.RMSNorm(dim_size)

    def forward(self, x, states=None):
        b, s, d = x.shape
        ns, ds = self.n_slots, self.d_sub

        combined = self.combined(x).view(b, s, 4, self.heads, -1)
        out, out1, out2, out3 = combined.unbind(2)
        out = out.permute(0, 3, 1, 2)
        out1 = out1.permute(0, 3, 1, 2)
        out2 = out2.permute(0, 3, 1, 2)
        out3 = out3.permute(0, 3, 1, 2)

        sg = self.slot_proj(x).reshape(b, s, 4, ns, ds).permute(0, 1, 3, 2, 4)
        af = torch.sigmoid(sg[..., 0, :])
        ff = torch.sigmoid(sg[..., 1, :])
        i_f = torch.sigmoid(sg[..., 2, :])
        cf = torch.tanh(sg[..., 3, :])
        A = af * ff + (1 - af)
        B_coeff = af * i_f * cf

        A_t = A.permute(0, 2, 1, 3).contiguous()
        B_t = B_coeff.permute(0, 2, 1, 3).contiguous()
        bns = b * ns
        g_t = torch.log(A_t.clamp(min=1e-8)).reshape(bns, s, ds)
        x_t = B_t.reshape(bns, s, ds)
        st_in = states.reshape(bns, ds) if states is not None else None
        H_flat, st_out = chunk_hgrn(x_t, g_t, initial_state=st_in, output_final_state=True)
        H = H_flat.reshape(b, ns, s, ds)
        new_states = st_out.reshape(b, ns, ds)

        H_cat = H.permute(0, 2, 1, 3).reshape(b, s, d)
        out4 = H_cat.reshape(b, s, self.heads, self.d_head).permute(0, 3, 1, 2)

        cat = torch.cat([out, out1, out2, out3, out4], dim=-1)
        cat_flat = cat.transpose(1, 2).reshape(b, s, -1)
        gen = self.gen_gate(cat_flat)
        gen = self.gen_norm(gen)
        return gen, new_states


class FeedForward(nn.Module):
    def __init__(self, hidden_size, expand=4):
        super().__init__()
        d_exp = hidden_size * expand
        self.gate = nn.Linear(hidden_size, d_exp, bias=False)
        self.up = nn.Linear(hidden_size, d_exp, bias=False)
        self.down = nn.Linear(d_exp, hidden_size, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.down(self.silu(self.gate(x)) * self.up(x))


class SSMLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, n_slots=4):
        super().__init__()
        self.ssm = MultiSlotFLayer(hidden_size, num_heads, n_slots)
        self.ffn = FeedForward(hidden_size)
        self.norm1 = nn.RMSNorm(hidden_size)
        self.norm2 = nn.RMSNorm(hidden_size)

    def forward(self, x, states=None):
        h = self.norm1(x)
        ssm_out, s = self.ssm(h, states)
        x = x + ssm_out
        x = x + self.ffn(self.norm2(x))
        return x, s


# ============================================================
# Linear SlowMemory — h_t = A(x)·h_{t-1} + B(x), 低频长程趋势 (fla HGRN)
# ============================================================
class LinearSlowMemory(nn.Module):
    def __init__(self, d_model, rank=None):
        super().__init__()
        d = d_model
        r = rank or max(d // 4, 32)
        self.W_down = nn.Linear(d, r, bias=False)
        self.W_A = nn.Linear(r, d, bias=True)
        self.W_B = nn.Linear(r, d, bias=True)
        self.W_gate = nn.Linear(r, 1, bias=True)
        nn.init.constant_(self.W_A.bias, 2.0)

    def forward(self, x_seq, h0):
        z = self.W_down(x_seq)
        A = torch.sigmoid(self.W_A(z))
        Bv = self.W_B(z)
        g = torch.log(A.clamp(min=1e-8))
        H, h_final = chunk_hgrn(Bv, g, initial_state=h0, output_final_state=True)
        alpha = torch.sigmoid(self.W_gate(z))
        Y = alpha * H + (1.0 - alpha) * x_seq
        return Y, h_final

    def step(self, x_t, h_prev):
        z = self.W_down(x_t)
        A = torch.sigmoid(self.W_A(z))
        Bv = self.W_B(z)
        h = A * h_prev + Bv
        alpha = torch.sigmoid(self.W_gate(z))
        y = alpha * h + (1.0 - alpha) * x_t
        return y, h


# ============================================================
# FRSMASH v3.7 DirectAdd
# ============================================================
class FRSMASHv37(nn.Module):
    """
    FRSMASH v3.7 = 多槽 SSM 骨干 + 线性 SlowMemory + GLA recall, DirectAdd 融合.
    fused = fusion_norm(x_ash + x_mem + x_emb) + x_recall
    """
    def __init__(self, voc_size, hidden_size, num_heads, num_layers, n_slots=4, max_pe=16384):
        super().__init__()
        self.D = hidden_size
        self.n_slots = n_slots
        self.num_layers = num_layers
        self.num_ssm = num_layers

        self.em = nn.Embedding(voc_size, hidden_size, padding_idx=0)
        _pe = torch.zeros(max_pe, hidden_size)
        _pos = torch.arange(max_pe).unsqueeze(1)
        _div = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000) / hidden_size))
        _pe[:, 0::2] = torch.sin(_pos * _div)
        _pe[:, 1::2] = torch.cos(_pos * _div)
        self.register_buffer('pe', _pe)

        self.layers = nn.ModuleList([
            SSMLayer(hidden_size, num_heads, n_slots) for _ in range(num_layers)
        ])
        self.final_norm = nn.RMSNorm(hidden_size)

        self.mem_input_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.slow_cell = LinearSlowMemory(hidden_size)
        self.mem_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mem_norm = nn.RMSNorm(hidden_size)

        self.recall = GlaRecall(hidden_size, heads=num_heads, d_h=64)
        self.recall_norm = nn.RMSNorm(hidden_size)

        self.fusion_norm = nn.RMSNorm(hidden_size)
        self.head = nn.Linear(hidden_size, voc_size, bias=False)

    def forward(self, x, states=None, h_slow=None, recall_state=None,
                return_state=False, pos_offset=0):
        B, T = x.shape
        D = self.D
        dt = self.head.weight.dtype
        x_emb = self.em(x).to(dt) + self.pe[pos_offset:pos_offset + T].to(dt)

        if states is None:
            states = [None] * self.num_ssm
        if h_slow is None:
            h_slow = torch.zeros(B, D, device=x.device, dtype=dt)

        h = x_emb
        new_states = [] if return_state else None
        for i, layer in enumerate(self.layers):
            s_in = states[i] if return_state else None
            h, s = layer(h, s_in)
            if return_state:
                new_states.append(s)
        x_ash = self.final_norm(h)

        inp_seq = self.mem_input_proj(x_emb)
        H_slow, h_slow = self.slow_cell(inp_seq, h_slow)
        x_mem = self.mem_norm(self.mem_proj(H_slow))

        if return_state or recall_state is not None:
            recall_out, recall_state = self.recall(x_emb, initial_state=recall_state, return_state=True)
        else:
            recall_out = self.recall(x_emb)
        x_recall = self.recall_norm(recall_out)

        fused = self.fusion_norm(x_ash + x_mem + x_emb) + x_recall
        logits = self.head(fused)
        if return_state:
            return logits, new_states, h_slow, recall_state
        return logits

    @torch.no_grad()
    def generate_step(self, token_id, states, h_slow, recall_state=None, pos=0):
        dt = self.head.weight.dtype
        x = self.em(token_id).to(dt) + self.pe[pos:pos + 1].to(dt)
        h = x
        new_states = []
        for i, layer in enumerate(self.layers):
            h, s = layer(h, states[i])
            new_states.append(s)
        x_ash = self.final_norm(h[:, 0])
        inp = self.mem_input_proj(x[:, 0])
        y_slow, h_slow = self.slow_cell.step(inp, h_slow)
        x_mem = self.mem_proj(y_slow)
        o_recall, recall_state = self.recall.step(x[:, 0].float(), recall_state)
        x_recall = self.recall_norm(o_recall.to(dt))
        fused = self.fusion_norm(x_ash + x_mem + x[:, 0]) + x_recall
        logits = self.head(fused)
        return logits, new_states, h_slow, recall_state, pos + 1


if __name__ == '__main__':
    import sys
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    VS = 23005
    H, L, HD = 448, 7, 8
    m = FRSMASHv37(VS, H, HD, L).to(dev)
    n = sum(p.numel() for p in m.parameters())
    print(f'FRSMASH v3.7 DirectAdd: H={H} L={L} heads={HD} => {n:,} params ({n/1e6:.1f}M)')
    x = torch.randint(0, VS, (4, 256), device=dev)
    o = m(x)
    print('forward:', o.shape, o.device)
    F.cross_entropy(o.reshape(-1, VS), torch.randint(0, VS, (4 * 256,), device=dev)).backward()
    print('backward: OK')
