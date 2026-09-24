#!/usr/bin/env python3
"""CEDLR-Hybrid2 30M 缩放基准: 实测 ms/step -> 全量 PT+SFT 时间折算."""
import sys, os, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lowrank import RubikLowRankFast
from run_lm import SWALayer
from scan import affine_scan, chunked_affine_scan
from torch.utils.checkpoint import checkpoint as ckpt

DEV = "cuda"
V, D, H, LAY, B = 23005, 320, 10, 12, 32
P, Q = 192, 64
FFN = 896
RUBIK_WIN = 64


class SwiGLU(nn.Module):
    def __init__(self, d, h=FFN):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.gate = nn.Linear(d, h)
        self.up = nn.Linear(d, h)
        self.down = nn.Linear(h, d)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)

    def forward(self, x):
        h = self.ln(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class RubikAttn30(nn.Module):
    """低秩非交换注意力 (训练用逐 token 递推, fp32 岛)."""
    def __init__(self, d, H=H, r=2):
        super().__init__()
        self.H, self.dh, self.r = H, d // H, r
        self.ln = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.uv = nn.Linear(d, H * 2 * r * self.dh)
        self.out = nn.Linear(d, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.lam = nn.Parameter(torch.full((H, self.dh, 1), -1.0))
        nn.init.normal_(self.uv.weight, 0.0, 0.01)
        nn.init.zeros_(self.uv.bias)

    def forward(self, x, state=None):
        B, L, _ = x.shape
        H, dh, r = self.H, self.dh, self.r
        with torch.autocast("cuda", enabled=False):
            x = self.ln(x).float()
            q = self.q(x).float().view(B, L, H, dh)
            k = self.k(x).float().view(B, L, H, dh)
            v = self.v(x).float().view(B, L, H, dh)
            uv = self.uv(x).float().view(B, L, H, 2 * r, dh)
            u = uv[:, :, :, :r, :]
            w = uv[:, :, :, r:, :]
            P = torch.cat([u, -w], dim=-2).transpose(-2, -1)
            Q = torch.cat([w, u], dim=-2).transpose(-2, -1)
            X = torch.einsum("nlhpi,nlhpj->nlhij", Q, P)
            eye2r = torch.eye(2 * r, device=x.device).view(1, 1, 2 * r, 2 * r)
            gX = torch.linalg.solve(X + 1e-4 * eye2r,
                                    torch.matrix_exp(X) - eye2r)
            lam = torch.sigmoid(self.lam).unsqueeze(0).float()
            # A = Λ + (ΛP)(gX Q^T)  — bf16 扫描数组 (流量减半), bmm 单趟
            lamP = lam * P                                          # (B,L,H,dh,2r)
            gQ = gX @ Q.transpose(-2, -1)                                       # (B,L,H,2r,dh)
            low = lamP @ gQ                                                   # (B,L,H,dh,dh)
            lamD = lam * torch.eye(
                dh, device=x.device).view(1, 1, dh, dh)                  # (1,H,dh,dh)
            A = lamD + low
            W = k.unsqueeze(-1) * v.unsqueeze(-2)
            if self.training:
                A_star, B_star = ckpt(affine_scan, A, W, use_reentrant=False)
            else:
                A_star, B_star = affine_scan(A, W)
            B_star = B_star.float()
            o = torch.einsum("nlhp,nlhpq->nlhq", q, B_star) / (self.dh ** 0.5)
            out = o.reshape(B, L, H * dh)
            if torch.is_autocast_enabled():
                out = out.to(torch.bfloat16)
        return out, None


class SWA30(nn.Module):
    def __init__(self, d, H=H, win=RUBIK_WIN):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.H, self.dh, self.win = H, d // H, win
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        B, L, d = x.shape
        h = self.ln(x)
        qkv = self.qkv(h).view(B, L, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = q @ k.transpose(-1, -2) / self.dh ** 0.5
        i = torch.arange(L, device=x.device)
        att = att.masked_fill((i[:, None] < i[None, :])[None, None], float("-inf"))
        o = torch.softmax(att, -1) @ v
        return x + self.out(o.transpose(1, 2).reshape(B, L, d))


class Block30(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == "rubik":
            self.attn = RubikAttn30(D, H)
        else:
            self.attn = SWA30(D, H, win=RUBIK_WIN)
        self.ffn = SwiGLU(D)

    def forward(self, x, state=None):
        if self.kind == "rubik":
            a, state = self.attn(x, state)
            x = x + a
        else:
            x = self.attn(x)
        x = self.ffn(x)
        return x, state


class CED30(nn.Module):
    """CEDLR-Hybrid2 30M: enc 6 层 (前缀 P) + dec 6 层 (后缀 Q), KV 从 H_enc 投影."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.kinds = ["rubik", "rubik", "swa"] * 2        # enc 6
        self.enc = nn.ModuleList([Block30(k) for k in self.kinds])
        self.dec_kinds = ["rubik", "rubik", "swa"] * 2    # dec 6
        self.dec = nn.ModuleList([Block30(k) for k in self.dec_kinds])
        self.dec_kv = nn.ModuleList([nn.Linear(D, 2 * D) for _ in range(LAY)])
        self.dec_ln = nn.ModuleList([nn.LayerNorm(D) for _ in range(LAY)])
        self.dec_q = nn.ModuleList([nn.Linear(D, D) for _ in range(LAY)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, V)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):     # x: (B, P+Q)
        B = x.shape[0]
        h = self.emb(x)
        he = h[:, :P]
        st = None
        for layer in self.enc:
            he, st = layer(he, st)
        dec = h[:, P:]
        for l in range(6):                                # 解码器 6 层
            dec, _ = self.dec[l](dec, None)               # 局部 (swa/rubik 自注意力因果)
            C = self.dec_kv[l](he)                        # (B,P,2D)
            hh = self.dec_ln[l](dec)
            qq = self.dec_q[l](hh).view(B, Q, H, D // H).transpose(1, 2)
            k, v = C.view(B, P, H, 2, D // H).permute(0, 2, 3, 1, 4)[..., 0, :, :], \
                   C.view(B, P, H, 2, D // H).permute(0, 2, 3, 1, 4)[..., 1, :, :]
            att = qq @ k.transpose(-1, -2) / (D // H) ** 0.5
            o = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(B, Q, D)
            dec = dec + o
        return self.head(self.ln_f(dec))


def main():
    import sys
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    torch.manual_seed(0)
    m = CED30().to(DEV)
    n = sum(p.numel() for p in m.parameters())
    print("CEDLR-Hybrid2-30M 参数: %.2fM  B=%d" % (n / 1e6, B), flush=True)
    x = torch.randint(0, V, (B, P + Q), device=DEV)
    y_dec = x[:, P + 1:]                                  # 解码段 next-token 目标
    opt = torch.optim.AdamW(m.parameters(), lr=6e-4)
    # 预热
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(x)
            loss = F.cross_entropy(lo[:, :-1].reshape(-1, V), y_dec.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    torch.cuda.synchronize()
    t0 = time.time()
    iters = 10
    for _ in range(iters):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(x)
            loss = F.cross_entropy(lo[:, :-1].reshape(-1, V), y_dec.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / iters * 1000
    mem = torch.cuda.max_memory_allocated() / 2**30
    print("实测: %.0f ms/step (B=%d, L=%d), 峰值 %.1f GB" % (ms, B, P + Q, mem), flush=True)
    pt_steps = (1265225 + B - 1) // B                    # PT 1 epoch
    sft_steps = (905000 + B - 1) // B                    # SFT 1 epoch (905k 样本)
    print("折算 (全量 1 epoch 协议):", flush=True)
    print("  PT  %d 步  ≈ %.1f h" % (pt_steps, ms * pt_steps / 3.6e6))
    print("  SFT %d 步  ≈ %.1f h" % (sft_steps, ms * sft_steps / 3.6e6))
    print("  合计      ≈ %.1f h (单卡, 当前争用水平)" % (ms * (pt_steps + sft_steps) / 3.6e6), flush=True)


if __name__ == "__main__":
    main()
