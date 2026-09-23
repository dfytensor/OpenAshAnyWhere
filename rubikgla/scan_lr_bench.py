#!/usr/bin/env python3
"""无循环 scan 版低秩注意力 vs 逐 token 循环版: 精度 + 速度 (30M 配置实测)."""
import sys, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan import affine_scan


class RubikLRScanAttn(nn.Module):
    """低秩非交换注意力, 全批量无循环:
    G_t = I + P g(X) Q^T (全量批量算) -> affine_scan -> S = B* (S0=0) -> o = q S."""
    def __init__(self, d, H=10, r=2):
        super().__init__()
        self.H, self.dh, self.r = H, d // H, r
        self.ln = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.uv = nn.Linear(d, H * 2 * r * self.dh)
        self.out = nn.Linear(d, d)
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
            P = torch.cat([u, -w], dim=-2).transpose(-2, -1)    # (B,L,H,dh,2r)
            Q = torch.cat([w, u], dim=-2).transpose(-2, -1)
            X = torch.einsum("nlhpi,nlhpj->nlhij", Q, P)
            eye = torch.eye(2 * r, device=x.device).view(1, 1, 1, 2 * r, 2 * r)
            E = torch.matrix_exp(X)
            gX = torch.linalg.solve(X + 1e-4 * eye, E - eye)
            # G = I + P gX Q^T  (全量批量, 无循环)
            G = torch.eye(dh, device=x.device).view(1, 1, 1, dh, dh) \
                + P @ (gX @ Q.transpose(-2, -1))
            lam = torch.sigmoid(self.lam).unsqueeze(0).float()
            A = lam * G
            Bw = k.unsqueeze(-1) * v.unsqueeze(-2)              # (B,L,H,dh,dh) 外积
            A_star, B_star = affine_scan(A, Bw)
            # S_t = A*_t S_0 + B*_t, S_0 = 0 => S_t = B*_t
            o = torch.einsum("nlhp,nlhpq->nlhq", q, B_star)
            out = o.reshape(B, L, H * dh)
            if torch.is_autocast_enabled():
                out = out.to(torch.bfloat16)
        return out, None


def loop_ref(layer, x):
    """逐 token 循环参考 (与 lowrank.RubikLowRankFast 相同语义)."""
    B, L, _ = x.shape
    H, dh, r = layer.H, layer.dh, layer.r
    with torch.autocast("cuda", enabled=False):
        x = layer.ln(x).float()
        q = layer.q(x).view(B, L, H, dh)
        k = layer.k(x).view(B, L, H, dh)
        v = layer.v(x).view(B, L, H, dh)
        uv = layer.uv(x).view(B, L, H, 2 * r, dh)
        u = uv[:, :, :, :r, :]
        w = uv[:, :, :, r:, :]
        P = torch.cat([u, -w], dim=-2).transpose(-2, -1)
        Q = torch.cat([w, u], dim=-2).transpose(-2, -1)
        state = x.new_zeros(B, H, dh, dh)
        lam = torch.sigmoid(layer.lam).unsqueeze(0).float()
        eye2r = torch.eye(2 * r, device=x.device).view(1, 1, 2 * r, 2 * r)
        outs = []
        for t in range(L):
            X = torch.einsum("nhpi,nhpj->nhij", Q[:, t], P[:, t])
            gX = torch.linalg.solve(X + 1e-4 * eye2r,
                                    torch.matrix_exp(X) - eye2r)
            write = k[:, t].view(B, H, dh, 1) * v[:, t].view(B, H, 1, dh)
            z = torch.einsum("nhpa,nhpc->nhac", Q[:, t], state)
            z = gX @ z
            state = lam * (state + P[:, t] @ z) + write
            o = (q[:, t].view(B, H, 1, dh) @ state).squeeze(-2)
            outs.append(o.reshape(B, H * dh))
        return torch.stack(outs, 1)


def main():
    d, H, r = 320, 10, 2
    torch.manual_seed(0)
    sl = RubikLRScanAttn(d, H, r).cuda()
    x = torch.randn(8, 256, d, device="cuda")
    with torch.no_grad():
        o_scan, _ = sl(x)
    o_loop = loop_ref(sl, x)
    diff = (o_scan - o_loop).abs().max().item()
    print("scan vs loop 精度: max diff = %.2e" % diff)

    for tag, use_scan in [("loop", False), ("scan", True)]:
        torch.manual_seed(0)
        lay = RubikLRScanAttn(d, H, r).cuda()
        x = torch.randn(48, 256, d, device="cuda")
        for _ in range(2):
            o = lay(x) if use_scan else loop_ref(lay, x)
            if isinstance(o, tuple):
                o = o[0]
            o.float().pow(2).mean().backward()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(3):
            o = lay(x) if use_scan else loop_ref(lay, x)
            if isinstance(o, tuple):
                o = o[0]
            o.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        print("%s: %.0f ms/层 fwd+bwd (B=48 L=256 dh=32)" %
              (tag, (time.time() - t0) / 3 * 1000))


if __name__ == "__main__":
    import time
    main()
