# 低秩非交换门控层 (H4 消融): A = u v^T - v u^T = P Q^T, P=[u,-v], Q=[v,u]
# G S = S + P g(Q^T P) (Q^T S),  g(X) = sum_{m>=0} X^m/(m+1)!  (12 项 Taylor)
# 每 token O(4r dh^2), 无需构造 G 或 matrix_exp
import torch
import torch.nn as nn


class RubikLowRankLayer(nn.Module):
    def __init__(self, d, H=4, r=2, decay=True):
        super().__init__()
        self.H, self.dh, self.r = H, d // H, r
        self.decay = decay
        self.ln_in = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.uv = nn.Linear(d, H * 2 * r * self.dh)
        nn.init.normal_(self.uv.weight, 0.0, 0.01)
        nn.init.zeros_(self.uv.bias)
        if decay:
            self.lam = nn.Parameter(torch.full((H, self.dh, 1), -1.0))

    def forward(self, x, state=None):
        B, L, _ = x.shape
        H, dh, r = self.H, self.dh, self.r
        with torch.autocast("cuda", enabled=False):
            x = self.ln_in(x).float()
            q = self.q(x).view(B, L, H, dh)
            k = self.k(x).view(B, L, H, dh)
            v = self.v(x).view(B, L, H, dh)
            uv = self.uv(x).view(B, L, H, 2 * r, dh)
            u = uv[:, :, :, :r, :]                              # (B,L,H,r,dh)
            w = uv[:, :, :, r:, :]
            P = torch.cat([u, -w], dim=-2).transpose(-2, -1)    # (B,L,H,dh,2r)
            Q = torch.cat([w, u], dim=-2).transpose(-2, -1)
            if state is None:
                state = x.new_zeros(B, H, dh, dh)
            else:
                state = state.float()
            lam = torch.sigmoid(self.lam).unsqueeze(0).float() if self.decay else None
            outs = []
            for t in range(L):
                Pt, Qt = P[:, t], Q[:, t]
                X = torch.einsum("nhpi,nhpj->nhij", Qt, Pt)     # (B,H,2r,2r)
                Gx = torch.eye(2 * r, device=x.device).view(1, 1, 2 * r, 2 * r).expand(B, H, -1, -1)
                term = Gx.clone()
                for m in range(1, 13):
                    term = term @ X / (m + 1)
                    Gx = Gx + term
                write = k[:, t].view(B, H, dh, 1) * v[:, t].view(B, H, 1, dh)
                z = torch.einsum("nhpi,nhpc->nhic", Qt, state)  # Q^T S -> (B,H,2r,dh)
                z = Gx @ z
                state = state + P[:, t] @ z                     # S + P g(X) Q^T S
                if self.decay:
                    state = lam * state
                state = state + write
                o = (q[:, t].view(B, H, 1, dh) @ state).squeeze(-2)
                outs.append(o.reshape(B, H * dh))
            out = torch.stack(outs, 1)
            if torch.is_autocast_enabled():
                out = out.to(torch.bfloat16)
        return out, state


def lowrank_ref_check(B=2, H=2, dh=16, r=2, seed=0):
    """正确性: 低秩 G·S vs 全秩 matrix_exp(skew(uv^T-vu^T))·S."""
    torch.manual_seed(seed)
    u = torch.randn(B, H, dh, r) * 0.3
    w = torch.randn(B, H, dh, r) * 0.3
    P = torch.cat([u, -w], -1)
    Q = torch.cat([w, u], -1)
    S = torch.randn(B, H, dh, dh)
    # 低秩
    X = torch.einsum("bhpi,bhpj->bhij", Q, P)
    Gx = torch.eye(2 * r).expand(B, H, -1, -1)
    term = Gx.clone()
    for m in range(1, 13):
        term = term @ X / (m + 1)
        Gx = Gx + term
    z = torch.einsum("nhpa,nhpc->nhac", Q, S)
    low = S + P @ (Gx @ z)
    # 全秩参考
    A = P @ Q.transpose(-2, -1)
    G = torch.matrix_exp(A)
    ref = G @ S
    err = (low - ref).abs().max().item()
    # 正交性: G^T G = I
    orth = (G.transpose(-2, -1) @ G - torch.eye(dh)).abs().max().item()
    return err, orth


if __name__ == "__main__":
    err, orth = lowrank_ref_check()
    print("低秩 exp 恒等式误差: %.2e  |  全秩 G 正交性误差: %.2e" % (err, orth))
    assert err < 1e-3 and orth < 1e-4, "FAIL"
    print("PASS")
