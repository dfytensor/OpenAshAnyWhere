"""OpenASH + FactRegister (QueryFirst-M1): 放大写寄存器集成进真 OpenASH (768x12).

机制:
  - MARK_OPEN 开启写入模式, MARK_CLOSE 时把前一个 token (答案值) 的键 ×(1+amp)
    放大写入 M_reg (硬 max + STE)
  - 每步向输入注入幅度门控的寄存器信号: x_emb += inject(σ(γ(M-θ)) ⊙ M)
  - decoder 层不变 (批量前向), 模型学会在提问处从寄存器读出答案
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MARK_OPEN, MARK_CLOSE, QUESTION = 23005, 23006, 23007
BETA = 8.0


class FactRegister(nn.Module):
    def __init__(self, d=768, amp=2.0):
        super().__init__()
        self.d = d
        self.amp = amp
        self.Wk = nn.Linear(d, d)
        self.inject = nn.Linear(d, d)
        self.gamma = nn.Parameter(torch.tensor(0.2))
        self.theta = nn.Parameter(torch.tensor(0.8))
        nn.init.normal_(self.Wk.weight, 0.0, 0.02)
        nn.init.normal_(self.inject.weight, 0.0, 0.02)

    def forward(self, x_emb, tok_ids):
        """x_emb: [B, T, d]; tok_ids: [B, T]. 返回注入后的 x_emb 与寄存器状态 M."""
        B, T, d = x_emb.shape
        dev = x_emb.device
        M = torch.zeros(B, d, device=dev)
        mode = torch.zeros(B, dtype=torch.bool, device=dev)
        prev = torch.zeros(B, d, device=dev)
        for t in range(T):
            tok = tok_ids[:, t]
            mode = (mode | (tok == MARK_OPEN)) & (tok != MARK_CLOSE)
            close = tok == MARK_CLOSE
            if close.any():
                k = self.Wk(prev[close]) * (1.0 + self.amp)
                m0 = M[close]
                s = torch.sigmoid(BETA * (k - m0))
                M_soft = m0 + s * (k - m0)
                M_hard = torch.maximum(m0, k)
                M[close] = M_hard + (M_soft - M_hard).detach()
            prev = x_emb[:, t]
        gate = torch.sigmoid(self.gamma * (M - self.theta))
        inj = self.inject(gate * M).unsqueeze(1)          # [B, 1, d] 广播到所有位置
        return x_emb + inj, M


class OpenASHReg(nn.Module):
    def __init__(self, base_model, voc_extra=3):
        super().__init__()
        self.base = base_model
        self.d = base_model.head_score.weight.shape[1]
        self.register = FactRegister(self.d)
        em = base_model.em.weight.data
        add = torch.zeros(voc_extra, self.d, dtype=em.dtype, device=em.device)
        nn.init.normal_(add, 0.0, 0.02)
        base_model.em.weight.data = torch.cat([em, add], 0)
        hs = base_model.head_score.weight.data
        addh = torch.zeros(voc_extra, self.d, dtype=hs.dtype, device=hs.device)
        nn.init.normal_(addh, 0.0, 0.02)
        base_model.head_score.weight.data = torch.cat([hs, addh], 0)
        self.voc = base_model.head_score.weight.shape[0]

    def forward(self, x, state=None):
        B, T = x.shape
        x_emb = self.base.em(x)
        x_emb, M = self.register(x_emb, x)
        if state is None:
            state = [None] * len(self.base.decoder_layers)
        else:
            state = list(state)
        i = 0
        h = x_emb
        for ii, layer in enumerate(self.base.decoder_layers):
            x1, state[i] = layer(h, state[i])
            h = x1 + h
            i += 1
        out = self.base.head_score(h)
        state = state + [M]
        return out, state


def build_reg_model(ckpt_path, stable=False, R=10.0):
    """加载 768x12 OpenASH checkpoint, 套上寄存器 (stable=True 用范数cap基座)."""
    if stable:
        from openash_stable import OpenASHStable
        base = OpenASHStable(voc_size=23005, hidden_size=768, num_heads=8, num_layers=12, R=R)
    else:
        from open_ash import OpenASH
        base = OpenASH(voc_size=23005, hidden_size=768, num_heads=8, num_layers=12)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    base.load_state_dict(sd, strict=False)
    return OpenASHReg(base)
