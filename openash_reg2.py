"""OpenASH + MultiFactRegister (M2): 多槽寄存器 + 内容寻址读出.

机制:
  - 每个 [MARK_OPEN 事实 MARK_CLOSE] 写入一个槽:
      slot.M  = 值 token 键 ×(1+amp)  (放大写, 硬max+STE)
      slot.K  = 事实文本 token 键的均值 (查询键)
  - 提问处: q = 问题 token 键的均值; gate_k = σ(γ·(cos(q, K_k) − θ))
    注入 Σ gate_k·M_k, 模型输出匹配槽的值
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MARK_OPEN, MARK_CLOSE, QUESTION = 23005, 23006, 23007
BETA = 8.0


class MultiFactRegister(nn.Module):
    def __init__(self, d=768, n_slots=8, amp=2.0):
        super().__init__()
        self.d = d
        self.n_slots = n_slots
        self.amp = amp
        self.Wk = nn.Linear(d, d)
        self.inject = nn.Linear(d, d)
        self.gamma = nn.Parameter(torch.tensor(5.0))
        self.theta = nn.Parameter(torch.tensor(0.3))
        nn.init.normal_(self.Wk.weight, 0.0, 0.02)
        nn.init.normal_(self.inject.weight, 0.0, 0.02)

    def forward(self, x_emb, tok_ids):
        """x_emb: [B,T,d]; tok_ids: [B,T]. 逐 token 扫描:
        MARK_OPEN 开窗, 窗内累积事实键均值; MARK_CLOSE 写入下一槽."""
        B, T, d = x_emb.shape
        dev = x_emb.device
        M = torch.zeros(B, self.n_slots, d, device=dev)
        K = torch.zeros(B, self.n_slots, d, device=dev)
        # 状态: 槽指针, 窗口
        ptr = torch.zeros(B, dtype=torch.long, device=dev)
        in_win = torch.zeros(B, dtype=torch.bool, device=dev)
        win_sum = torch.zeros(B, d, device=dev)
        win_cnt = torch.zeros(B, device=dev)
        prev = torch.zeros(B, d, device=dev)
        q_acc = torch.zeros(B, d, device=dev)
        q_cnt = torch.zeros(B, device=dev)
        for t in range(T):
            tok = tok_ids[:, t]
            xe = x_emb[:, t]
            # 窗口开/关
            in_win = (in_win | (tok == MARK_OPEN)) & (tok != MARK_CLOSE)
            is_close = tok == MARK_CLOSE
            # 窗口内累积 (事实文本键均值, 不含值)
            win_sum = win_sum + torch.where(in_win.unsqueeze(1), self.Wk(xe), torch.zeros_like(xe))
            win_cnt = win_cnt + in_win.float()
            # 提问累积 (问题 token 键均值)
            q_acc = q_acc + torch.where((tok == QUESTION).unsqueeze(1), self.Wk(xe), torch.zeros_like(xe))
            q_cnt = q_cnt + (tok == QUESTION).float()
            # MARK_CLOSE: 写槽
            if is_close.any():
                k = self.Wk(prev[is_close]) * (1.0 + self.amp)      # 值键放大
                p = ptr[is_close]
                qk = win_sum[is_close] / win_cnt[is_close].clamp(min=1).unsqueeze(1)
                for bi, (pp, qq) in enumerate(zip(p.tolist(), qk)):
                    m0 = M[is_close][bi, pp]
                    s = torch.sigmoid(BETA * (k[bi] - m0))
                    M_soft = m0 + s * (k[bi] - m0)
                    M_hard = torch.maximum(m0, k[bi])
                    M[is_close][bi, pp] = M_hard + (M_soft - M_hard).detach()
                    K[is_close][bi, pp] = qq
                ptr = (ptr + 1) % self.n_slots
                win_sum = torch.zeros_like(win_sum)
                win_cnt = torch.zeros_like(win_cnt)
            prev = xe
        # 内容寻址读出: q 与各槽键的余弦
        q = q_acc / q_cnt.clamp(min=1).unsqueeze(1)
        qn = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        Kn = F.normalize(K, dim=-1)
        cos = torch.einsum("bd,bkd->bk", qn, Kn)
        gate = torch.sigmoid(self.gamma * (cos - self.theta))        # [B, K]
        sig = torch.einsum("bk,bkd->bd", gate, M)                     # 加权读
        inj = self.inject(sig).unsqueeze(1)
        return x_emb + inj, M, K


class OpenASHReg2(nn.Module):
    def __init__(self, base_model, n_slots=8, voc_extra=3):
        super().__init__()
        self.base = base_model
        self.d = base_model.head_score.weight.shape[1]
        self.register = MultiFactRegister(self.d, n_slots)
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
        x_emb = self.base.em(x)
        x_emb, M, K = self.register(x_emb, x)
        if state is None:
            state = [None] * len(self.base.decoder_layers)
        else:
            state = list(state)
        h = x_emb
        for ii, layer in enumerate(self.base.decoder_layers):
            x1, state[ii] = layer(h, state[ii])
            h = x1 + h
        out = self.base.head_score(h)
        state = state + [M]
        return out, state


def build_reg2_model(ckpt_path, stable=False, R=10.0, n_slots=8):
    if stable:
        from openash_stable import OpenASHStable
        base = OpenASHStable(voc_size=23005, hidden_size=768, num_heads=8, num_layers=12, R=R)
    else:
        from open_ash import OpenASH
        base = OpenASH(voc_size=23005, hidden_size=768, num_heads=8, num_layers=12)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    base.load_state_dict(sd, strict=False)
    return OpenASHReg2(base, n_slots=n_slots)
