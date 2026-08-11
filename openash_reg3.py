"""OpenASH + LearnedGateRegister (M3): 无标记事实, 学习式写入门 + 内容寻址读出.

机制:
  - g_t = sigmoid(Wg(x_t)+b) 写入门 (STE: 前向硬, 反向软)
  - 门开 span 累积: span_sum/cnt = 键均值 (查询键), v = 软最后token键 (值, 放大写)
  - 门落时提交到下一槽; 提问处 cos 内容寻址读
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

QUESTION = 23007
BETA = 8.0


class LearnedGateRegister(nn.Module):
    def __init__(self, d=768, n_slots=8, amp=2.0):
        super().__init__()
        self.d = d
        self.n_slots = n_slots
        self.amp = amp
        self.Wk = nn.Linear(d, d)
        self.Wg = nn.Linear(d, 1)
        self.inject = nn.Linear(d, d)
        self.gamma = nn.Parameter(torch.tensor(5.0))
        self.theta = nn.Parameter(torch.tensor(0.3))
        nn.init.normal_(self.Wk.weight, 0.0, 0.02)
        nn.init.normal_(self.Wg.weight, 0.0, 0.02)
        nn.init.normal_(self.inject.weight, 0.0, 0.02)

    def forward(self, x_emb, tok_ids):
        B, T, d = x_emb.shape
        dev = x_emb.device
        M = torch.zeros(B, self.n_slots, d, device=dev)
        K = torch.zeros(B, self.n_slots, d, device=dev)
        ptr = torch.zeros(B, dtype=torch.long, device=dev)
        g_raw = self.Wg(x_emb).squeeze(-1)                    # [B,T]
        g_hard = (g_raw > 0).float()
        g_eff = g_hard + (torch.sigmoid(g_raw) - g_hard).detach()   # STE
        span_sum = torch.zeros(B, d, device=dev)
        span_cnt = torch.zeros(B, device=dev)
        v = torch.zeros(B, d, device=dev)
        prev_committed = torch.zeros(B, dtype=torch.bool, device=dev)
        for t in range(T):
            g = g_eff[:, t]
            k = self.Wk(x_emb[:, t])
            span_sum = span_sum + g.unsqueeze(1) * k
            span_cnt = span_cnt + g
            v = g.unsqueeze(1) * k + (1 - g).unsqueeze(1) * v
            committed = g > 0.5
            end = prev_committed & ~committed
            if end.any():
                qk = span_sum / span_cnt.clamp(min=1).unsqueeze(1)
                bi_idx = end.nonzero().squeeze(-1)
                # 只写空槽: 找每个样本的第一个空槽; 全满则丢弃 (针永不覆盖)
                occ = (K.norm(dim=-1) > 1e-6)                    # [B, K]
                free = ~occ
                write_mask = free.any(dim=-1)                     # [B]
                bi_w = bi_idx[write_mask[bi_idx]]
                if bi_w.numel() > 0:
                    pp_idx = torch.zeros_like(bi_w)
                    for oi, bbi in enumerate(bi_w.tolist()):
                        pp_idx[oi] = free[bbi].nonzero()[0]
                    kk = v[bi_w] * (1.0 + self.amp)
                    m0 = M[bi_w, pp_idx]
                    s = torch.sigmoid(BETA * (kk - m0))
                    M_soft = m0 + s * (kk - m0)
                    M_hard = torch.maximum(m0, kk)
                    new_val = M_hard + (M_soft - M_hard).detach()
                    M = torch.index_put(M, (bi_w, pp_idx), new_val)
                    K = torch.index_put(K, (bi_w, pp_idx), qk[bi_w])
                span_sum = torch.zeros_like(span_sum)
                span_cnt = torch.zeros_like(span_cnt)
                v = torch.zeros_like(v)
            prev_committed = committed
        q_acc = torch.zeros(B, d, device=dev)
        q_cnt = torch.zeros(B, device=dev)
        for t in range(T):
            is_q = tok_ids[:, t] == QUESTION
            q_acc = q_acc + torch.where(is_q.unsqueeze(1), self.Wk(x_emb[:, t]), torch.zeros_like(x_emb[:, t]))
            q_cnt = q_cnt + is_q.float()
        q = q_acc / q_cnt.clamp(min=1).unsqueeze(1)
        qn = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        Kn = F.normalize(K, dim=-1)
        cos = torch.einsum("bd,bkd->bk", qn, Kn)
        gate = torch.sigmoid(self.gamma * (cos - self.theta))
        sig = torch.einsum("bk,bkd->bd", gate, M)
        inj = self.inject(sig).unsqueeze(1)
        return x_emb + inj, M, K


class OpenASHReg3(nn.Module):
    def __init__(self, base_model, n_slots=8, voc_extra=3):
        super().__init__()
        self.base = base_model
        self.d = base_model.head_score.weight.shape[1]
        self.register = LearnedGateRegister(self.d, n_slots)
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


def build_reg3_model(ckpt_path, stable=True, R=10.0, n_slots=8):
    if stable:
        from openash_stable import OpenASHStable
        base = OpenASHStable(voc_size=23005, hidden_size=768, num_heads=8, num_layers=12, R=R)
    else:
        from open_ash import OpenASH
        base = OpenASH(voc_size=23005, hidden_size=768, num_heads=8, num_layers=12)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    base.load_state_dict(sd, strict=False)
    return OpenASHReg3(base, n_slots=n_slots)
