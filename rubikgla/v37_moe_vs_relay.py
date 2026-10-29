#!/usr/bin/env python3
"""判决性实验: 序列级接力 (多模型+state 接力) vs token 级 MoE — 域分段语料正面对抗.

数据: 3 域 Markov (同词表 512, 不同转移结构 = 真域差异), 训练块 1024/条.
R 序列级接力: 共享基座预训练 → 3 域专家 finetune; 推理逐段路由 + s0 接力 (跨模型交接).
M token级MoE: 共享基座 + MoE FFN (3 专家, oracle 域路由 = 给 MoE 最好条件).
终评: 混拼 held-out (6×1024 交替) — 总/各域后缀 NLL + 跨域切换后首 64 token NLL
      (relay_min 教训的直接检验).
"""
import sys, os, time, random
os.environ.setdefault("DELTA_STAGES", "1")
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_triton import delta_scan_triton, delta_scan_states

DEV = "cuda"
V, D, H, KV_H, SLOTS = 512, 128, 8, 4, 2
DH = D // H
G = KV_H * SLOTS
SEG = 1024
N_DOM = 3
STEPS_BASE = 400
STEPS_FT = 250
STEPS_MOE = 650
LR = 3e-4


def log(m, **kw):
    print(m, flush=True)


def count_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def ce(logits, x):
    return F.cross_entropy(logits[:, :-1].reshape(-1, V).float(),
                           x[:, 1:].reshape(-1), ignore_index=0)


# ---------------- 三域 Markov 数据 ----------------
def make_chains(seed=0):
    g = torch.Generator().manual_seed(seed)
    chains = []
    for _ in range(N_DOM):
        T = torch.zeros(V, V)
        for t in range(V):
            nxt = torch.randperm(V, generator=g)[:8]
            T[t, nxt] = 1.0 / 8
        chains.append(T)
    return chains


def walk(chain, ln):
    x = torch.zeros(1, ln, dtype=torch.long)
    x[0, 0] = int(torch.randint(0, V, (1,)).item())
    for t in range(1, ln):
        x[0, t] = int(torch.multinomial(chain[x[0, t - 1]], 1).item())
    return x


def gen_blocks(chains, dom, n, ln):
    return [walk(chains[dom], ln) for _ in range(n)]


# ---------------- 模型 ----------------
class MARMS(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        ms = x.float().pow(2).mean(-1, keepdim=True) + 1e-6
        return (x.float() * torch.rsqrt(ms) * self.gamma).to(x.dtype)


class DeltaAttn(nn.Module):
    def __init__(self, moe_experts=0):
        super().__init__()
        self.ln = MARMS(D)
        self.wq = nn.Linear(D, D, bias=False)
        self.wk = nn.Linear(D, G * DH, bias=False)
        self.wv = nn.Linear(D, G * DH, bias=False)
        self.out = nn.Linear(D, D, bias=False)
        nn.init.normal_(self.wq.weight, std=0.02)
        nn.init.normal_(self.wk.weight, std=0.01)
        nn.init.normal_(self.wv.weight, std=0.01)
        nn.init.zeros_(self.out.weight)
        self.w_rate = nn.Linear(D, G * DH, bias=True)
        nn.init.zeros_(self.w_rate.weight)
        nn.init.constant_(self.w_rate.bias, -2.0)
        self.w_eta = nn.Linear(D, G * DH, bias=True)
        nn.init.zeros_(self.w_eta.weight)
        nn.init.constant_(self.w_eta.bias, -2.0)
        self.moe_experts = moe_experts
        if moe_experts:
            self.experts = nn.ModuleList([
                nn.Sequential(nn.Linear(D, 2 * D), nn.SiLU(), nn.Linear(2 * D, D))
                for _ in range(moe_experts)])
            self.ln_ffn = MARMS(D)

    def ffn_moe(self, x, dom=None):
        """MoE FFN: oracle 域路由 (M 专用); dom=None → 专家 0."""
        h = self.ln_ffn(x)
        return x + self.experts[dom if dom is not None else 0](h)

    def _scan(self, x, s0=None):
        B, L, _ = x.shape
        h = self.ln(x)
        q = self.wq(h).view(B, L, H, DH)
        k = self.wk(h).view(B, L, G, DH)
        v = self.wv(h).view(B, L, G, DH)
        rate = F.softplus(self.w_rate(h)).view(B, L, G, DH)
        eta = torch.sigmoid(self.w_eta(h)).view(B, L, G, DH)
        y = delta_scan_triton(k, v, rate, eta, q, KV_H, SLOTS, H // KV_H, None, s0)
        return y.view(B, L, KV_H, SLOTS, H // KV_H, DH).sum(3).reshape(B, L, H * DH)

    def forward(self, x, s0=None):
        return x + self.out(self._scan(x, s0).to(x.dtype))

    def scan_states(self, x, s0=None):
        """无梯度: (y (B,L,H*DH), 末状态 (B,G,DH,DH)) — s0 接力用."""
        B, L, _ = x.shape
        h = self.ln(x)
        q = self.wq(h).view(B, L, H, DH)
        k = self.wk(h).view(B, L, G, DH)
        v = self.wv(h).view(B, L, G, DH)
        rate = F.softplus(self.w_rate(h)).view(B, L, G, DH)
        eta = torch.sigmoid(self.w_eta(h)).view(B, L, G, DH)
        y, s_end = delta_scan_states(k, v, rate, eta, q, KV_H, SLOTS, H // KV_H, s0=s0)
        return y.view(B, L, KV_H, SLOTS, H // KV_H, DH).sum(3).reshape(B, L, H * DH), s_end


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = MARMS(D)
        self.gate = nn.Linear(D, 2 * D, bias=False)
        self.up = nn.Linear(D, 2 * D, bias=False)
        self.down = nn.Linear(2 * D, D, bias=False)

    def forward(self, x):
        h = self.ln(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Station(nn.Module):
    def __init__(self, moe_experts=0):
        super().__init__()
        self.attn = DeltaAttn(moe_experts=moe_experts)
        self.ffn_dense = FFN()

    def forward(self, x, s0=None, dom=None):
        x = self.attn(x, s0=s0)
        if self.attn.moe_experts:
            return self.attn.ffn_moe(x, dom)
        return self.ffn_dense(x)


class Net(nn.Module):
    """R 域专家 & M 底座 同构 (M 加 MoE FFN)."""

    def __init__(self, moe_experts=0):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.st1 = Station(moe_experts=moe_experts)
        self.st2 = Station(moe_experts=moe_experts)
        self.ln_f = MARMS(D)
        self.head = nn.Linear(D, V, bias=False)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, ids, s0_list=None, dom=None):
        s0s = s0_list if s0_list is not None else [None, None]
        h = self.emb(ids)
        h = self.st1(h, s0=s0s[0], dom=dom)
        h = self.st2(h, s0=s0s[1], dom=dom)
        return self.head(self.ln_f(h))


def ce(logits, x):
    return F.cross_entropy(logits[:, :-1].reshape(-1, V).float(),
                           x[:, 1:].reshape(-1), ignore_index=0)


# ---------------- 数据 ----------------
def main():
    torch.manual_seed(0)
    rng = random.Random(0)
    chains = make_chains()

    train_blocks = {d: gen_blocks(chains, d, 40, SEG) for d in range(N_DOM)}
    eval_blocks = {d: gen_blocks(chains, d, 2, SEG) for d in range(N_DOM)}
    eval_mix = torch.cat([eval_blocks[i % N_DOM][i // N_DOM]
                          for i in range(2 * N_DOM)], dim=1).to(DEV)   # (1, 6144)
    dom_order = [i % N_DOM for i in range(2 * N_DOM)]
    log("eval_mix %s | 域顺序 %s" % (tuple(eval_mix.shape), dom_order))

    # ---- 阶段1: 共享基座 (域盲混合 round-robin) ----
    base = Net().to(DEV)
    log("base %.2fM, 预训练 %d 步 (混合 round-robin)" % (count_params(base), STEPS_BASE))
    o = torch.optim.AdamW(base.parameters(), lr=LR)
    for stp in range(STEPS_BASE):
        d = stp % N_DOM
        x = train_blocks[d][rng.randrange(40)].to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = base(x)
        l = ce(lo, x)
        if not torch.isfinite(l):
            o.zero_grad(); continue
        o.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
        o.step()
        if stp % 100 == 0:
            log("  base %d/%d loss %.4f" % (stp, STEPS_BASE, l.item()), flush=True)

    # ---- R: 3 域专家 (base 克隆 → 域 finetune) ----
    R = []
    for d in range(N_DOM):
        m = Net().to(DEV)
        m.load_state_dict(base.state_dict())
        m.train()
        o2 = torch.optim.AdamW(m.parameters(), lr=LR * 0.5)
        r2 = random.Random(100 + d)
        for stp in range(STEPS_FT):
            x = train_blocks[d][r2.randrange(40)].to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = m(x)
            l = ce(lo, x)
            if not torch.isfinite(l):
                o2.zero_grad(); continue
            o2.zero_grad(set_to_none=True)
            l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            o2.step()
        R.append(m.eval())
    log("R: 3 域专家 × %.2fM = %.2fM 总" % (count_params(R[0]), 3 * count_params(R[0])))

    # ---- M: MoE (oracle 域路由) ----
    Mo = Net(moe_experts=N_DOM).to(DEV)
    Mo.load_state_dict(base.state_dict(), strict=False)
    Mo.train()
    o3 = torch.optim.AdamW([p for p in Mo.parameters() if p.requires_grad], lr=LR)
    r3 = random.Random(200)
    for stp in range(STEPS_MOE):
        d = r3.randrange(N_DOM)
        x = train_blocks[d][r3.randrange(40)].to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = Mo(x, dom=d)
        l = ce(lo, x)
        if not torch.isfinite(l):
            o3.zero_grad(); continue
        o3.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(Mo.parameters(), 1.0)
        o3.step()
        if stp % 150 == 0:
            log("  M MoE %d/%d loss %.4f" % (stp, STEPS_MOE, l.item()), flush=True)
    Mo.eval()
    log("M MoE %.2fM (含 3×专家 FFN)" % count_params(Mo))

    # ---- 终评: R 逐段路由 + s0 接力 ----
    def relay_eval():
        s0_list = [None, None]
        tot_ce, n_tok = 0.0, 0
        dom_ce = [0.0] * N_DOM
        dom_n = [0] * N_DOM
        first64 = []
        with torch.no_grad():
            for i in range(2 * N_DOM):
                seg = eval_mix[:, i * SEG:(i + 1) * SEG]
                dom = i % N_DOM
                m = R[dom]
                lo = m(seg, s0_list=s0_list)
                l = ce(lo, seg)
                tot_ce += l.item() * (SEG - 1)
                n_tok += SEG - 1
                dom_ce[dom] += l.item() * (SEG - 1)
                dom_n[dom] += SEG - 1
                if i >= N_DOM:                       # 第 2 轮起: 检查切换后首 64 token
                    first64.append(ce(lo[:, :64], seg[:, :64]).item())
                # s0 接力: 本域模型的 2 站末状态 → 下一段 (跨模型交接)
                h = m.emb(seg)
                _, s1 = m.st1.attn.scan_states(m.st1.attn.ln(h), s0=s0_list[0])
                ln2_in = m.st1.ffn_dense(h + m.st1.attn.out(
                    m.st1.attn._scan(m.st1.attn.ln(h), s0=s0_list[0]).to(h.dtype)))
                _, s2 = m.st2.attn.scan_states(m.st2.attn.ln(ln2_in), s0=s0_list[1])
                s0_list = [s1, s2]
        return tot_ce / n_tok, [c / max(nn_, 1) for c, nn_ in zip(dom_ce, dom_n)], first64

    nll_r, dom_r, first64 = relay_eval()

    # ---- 终评: M 单前向 ----
    with torch.no_grad():
        lo_m = Mo(eval_mix)
    nll_m = ce(lo_m, eval_mix).item()
    # 逐域 M NLL (按段)
    dom_m = [0.0] * N_DOM
    dom_n_m = [0] * N_DOM
    with torch.no_grad():
        for i in range(2 * N_DOM):
            seg = eval_mix[:, i * SEG:(i + 1) * SEG]
            dom = i % N_DOM
            lo = Mo(seg, dom=dom)
            l = ce(lo, seg)
            dom_m[dom] += l.item() * (SEG - 1)
            dom_n_m[dom] += SEG - 1
    dom_m = [c / max(nn_, 1) for c, nn_ in zip(dom_m, dom_n_m)]

    log("=== 终评 (混拼 held-out 6144, 域顺序 %s) ===" % dom_order)
    log("R 序列级接力 (3 域专家, 总 %.2fM): 总NLL %.4f | 各域 %s"
        % (3 * count_params(R[0]), nll_r, ["%.4f" % x for x in dom_r]))
    log("M token级MoE (%.2fM, oracle路由):    总NLL %.4f | 各域 %s"
        % (count_params(Mo), nll_m, ["%.4f" % x for x in dom_m]))
    log("R/M 参数比 %.1f× | R−M = %+.4f" % (3 * count_params(R[0]) / count_params(Mo),
                                            nll_r - nll_m))
    log("R 跨域切换后首64 NLL: %s (段内均值 %.4f — 差值=切换代价)"
        % (["%.4f" % x for x in first64], sum(first64) / len(first64)))
    log("MOE_VS_RELAY DONE", flush=True)


if __name__ == "__main__":
    main()
