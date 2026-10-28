#!/usr/bin/env python3
"""v36 端到端最小实验: 真·多state bank 检索 vs 单state, 配对联想回忆任务.

任务: chunk0 埋 8 对 (key→value), chunk1/2 各 16 个 key 查询 → 必须跨 chunk 检索.
读出:
  A) 单state:  全序列扫描, y_t = q_t·S_t                      (线性注意力基线)
  B) 多state:  每 chunk 独立扫描 → bank,
               y_t = 本chunk局部读出 + Σ_j softmax_j(score)·(q_t·S_j)
               score = q_t · W_g · mean_j(S_j)   (检索头零初始化 = Test C 条件)
  C) B + 门辅助损失 (源chunk CE, 暖启动)
端到端: 主损失无任何检索标签; 梯度: softmax → W → summary → 扫描 k,v (稠密).
测量: 召回准确率 / 初始化源写入梯度范数 (bank 短路长衰减链假说).
"""
import torch, time
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda"
torch.manual_seed(0)
B, L, CH, G, DH, NS = 16, 384, 128, 4, 16, 2
NC = L // CH
N_PAIR, N_Q = 8, 16
FILL0, KEY0, VAL0 = 1, 65, 73
V_FILL, V_KEY, V_VAL = 64, 8, 8
VOCAB = VAL0 + V_VAL
D = 64

# ---------------- 数据 ----------------
def make_batch(bs, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(FILL0, FILL0 + V_FILL, (bs, L), generator=g)
    tgt = torch.zeros(bs, L, dtype=torch.long)            # >0 处计损失
    gate_tgt = torch.full((bs, L), -1, dtype=torch.long)  # ≥0 处计门辅助
    cm = torch.zeros(bs, L)                               # 内容掩码: 1 = 允许写入 state
    for b in range(bs):
        pairs = torch.stack([torch.arange(KEY0, KEY0 + V_KEY),
                             VAL0 + torch.randperm(V_VAL, generator=g)], dim=1)
        pairs = pairs[torch.randperm(N_PAIR, generator=g)]
        pos = torch.randperm(CH - 16, generator=g)[:16]
        for i in range(N_PAIR):                               # 每对排序: key 在前, value 在后
            p, q = int(pos[2 * i].item()), int(pos[2 * i + 1].item())
            p, q = min(p, q), max(p, q)
            ids[b, p], ids[b, q] = pairs[i][0], pairs[i][1]
            cm[b, p] = cm[b, q] = 1.0
        for c in (1, 2):
            qpos = torch.randperm(CH - 2, generator=g)[:N_Q] + 1
            for i in range(N_Q):
                t = c * CH + qpos[i]
                ids[b, t] = pairs[i % N_PAIR][0]
                tgt[b, t] = pairs[i % N_PAIR][1]
                gate_tgt[b, t] = 0                        # 源 chunk = 0
    return ids.to(DEV), tgt.to(DEV), gate_tgt.to(DEV), cm.to(DEV)

# ---------------- 扫描 ----------------
def scan(k, v, rate, eta):
    """(B,L,G,DH) → ys (B,G,DH,L,DH), S_end (B,G,DH,DH); 每次从零状态."""
    S = torch.zeros(k.shape[0], G, DH, DH, device=k.device, dtype=torch.float32)
    ys = []
    for t in range(k.shape[1]):
        kt, vt = k[:, t], v[:, t]
        rr = (kt.unsqueeze(-1) * S).sum(-2)
        S = torch.exp(-rate[:, t]).unsqueeze(-1) * (S - eta[:, t].unsqueeze(-1) * kt.unsqueeze(-1) * rr.unsqueeze(-2)) \
            + kt.unsqueeze(-1) * vt.unsqueeze(-2)
        ys.append(S)
    return torch.stack(ys, dim=3), S

def readout(q, ys):
    """q (B,L,G,NS,DH), ys (B,G,DH,L,DH) → (B,L,G*NS*DH)"""
    y = torch.einsum("blgnd,bgdls->blgns", q, ys)
    return y.reshape(*q.shape[:2], G * NS * DH)

# ---------------- 模型 ----------------
class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D)
        self.ln = nn.LayerNorm(D)
        self.wq = nn.Linear(D, G * NS * DH, bias=False)
        self.wk = nn.Linear(D, G * DH, bias=False)
        self.wv = nn.Linear(D, G * DH, bias=False)
        self.w_rate = nn.Linear(D, G * DH, bias=True)
        self.w_eta = nn.Linear(D, G * DH, bias=True)
        nn.init.normal_(self.emb.weight, std=0.6)
        nn.init.constant_(self.w_rate.bias, -4.0)   # rate≈0.018 → 128 步保持 ~0.10
        nn.init.constant_(self.w_eta.bias, -2.0)    # eta≈0.12
        self.shift_w = nn.Parameter(torch.tensor(0.5))   # token-shift: 邻token混入 → k 上下文化 (RWKV 同款)

    def qkvr(self, ids, cm):
        e = self.ln(self.emb(ids))
        e_prev = torch.zeros_like(e)
        e_prev[:, 1:] = e[:, :-1]
        h = e + self.shift_w * e_prev               # 绑定机制: value 写入的 k 携带前缀 key 成分
        vw = lambda x, n: x.view(x.shape[0], x.shape[1], *n).float()
        k = vw(self.wk(h), (G, DH)) * cm.unsqueeze(-1).unsqueeze(-1)   # 写入门: 非内容不写入
        v = vw(self.wv(h), (G, DH)) * cm.unsqueeze(-1).unsqueeze(-1)
        return (vw(self.wq(h), (G, NS, DH)), k, v,
                F.softplus(vw(self.w_rate(h), (G, DH))), torch.sigmoid(vw(self.w_eta(h), (G, DH)))), h

class SingleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tr = Trunk()
        self.out = nn.Linear(G * NS * DH, D, bias=False)
        self.head = nn.Linear(D, VOCAB, bias=False)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, ids, gate_tgt=None, cm=None):
        (q, k, v, r, e), h = self.tr.qkvr(ids, cm)
        ys, _ = scan(k, v, r, e)
        return self.head(self.out(readout(q, ys).to(h.dtype)) + h), None

class MultiModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tr = Trunk()
        self.out = nn.Linear(G * NS * DH, D, bias=False)
        self.head = nn.Linear(D, VOCAB, bias=False)
        self.w_ret = nn.ModuleList([nn.Linear(DH, DH) for _ in range(G)])
        self.logit_scale = nn.Parameter(torch.tensor(4.0))
        self.ln_loc = nn.LayerNorm(G * NS * DH)            # 双路径各自归一化: 消除尺度竞争
        self.ln_bank = nn.LayerNorm(G * NS * DH)
        self.bank_scale = nn.Parameter(torch.tensor(0.5))  # 可学习混合系数
        for m in self.w_ret:
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)   # Test C 条件: 零初始化

    def forward(self, ids, gate_tgt=None, cm=None):
        (q, k, v, r, e), h = self.tr.qkvr(ids, cm)
        Bx = ids.shape[0]
        y = torch.zeros(Bx, L, G * NS * DH, device=DEV)
        gate_logp = []
        bank = []                                              # (J, B, G, DH, DH)
        for c in range(NC):
            sl = slice(c * CH, (c + 1) * CH)
            ys_c, S_c = scan(k[:, sl], v[:, sl], r[:, sl], e[:, sl])
            y[:, sl] = self.ln_loc(readout(q[:, sl], ys_c))    # 本 chunk 局部读出 (归一化)
            if c > 0:
                Sst = torch.stack(bank)                        # (J,B,G,DH,DH)
                summ = Sst.mean(dim=-1)                        # (J,B,G,DH)
                ws = torch.stack([self.w_ret[g](summ[:, :, g]) for g in range(G)], dim=1)  # (J,G,B,DH)
                qc = q[:, sl].view(Bx, CH, G, NS, DH)          # (B,CH,G,NS,DH)
                scores = torch.einsum("blgnd,jgbd->jblg", qc, ws)       # (J,B,CH,G)
                w = torch.softmax(self.logit_scale * scores, dim=0)    # chunk 维 softmax
                yb = torch.einsum("jblg,jbgis,blgnd->blgns", w, Sst, qc)
                y[:, sl] += self.bank_scale * self.ln_bank(yb.reshape(Bx, CH, G * NS * DH))
                if gate_tgt is not None:
                    m = gate_tgt[:, sl] >= 0
                    if m.any():
                        lp0 = torch.log_softmax(self.logit_scale * scores, dim=0)[0]   # (B,CH,G)
                        gate_logp.append(lp0[m].mean())
            bank.append(S_c)
        return self.head(self.out(y.to(h.dtype)) + h), (-torch.stack(gate_logp).mean() if gate_logp else None)

# ---------------- 训练后内部探针 ----------------
def probe(model, tag):
    """训后解码探针: 用模型自己的 out/head 解码各读出分量, 测目标 value 的 logit margin."""
    ids, tgt, gate, cm = make_batch(16, seed=999)
    tr = model.tr
    is_multi = isinstance(model, MultiModel)
    with torch.no_grad():
        e = tr.ln(tr.emb(ids))
        e_prev = torch.zeros_like(e)
        e_prev[:, 1:] = e[:, :-1]
        h = e + tr.shift_w * e_prev
        vw = lambda x, n: x.view(x.shape[0], x.shape[1], *n).float()
        q = vw(tr.wq(h), (G, NS, DH))
        k = (vw(tr.wk(h), (G, DH)) * cm.unsqueeze(-1).unsqueeze(-1)).float()
        v = (vw(tr.wv(h), (G, DH)) * cm.unsqueeze(-1).unsqueeze(-1)).float()
        rate = F.softplus(vw(tr.w_rate(h), (G, DH))).float()
        eta = torch.sigmoid(vw(tr.w_eta(h), (G, DH))).float()
        ysA, _ = scan(k, v, rate, eta)                                     # A: 全扫描
        Bx = ids.shape[0]
        qf = q.view(Bx, L, G, NS, DH)
        yA = torch.einsum("blgnd,bgdls->blgnd", qf, ysA).permute(0, 1, 2, 4, 3).reshape(Bx, L, G * NS * DH)
        logit_yA = model.head(model.out(yA.to(h.dtype)))                   # A: y-only 解码
        line = f"  [探针 {tag}]"
        # 指标: chunk-1 查询上, 目标 value 的 logit − 8 个 value 的平均 logit
        qm = (tgt > 0)
        qm[:, :CH] = False
        qm[:, 2 * CH:] = False
        idx = qm.nonzero(as_tuple=True)
        tv = (tgt[idx] - VAL0).unsqueeze(1)
        vlg = logit_yA[idx][:, VAL0:VAL0 + V_VAL]
        line += f" A全state y-only margin {(vlg.gather(1, tv).mean() - vlg.mean(1).mean()).item():+.3f}"
        if is_multi:
            sl = slice(CH, 2 * CH)
            S0 = scan(k[:, :CH], v[:, :CH], rate[:, :CH], eta[:, :CH])[1]
            Sst = S0.unsqueeze(0)                                          # J=1
            summ = Sst.mean(dim=-1)
            ws = torch.stack([model.w_ret[g](summ[:, :, g]) for g in range(G)], dim=1)
            qc = q[:, sl].view(-1, CH, G, NS, DH)
            scores = torch.einsum("blgnd,jgbd->jblg", qc, ws)
            w = torch.softmax(model.logit_scale * scores, dim=0)
            yb = torch.einsum("jblg,jbgis,blgnd->blgnd", w, Sst, qc).permute(0, 1, 2, 4, 3).reshape(-1, G * NS * DH)
            ys_l = scan(k[:, sl], v[:, sl], rate[:, sl], eta[:, sl])[0]
            yl = torch.einsum("blgnd,bgdls->blgnd", qc, ys_l).permute(0, 1, 2, 4, 3).reshape(-1, G * NS * DH)
            vlb = model.head(model.out(yb.to(h.dtype)))[:, VAL0:VAL0 + V_VAL]
            vll = model.head(model.out(yl.to(h.dtype)))[:, VAL0:VAL0 + V_VAL]
            line += (f" | B bank margin {(vlb.gather(1, tv).mean() - vlb.mean(1).mean()).item():+.3f}"
                     f" | B local margin {(vll.gather(1, tv).mean() - vll.mean(1).mean()).item():+.3f}")
        print(line, flush=True)

import statistics as st

# ---------------- 训练 ----------------
def align_init(model):
    """q头0 = k: 同 token 内积自动对齐, 检索信号第 0 步即存在 (解耦'对齐学习'与'状态携带')."""
    with torch.no_grad():
        wk = model.tr.wk.weight.view(G, DH, D)
        wq = model.tr.wq.weight.view(G, NS, DH, D)
        wq[:, 0] = wk
        wq[:, 1] = 0.0
    return model

def run(model_cls, steps, lr, seed, aux_w, tag, align=False, freeze_until=0):
    torch.manual_seed(seed)
    model = model_cls().to(DEV)
    if align:
        model = align_init(model)
    def set_freeze(fz):
        for p in model.tr.parameters():
            p.requires_grad = not fz
    set_freeze(freeze_until > 0)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    ids_v, tgt_v, gate_v, cm_v = make_batch(32, seed=777)
    qm = tgt_v > 0
    t0 = time.time()
    for st in range(steps):
        ids, tgt, gate, cm = make_batch(B, seed=st)
        opt.zero_grad()
        logits, aux = model(ids, gate, cm)
        mask = tgt > 0
        loss = F.cross_entropy(logits[mask].float(), tgt[mask])
        if aux is not None and aux_w > 0:
            loss = loss + aux_w * aux
        loss.backward()
        opt.step()
        if st + 1 == freeze_until:
            set_freeze(False)
            opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
            print(f"  [{tag}] -- 主干解冻, 全参联合训练 --", flush=True)
        if st in (0, 1, steps // 2) or st == steps - 1:
            with torch.no_grad():
                lv, _ = model(ids_v, gate_v, cm_v)
            acc = (lv[qm].argmax(-1) == tgt_v[qm]).float().mean().item()
            aux_s = f"  gateNLL {aux.item():.3f}" if aux is not None else ""
            print(f"  [{tag}] step {st:>4}  loss {loss.item():.4f}  val_acc {acc:.3f}{aux_s}", flush=True)
    print(f"  [{tag}] 用时 {time.time()-t0:.0f}s", flush=True)
    probe(model, tag)
    return model

# ---------------- 0) 初始化梯度: 源写入是否收到信号 ----------------
print("=" * 70)
print("0) 初始化梯度 (未训练): 源 token 行 vs filler 行的 emb 梯度范数")
print("=" * 70)
for cls, name in ((SingleModel, "单state"), (MultiModel, "多state")):
    torch.manual_seed(0)
    m = cls().to(DEV)
    ids, tgt, gate, cm = make_batch(B, seed=0)
    logits, _ = m(ids, gate, cm)
    mask = tgt > 0
    F.cross_entropy(logits[mask].float(), tgt[mask]).backward()
    g = m.tr.emb.weight.grad
    gk = g[KEY0:KEY0 + V_KEY].norm(dim=-1).mean().item()
    gf = g[FILL0:FILL0 + V_FILL].norm(dim=-1).mean().item()
    gv = g[VAL0:VAL0 + V_VAL].norm(dim=-1).mean().item()
    print(f"  {name:8s}: key行 {gk:.3e} | value行 {gv:.3e} | filler行 {gf:.3e} | key/filler {gk/max(gf,1e-12):.1f}×")

# ---------------- 1) 端到端训练 ----------------
print()
print("=" * 70)
print("1) token-shift 后端到端 1200 步 (对齐初始化; B 带 0.3 门辅助)")
print("=" * 70)
# run(SingleModel, 1200, 1.5e-3, 1, 0.0, "A 单state+shift", align=True)  # 已测: acc 0.265-0.273, y-only margin +0.059 (种子可复现)
run(MultiModel, 1200, 1.5e-3, 1, 0.3, "B 多state+shift", align=True)
print("\nDONE")
