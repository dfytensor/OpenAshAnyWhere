#!/usr/bin/env python3
"""v36-MultiState v2: 全长连续 v31 前向 (零基座损失) + chunk 边界前缀状态 bank 检索读出.

v1 教训 (见 V36_E2E_FINDINGS.md): 独立分块编码使 23 站基座塌缩 (loss 2.23→2.79).
v2: 编码 = 完整 v31 前向 (v36 ⊇ v31); bank = 同一权重 no-grad capture 在
    chunk 边界 (63,127,191,255) 收割的 5 站前缀状态 (detach).
    前缀态保持优势: 位置 t 读边界 c, 内容仅衰减 (t−c·CHUNK 与 c 的较小者), 与 t 无关.
配方 (轮9-10): 读出路径 MARMSNorm · bank 固定混入 + 0→1 warmup · 检索头端到端 ·
    未来/本块掩码 · 从 deqlmv31r_pt_full.pth 热启动.
终评: 512 后缀 NLL 三方 — v31 零state / v31+relay-s0 / v36-MS.
"""
import sys, os, time, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deqlm_v35_relay as M
from deqlm_v5_30m import get_batch, PT_CACHE

DEV = "cuda"
D, G, DH = M.D, M.KV_H * M.SLOTS, M.DH
V = M.V
STATIONS = 5
CHUNK = 64
FINETUNE_STEPS = 1200
LR = 5e-5
BS = 8
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "deqlmv31r_pt_full.pth")


def log(msg):
    print(msg, flush=True)


class MSReadout(nn.Module):
    """bank 检索读出: 每站 softmax(q·W·summary(S_k))·(q·S_k), 未来/本块掩码."""

    def __init__(self):
        super().__init__()
        self.ms_wq = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(STATIONS)])
        self.ms_wscore = nn.ModuleList([nn.Linear(DH, DH) for _ in range(STATIONS)])
        self.ms_down = nn.ModuleList([nn.Linear(G * DH, D, bias=False) for _ in range(STATIONS)])
        self.ms_ln = M.MARMSNorm(D)
        for j in range(STATIONS):
            nn.init.normal_(self.ms_wq[j].weight, std=0.02)
            nn.init.zeros_(self.ms_wscore[j].weight)
            nn.init.zeros_(self.ms_wscore[j].bias)
            nn.init.normal_(self.ms_down[j].weight, std=0.002)  # 小而非零: 梯度可达上游, 起始扰动≈0

    def forward(self, h, banks, chunk):
        """h (B,L,D) float; banks: 5 × (B,n_b,G,DH,DH) detached → (B,L,D) 读出增量."""
        B, L, _ = h.shape
        K = banks[0].shape[1]
        cidx = (torch.arange(L, device=h.device) // chunk).clamp(max=K - 1)
        y_ms = torch.zeros_like(h)
        for j in range(STATIONS):
            Sst = banks[j].float()                              # (B,K,G,DH,DH)
            summ = Sst.mean(dim=-1)                             # (B,K,G,DH)
            q = self.ms_wq[j](h).view(B, L, G, DH).float()
            ws = self.ms_wscore[j](summ)
            scores = torch.einsum("blgd,bkgd->blk", q, ws)
            mask = torch.arange(K, device=h.device).view(1, 1, K) >= cidx.view(1, L, 1)
            w = torch.softmax(scores.masked_fill(mask, -1e9), dim=-1)
            w = w * (cidx > 0).float().view(1, L, 1)            # chunk0 无历史
            y_read = torch.einsum("blk,blgd,bkgdt->blgt", w, q, Sst)
            y_ms = y_ms + self.ms_down[j](y_read.reshape(B, L, G * DH).to(h.dtype))
        return y_ms                                           # 无归一化: 0.02 小初始化 → 起始≈0, 随训练生长


class DeltaLoopLM_MS(nn.Module):
    """v36 v2: 全长 v31 前向 + 边界 bank 读出 (ramp=0 时严格 ≡ v31)."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.ms = MSReadout()

    def forward(self, ids, chunk=CHUNK, ramp=1.0):
        B, L = ids.shape
        bounds = list(range(chunk - 1, L, chunk))
        was_training = self.net.training
        with torch.no_grad():                                   # bank (detached)
            _, banks = self.net.capture_states(ids, boundary_idx=bounds)
        if was_training:
            self.net.train()                                    # 修复: capture 内部 eval() 必须恢复
        banks = [b.detach() for b in banks]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hh = self.net.emb(ids)
            hh = self.net.pre(hh)                           # 全长连续 23 站 (v31 精确, 到 ln_f 前)
            hh = self.net.loop_a(hh)
            hh = self.net.bridge(hh)
            hh = self.net.loop_c(hh)
            hh = self.net.post(hh)
        h = hh.float()
        if ramp != 0.0:
            h = h + ramp * self.ms(h, banks, chunk)
        return self.net.head(self.net.ln_f(h))


@torch.no_grad()
def nll_at(model, ids):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = model(ids)
    return F.cross_entropy(lo.float()[:, :-1].reshape(-1, lo.shape[-1]),
                           ids[:, 1:].reshape(-1), ignore_index=0).item()


def main():
    torch.manual_seed(0)
    random.seed(0)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    train = seqs[:int(0.95 * len(seqs))]
    val = [s for s in seqs[int(0.95 * len(seqs)):][:2000] if len(s) >= 256]
    log("v36-MS v2: 数据 %d 条 (train %d / val≥256 %d)" % (len(seqs), len(train), len(val)))

    net = M.DeltaLoopLM()
    net.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    net = net.to(DEV).eval()
    log("v31 checkpoint 加载 OK")

    ids_v = torch.stack([s[:256] for s in val[:64]]).to(DEV)
    base256 = nll_at(net, ids_v)
    log("基线 v31 NLL@256 = %.4f" % base256)

    model = DeltaLoopLM_MS(net).to(DEV)
    log("v36-MS v2 新增参数: %.2fM" % (sum(p.numel() for _, p in model.ms.named_parameters()) / 1e6))

    ids512 = torch.stack([torch.cat([val[i], val[i + 64]])[:512]
                          for i in range(32) if len(val[i + 64]) >= 256][:32]).to(DEV)
    tgt512 = ids512[:, 257:512]

    def suffix_nll(logits, start=0):
        lo = logits[:, start:start + 255].float()
        return F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tgt512.reshape(-1), ignore_index=0).item()

    with torch.no_grad():
        lo_zero = net(ids512[:, 256:])
        s0_list = net.capture_states(ids512[:, :256])
        lo_relay = net(ids512[:, 256:], s0_list=s0_list)
        lo_ms = model(ids512, ramp=1.0)
        lo_ms0 = model(ids512, ramp=0.0)
    r0 = F.cross_entropy(lo_ms0.float()[:, :-1].reshape(-1, V), ids512[:, 1:].reshape(-1),
                         ignore_index=0).item()
    log("sanity ramp=0 (应≈v31): %.4f" % r0)
    log("512后缀 NLL [finetune前]: 零state %.4f | relay-s0 %.4f | v36-MS %.4f"
        % (suffix_nll(lo_zero, 0), suffix_nll(lo_relay, 0), suffix_nll(lo_ms, 256)))

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95), fused=True)
    rng = random.Random(42)
    t0 = time.time()
    for stp in range(FINETUNE_STEPS):
        x, y = get_batch(train, bs=BS, device=DEV, rng=rng)
        opt.zero_grad(set_to_none=True)
        lo = model(x, ramp=1.0)
        loss = F.cross_entropy(lo.float().reshape(-1, V), y.reshape(-1), ignore_index=0)
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if stp % 100 == 0 or stp == FINETUNE_STEPS - 1:
            model.eval()
            vn = nll_at(model, ids_v)
            model.train()
            log("step %d/%d loss %.4f val256 %.4f (基线 %.4f) (%.0fs)"
                % (stp, FINETUNE_STEPS, loss.item(), vn, base256, time.time() - t0))

    model.eval()
    v256 = nll_at(model, ids_v)
    with torch.no_grad():
        lo_ms = model(ids512, ramp=1.0)
    log("=== 终评 ===")
    log("NLL@256:     v31 基线 %.4f → v36-MS %.4f (Δ %+.4f)" % (base256, v256, v256 - base256))
    log("512后缀 NLL: 零state %.4f | relay-s0 %.4f | v36-MS %.4f"
        % (suffix_nll(lo_zero, 0), suffix_nll(lo_relay, 0), suffix_nll(lo_ms, 256)))
    torch.save(model.state_dict(), os.path.join(HERE, "v36ms2_finetuned.pth"))
    log("checkpoint: v36ms2_finetuned.pth 已保存")


if __name__ == "__main__":
    main()
