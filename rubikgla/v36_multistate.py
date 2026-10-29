#!/usr/bin/env python3
"""v36-MultiState v3: 全长连续 v31 前向 + 每站中途 bank 注入 (CED 式) + 写入门.

v2 → v3:
  1. 每站中途注入: 站 j 输出后立即加 ramp·readout_j(bank_j), 后续站条件于检索历史 (CED 式)
  2. 写入门 (FINDINGS 必选项): g = sigmoid(w_gate(h)) 逐 token 缩放 k,v,
     bias +3 (≈0.95 近直通起步), 训练学习稀疏写入 (RWKV-6/Mamba 选择性方向)
  3. v31 checkpoint 热启动 (strict=False; 新增 gate/ms 参数保持初始化)
终评: 4096 / 16384 后缀 NLL 三方 — 零state / relay-s0链 / v36-MS.
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
CHUNK = 128
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "deqlmv31r_pt_full.pth")


def log(msg, **kw):
    print(msg, flush=True)


class StationReadout(nn.Module):
    """单站 bank 检索读出."""

    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(D, D, bias=False)
        self.wscore = nn.Linear(DH, DH)
        self.down = nn.Linear(G * DH, D, bias=False)
        nn.init.normal_(self.wq.weight, std=0.02)
        nn.init.zeros_(self.wscore.weight)
        nn.init.zeros_(self.wscore.bias)
        nn.init.normal_(self.down.weight, std=0.002)   # 小而非零: 起始≈0, 梯度可达上游

    def forward(self, h, Sst, chunk):
        """h (B,L,D) → (B,L,D); Sst (B,K,G,DH,DH) detached.
        按 K 循环累加: 避免大中间张量 (16k+ 上一次 einsum 物化 >2s 触发 Windows TDR)."""
        B, L, _ = h.shape
        K = Sst.shape[1]
        cidx = (torch.arange(L, device=h.device) // chunk).clamp(max=K - 1)
        q = self.wq(h.float()).view(B, L, G, DH).float()
        ws = self.wscore(Sst.float().mean(dim=-1))     # (B,K,G,DH)
        scores = torch.einsum("blgd,bkgd->blk", q, ws)
        mask = torch.arange(K, device=h.device).view(1, 1, K) >= cidx.view(1, L, 1)
        w = torch.softmax(scores.masked_fill(mask, -1e9), dim=-1)
        w = w * (cidx > 0).float().view(1, L, 1)       # chunk0 无历史
        y = torch.zeros(B, L, G, DH, device=h.device, dtype=torch.float32)
        for k in range(K):
            y = y + w[:, :, k].unsqueeze(-1).unsqueeze(-1) * torch.einsum(
                "blgd,bgdt->blgt", q, Sst[:, k].float())
        return self.down(y.reshape(B, L, G * DH).to(h.dtype))


def capture_segmented(net, ids, chunk, max_seg=2048, s0_list=None):
    """分段 capture + s0 链: 单内核 ≤2048 步, 避开 Windows TDR (16k+ 必需)."""
    L = ids.shape[1]
    bounds_all = list(range(chunk - 1, L, chunk))
    banks = [[] for _ in range(STATIONS)]
    cur = s0_list
    i = 0
    while i < L:
        j = min(i + max_seg, L)
        seg = ids[:, i:j]
        bnd = [b - i for b in bounds_all if i <= b < j]
        ends, bs = net.capture_states(seg, boundary_idx=bnd, s0_list=cur)
        for k in range(STATIONS):
            banks[k].append(bs[k])
        cur = ends
        i = j
    return [torch.cat(b, dim=1) for b in banks]


class DeltaLoopLM_MS(nn.Module):
    """v36 v3: 全长 v31 前向 + 每站中途 bank 注入 + 写入门 (ramp=0 时 ≡ v31)."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.ms = nn.ModuleList([StationReadout() for _ in range(STATIONS)])

    def forward(self, ids, chunk=CHUNK, ramp=1.0, s0_list=None):
        was_training = self.net.training
        s0s = s0_list if s0_list is not None else [None] * STATIONS
        banks = capture_segmented(self.net, ids, chunk, s0_list=s0s)  # 分段 capture (TDR 安全)
        if was_training:
            self.net.train()                            # 修复: capture 内部 eval() 必须恢复
        banks = [b.detach() for b in banks]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hh = self.net.emb(ids)
            for j, st in enumerate((self.net.pre, self.net.loop_a,
                                    self.net.bridge, self.net.loop_c, self.net.post)):
                hh = st(hh, s0=s0s[j])                  # 站 j (v31 精确, 支持接力 s0)
                if ramp != 0.0:
                    hh = hh + ramp * self.ms[j](hh, banks[j], chunk)   # CED 式中途注入
        h = hh.float()
        return self.net.head(self.net.ln_f(h))

    def capture_states(self, ids, boundary_idx=None, s0_list=None):
        """委托 net.capture_states (轮换/接力接口)."""
        was_training = self.net.training
        with torch.no_grad():
            out = self.net.capture_states(ids, boundary_idx=boundary_idx, s0_list=s0_list)
        if was_training:
            self.net.train()
        return out


def build_ms_from_v31(enable_gate=True, enable_sel=True):
    net = M.DeltaLoopLM()
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    log("v31 加载: missing %d tensors (gate/sel 预期) | unexpected %d" % (len(missing), len(unexpected)))
    for blk in (net.pre, net.loop_a.attn, net.bridge, net.loop_c.attn, net.post):
        blk.attn.use_gate = enable_gate
        blk.attn.use_sel = enable_sel
    return net.to(DEV).eval()


@torch.no_grad()
def nll_at(model, ids, **kw):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = model(ids, **kw)
    return F.cross_entropy(lo.float()[:, :-1].reshape(-1, lo.shape[-1]),
                           ids[:, 1:].reshape(-1), ignore_index=0).item()


def finetune(model, train, steps, bs, lr, chunk, log_every=100, save_path=None, curriculum=False):
    """crash-hardened: 探针前先存档, 探针失败 sync+重试.
    curriculum: 阶段1 (前1/4) 冻结主干只训 5 读头 (lr×5) → 阶段2 全参解冻."""
    def make_opt(params, lrv):
        return torch.optim.AdamW(params, lr=lrv, weight_decay=0.01, betas=(0.9, 0.95), fused=True)
    trunk = list(model.net.parameters())
    heads = list(model.ms.parameters())
    freeze_until = steps // 4 if curriculum else 0
    if curriculum:
        for p in trunk:
            p.requires_grad = False
        opt = make_opt(heads, lr * 5)
    else:
        opt = make_opt([p for p in model.parameters() if p.requires_grad], lr)
    rng = random.Random(42)
    t0 = time.time()
    for stp in range(steps):
        x, y = get_batch(train, bs=bs, device=DEV, rng=rng)
        opt.zero_grad(set_to_none=True)
        ramp = min(1.0, (stp + 1) / 300)            # 暖启动: 永不为 0 (冻结期也保梯度路径)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(x, chunk=chunk, ramp=ramp)
        loss = F.cross_entropy(lo.float().reshape(-1, V), y.reshape(-1), ignore_index=0)
        if not torch.isfinite(loss) or not loss.requires_grad:
            opt.zero_grad(set_to_none=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if curriculum and stp == freeze_until:
            for p in trunk:
                p.requires_grad = True
            opt = make_opt(model.parameters(), lr)
            log("  [课程] 主干解冻, 全参联合 (lr %.1e)" % lr, flush=True)
        if stp % log_every == 0 or stp == steps - 1:
            if save_path:
                torch.save(model.state_dict(), save_path)   # 探针前先存档 (崩溃不丢)
            model.eval()
            try:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                with torch.no_grad():
                    lo_v = model(x, chunk=chunk, ramp=ramp)
                vn = F.cross_entropy(lo_v.float().reshape(-1, V), y.reshape(-1), ignore_index=0).item()
            except Exception as e:
                torch.cuda.empty_cache()
                log("  [探针失败一次, 重试] %r" % (e,))
                with torch.no_grad():
                    lo_v = model(x, chunk=chunk, ramp=ramp)
                vn = F.cross_entropy(lo_v.float().reshape(-1, V), y.reshape(-1), ignore_index=0).item()
            model.train()
            log("step %d/%d loss %.4f train-batch-nll %.4f ramp %.2f (%.0fs)"
                % (stp, steps, loss.item(), vn, ramp, time.time() - t0))


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    train = seqs[:int(0.95 * len(seqs))]
    val = [s for s in seqs[int(0.95 * len(seqs)):][:2000] if len(s) >= 256]
    log("v36-MS v3: 数据 %d 条 (val≥256 %d)" % (len(seqs), len(val)))

    net = build_ms_from_v31()
    ids_v = torch.stack([s[:256] for s in val[:64]]).to(DEV)
    base256 = nll_at(net, ids_v)
    log("基线 v31 NLL@256 = %.4f" % base256)

    model = DeltaLoopLM_MS(net).to(DEV)
    model.train()
    n_new = sum(p.numel() for _, p in model.named_parameters() if not any(k in n_ for k in ("emb", "head", "pre.", "loop_a.", "bridge.", "loop_c.", "post.", "ln_f")))
    log("v36-MS v3 新增参数 ≈ %.2fM" % (n_new / 1e6))
    log("sanity ramp=0 NLL@256: %.4f (应≈%.4f)" % (nll_at(model, ids_v, ramp=0.0), base256))

    # ---- 4096 finetune 2000 步 ----
    LONG, BS, STEPS, LR = 4096, 2, 2000, 3e-5
    def blocks(src, n, ln, r):
        out = []
        while len(out) < n:
            buf, tot = [], 0
            while tot < ln:
                s = src[r.randrange(len(src))]
                buf.append(s)
                tot += len(s)
            out.append(torch.cat(buf)[:ln])
        return out

    finetune(model, train, STEPS, BS, LR, CHUNK, save_path=os.path.join(HERE, "v36ms_v3.pth"))
    torch.save(model.state_dict(), os.path.join(HERE, "v36ms_v3.pth"))

    # ---- 4096 三方终评 ----
    model.eval()
    ids4096 = torch.stack(blocks(val, 8, LONG, random.Random(7))).to(DEV)
    tgt4096 = ids4096[:, 2049:LONG]

    def suffix(logits, start, n):
        lo = logits[:, start:start + n].float()
        return F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tgt4096.reshape(-1), ignore_index=0).item()

    with torch.no_grad():
        lo_zero = net(ids4096[:, 2048:])
        s0 = net.capture_states(ids4096[:, :2048])
        lo_relay = net(ids4096[:, 2048:], s0_list=s0)
        lo_ms = model(ids4096, chunk=CHUNK, ramp=1.0)
    log("4096后缀 NLL: 零state %.4f | relay-s0 %.4f | v36-MS %.4f"
        % (suffix(lo_zero, 0, 2047), suffix(lo_relay, 0, 2047), suffix(lo_ms, 2048, 2047)))

    # ---- 16384 三方终评 (relay 8×2048 链 vs v36 全长 K=128) ----
    LONG2 = 16384
    SUFF = 2048
    ids16k = torch.stack(blocks(val, 4, LONG2, random.Random(11))).to(DEV)
    tgt16k = ids16k[:, LONG2 - SUFF + 1:LONG2]

    def suffix16k(logits, start):
        lo = logits[:, start:start + SUFF - 1].float()
        return F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tgt16k.reshape(-1), ignore_index=0).item()

    with torch.no_grad():
        lo_zero16 = net(ids16k[:, LONG2 - SUFF:])
        cur = ids16k[:, :2048]
        for seg in range(1, LONG2 // 2048):
            s0c = net.capture_states(cur)
            nxt = ids16k[:, seg * 2048:(seg + 1) * 2048]
            out_seg = net(nxt, s0_list=s0c)
            cur = torch.cat([cur, nxt], dim=1)
        lo_relay16 = out_seg                                    # 最后一段的 logits
        lo_ms16 = model(ids16k, chunk=CHUNK, ramp=1.0)
    log("16384后缀 NLL: 零state %.4f | relay-s0链 %.4f | v36-MS %.4f"
        % (suffix16k(lo_zero16, 0), suffix16k(lo_relay16, 0), suffix16k(lo_ms16, LONG2 - SUFF)))
    torch.save(model.state_dict(), os.path.join(HERE, "v36ms_v3.pth"))
    log("checkpoint: v36ms_v3.pth 已保存")
