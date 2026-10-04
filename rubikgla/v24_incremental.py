#!/usr/bin/env python3
"""v24 逐 token 增量推理验证:
1) 一致性: 并行扫描前向 vs 状态递归逐步前向, logits 应一致
2) 生成: 短提示逐 token 生成
3) 记忆账本: softmax KV (线性增长) vs RUBIK 状态 (恒定)"""
import sys, os, time, math
import torch
import torch.nn.functional as F
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"F:\OpenASH2605")
from deqlm_v24_hybrid2fast import Hybrid2FastLM, RubikLinearAttnV2, R_LR
from deqlm_v19_tuned23m import rope_cache, apply_rope
from deqlm_v5_30m import PT_CACHE, V, D, H, DEV

HERE = os.path.dirname(os.path.abspath(__file__))
KV_H, DH = 2, D // H


def stable_exp(A, terms=6, max_sq=10):
    with torch.no_grad():
        nrm = A.norm(dim=(-2, -1)).max().item()
    k = 0
    while nrm > 0.5 and k < max_sq:
        A = A / 2
        nrm /= 2
        k += 1
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype).view(*([1] * (A.dim() - 2)), DH, DH)
    E = eye + A
    T = A
    for m in range(2, terms + 1):
        T = T @ A / m
        E = E + T
    for _ in range(k):
        E = E @ E
    return E


class IncrementalV24:
    """逐 token 推理: softmax 部分带 KV cache, RUBIK 桥带常量状态."""

    def __init__(self, m):
        self.m = m.eval()
        self.reset()

    def reset(self):
        self.t = 0
        self.kv = {}                     # site -> (K, V) 各 (1,KV_H,t,DH)
        self.S = torch.zeros(H, DH, DH, device=DEV)              # RUBIK 常量状态
        self.vbuf = deque(maxlen=4)      # RUBIK 短卷积缓冲

    def _gqa(self, site, attn, x):
        h = attn.ln(x)
        q = attn.wq(h).view(1, 1, H, DH).transpose(1, 2)
        k = attn.wk(h).view(1, 1, KV_H, DH).transpose(1, 2)
        v = attn.wv(h).view(1, 1, KV_H, DH).transpose(1, 2)
        cos, sin = rope_cache(self.t + 1, DH, device=DEV)
        q = apply_rope(q, cos[-1:], sin[-1:])
        k = apply_rope(k, cos[-1:], sin[-1:])
        if site in self.kv:
            K, Vc = self.kv[site]
            K = torch.cat([K, k], 2)
            Vc = torch.cat([Vc, v], 2)
        else:
            K, Vc = k, v
        self.kv[site] = (K, Vc)
        Kf = K.repeat_interleave(H // KV_H, 1)
        Vf = Vc.repeat_interleave(H // KV_H, 1)
        att = q @ Kf.transpose(-1, -2) / DH ** 0.5
        o = (torch.softmax(att, -1) @ Vf).transpose(1, 2).reshape(1, 1, D)
        return x + attn.out(o)

    def _block(self, site, blk, x):
        return blk.ffn(self._gqa(site, blk.attn, x))

    def _rubik(self, x):
        a = self.m.rubik_bridge.attn
        h = a.ln_in(x).float()
        q = a.q(h).view(1, 1, H, DH)
        k = a.k(h).view(1, 1, H, DH)
        v_raw = a.v(h).view(1, 1, H, DH)
        # 短因果卷积: out[t] = w3*v[t] + w2*v[t-1] + w1*v[t-2] + w0*v[t-3]
        self.vbuf.append(v_raw)
        v_conv = torch.zeros_like(v_raw)
        for tap in range(4):
            idx = len(self.vbuf) - 1 - (3 - tap)
            if idx >= 0:
                wt = a.dwconv.weight[:, 0, tap].view(1, 1, H, DH)
                v_conv = v_conv + wt * self.vbuf[idx]
        v = v_raw + v_conv
        lam = torch.sigmoid(a.w_lam(h)).view(1, 1, H, DH)
        uv = a.uv(h).view(1, 1, H, 2 * a.r, DH)
        u, w = uv[:, :, :, :a.r, :], uv[:, :, :, a.r:, :]
        Ut, Wt = u.transpose(-2, -1), w.transpose(-2, -1)
        A = Ut @ Wt.transpose(-2, -1) - Wt @ Ut.transpose(-2, -1)   # (1,1,H,DH,DH)
        G = stable_exp(A)
        A_full = lam.squeeze(0).squeeze(0).unsqueeze(-1) * G.squeeze(0).squeeze(0)  # (H,DH,DH)
        k_ = k.squeeze(0).squeeze(0)                                 # (H,DH)
        v_ = v.squeeze(0).squeeze(0)
        W_ = k_.unsqueeze(-1) * v_.unsqueeze(-2)                     # (H,DH,DH)
        self.S = A_full @ self.S + W_
        o = torch.einsum("hp,hpq->hq", q.squeeze(0).squeeze(0), self.S) / math.sqrt(DH)
        o = o.reshape(1, 1, D).to(x.dtype)
        y = x + a.out(o)
        return self.m.rubik_bridge.ffn(y)

    @torch.no_grad()
    def step(self, tok):
        x = self.m.emb(torch.tensor([[tok]], device=DEV))
        x = self._block("pre", self.m.pre, x)
        for j in range(self.m.loop_a.kmax):
            x = x + self.m.loop_a.alpha() * self._block("loopA%d" % j, self.m.loop_a.attn, x)
        x = self._rubik(x)
        for j in range(self.m.loop_c.kmax):
            x = x + self.m.loop_c.alpha() * self._block("loopC%d" % j, self.m.loop_c.attn, x)
        x = self._block("post", self.m.post, x)
        logits = self.m.head(self.m.ln_f(x))
        self.t += 1
        return logits[0, -1].float()


def main():
    m = Hybrid2FastLM().to(DEV)
    m.load_state_dict(torch.load(os.path.join(HERE, "deqlm24_pt_full.pth"), map_location=DEV, weights_only=True))
    m.eval()
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    val = seqs[:200]

    # ═══ 1) 一致性验证 ═══
    print("=" * 56)
    print("[1] 一致性: 并行扫描 vs 状态递归")
    inc = IncrementalV24(m)
    torch.manual_seed(0)
    agree, tot, maxdiff = 0, 0, 0.0
    n_seq, n_tok = 5, 100
    for si in range(n_seq):
        ids = val[si][:n_tok].tolist()
        if len(ids) < 8:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            par = m(torch.tensor([ids], device=DEV))[0].float()
        inc.reset()
        for t in range(len(ids) - 1):
            lg = inc.step(ids[t])
            d = (lg - par[t]).abs().max().item()
            maxdiff = max(maxdiff, d)
            agree += int(lg.argmax().item() == par[t].argmax().item())
            tot += 1
    print("  %d 序列 × %d token: logits 最大差 %.4f, top-1 一致率 %.1f%%"
          % (n_seq, n_tok - 1, maxdiff, 100.0 * agree / tot))

    # ═══ 2) 逐 token 生成 ═══
    from open_ash_voc import OpenASHVoc
    voc = OpenASHVoc(agent_voc_path=r"F:\OpenASH2605\open_ash_voc_agent.json")
    print("\n[2] 逐 token 生成 (RUBIK 状态恒定 20KB)")
    for prompt in ["人工智能是", "中国的首都"]:
        ids = voc.encode(prompt)
        inc.reset()
        for t in ids:
            lg = inc.step(t)
        gen = []
        torch.manual_seed(7)
        for _ in range(48):
            lg = lg.clone()
            for tok in set(gen + ids[-20:]):
                lg[tok] /= 1.2
            vk, ik = torch.topk(lg, 50)
            lg2 = torch.full_like(lg, float("-inf"))
            lg2[ik] = vk
            sp, si2 = torch.sort(lg2, descending=True)
            cum = torch.cumsum(torch.softmax(sp, -1), -1)
            sp[cum - torch.softmax(sp, -1) >= 0.9] = float("-inf")
            lg3 = torch.full_like(lg2, float("-inf"))
            lg3[si2] = sp
            nxt = torch.multinomial(torch.softmax(lg3, -1), 1).item()
            gen.append(nxt)
            lg = inc.step(nxt)
        print("  [提示] %s" % prompt)
        print("  [生成] %s" % voc.decode(gen)[:200])

    # ═══ 3) 记忆与延迟账本 ═══
    print("\n[3] 记忆账本 @ t=256")
    kv_floats = 0
    for K, Vc in inc.kv.values():
        kv_floats += K.numel() + Vc.numel()
    rubik_floats = inc.S.numel()
    print("  softmax KV cache: %d 位置 × %d 站点 = %.1f KB (bf16) — 随长度线性增长"
          % (inc.t, len(inc.kv), kv_floats * 2 / 1024))
    print("  RUBIK 状态: %d 元素 = %.1f KB — 恒定, 与长度无关" % (rubik_floats, rubik_floats * 2 / 1024))
    # 延迟
    inc.reset()
    for t in voc.encode("量子计算"):
        lg = inc.step(t)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        nxt = lg.argmax().item()
        lg = inc.step(nxt)
    torch.cuda.synchronize()
    print("  逐 token 延迟: %.0f ms/token (Python 循环未优化)" % ((time.time() - t0) / 50 * 1000))


if __name__ == "__main__":
    main()
