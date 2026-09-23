#!/usr/bin/env python3
"""CED 简化复现 (DeepSeek-V4.1-Flash 框架, 无 MoE): 编码器处理前缀 P=96,
解码器全局 KV 从编码器末层隐状态投影 (C_l = H_3 W_l^KV), 局部 SWA 逐层,
生成后缀 32 token. 两个模型 + 两个 flat 对照, minimind 数据, 同预算对比."""
import sys, os, time, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import RubikLayer

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "ced_results.json")
PT_CACHE = r"F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt"
DEV = "cuda"
V, D, H, L, B, STEPS = 23005, 128, 4, 3, 64, 2500
P, Q = 96, 32            # 前缀(编码器) / 后缀(解码器生成)
LR = 6e-4


class SWA32(nn.Module):
    """解码器局部滑窗因果注意力 (win=32, 覆盖整个 32 长后缀), pre-norm."""
    def __init__(self, d, H=H, win=32):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.H, self.dh, self.win = H, d // H, win
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, L, d = x.shape
        h = self.ln(x)
        qkv = self.qkv(h).view(B, L, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = q @ k.transpose(-1, -2) / self.dh ** 0.5
        i = torch.arange(L, device=x.device)
        att = att.masked_fill((i[:, None] < i[None, :])[None, None], float("-inf"))
        o = torch.softmax(att, -1) @ v
        return x + self.out(o.transpose(1, 2).reshape(B, L, d))


class CrossAttn(nn.Module):
    """解码器全局: 对编码器投影 KV (C_l, 96 entries) 的跨注意力, pre-norm."""
    def __init__(self, d, H=H):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.H, self.dh = H, d // H
        self.q = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def forward(self, x, C, Z):
        # C: (B,P,d) K;  Z: (B,P,d) V;  x: (B,Q,d)
        B, L, d = x.shape
        h = self.ln(x)
        q = self.q(h).view(B, L, self.H, self.dh).transpose(1, 2)
        k = C.view(B, -1, self.H, self.dh).transpose(1, 2)
        v = Z.view(B, -1, self.H, self.dh).transpose(1, 2)
        att = q @ k.transpose(-1, -2) / self.dh ** 0.5
        o = torch.softmax(att, -1) @ v
        return x + self.out(o.transpose(1, 2).reshape(B, L, d))


class CEDModel(nn.Module):
    """CED: 编码器 3 层处理前缀 -> H3; 解码器每层 KV = H3 @ W_l^KV, 局部 SWA.
    variant: 'rubik' (纯) / 'hybrid2' ([rubik,rubik,swa] x2)."""
    def __init__(self, variant, V=V, d=D):
        super().__init__()
        self.variant = variant
        self.emb = nn.Embedding(V, d)
        self.enc = nn.ModuleList([RubikLayer(d, H, decay=True) for _ in range(L)])
        if variant == "hybrid2":
            self.enc[2] = SWA32(d, H, win=48)   # 编码器第 3 层换 SWA
        self.dec_kv = nn.ModuleList([nn.Linear(d, 2 * d) for _ in range(L)])   # C_l, Z_l
        self.dec_local = nn.ModuleList([SWA32(d, H, win=32) for _ in range(L)])
        self.dec_cross = nn.ModuleList([CrossAttn(d, H) for _ in range(L)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):        # x: (B, P+Q=128)
        h = self.emb(x)
        H3 = h[:, :P]
        for layer in self.enc:
            if isinstance(layer, RubikLayer):
                H3, _ = layer(H3)
            else:
                H3 = layer(H3)
        dec_in = h[:, P:]         # 解码器输入: 后缀 32 token (teacher forcing)
        for l in range(L):
            dec_in = self.dec_local[l](dec_in)
            C = self.dec_kv[l](H3)
            dec_in = self.dec_cross[l](dec_in, C[:, :, :D], C[:, :, D:])
        return self.head(self.ln_f(dec_in))


class FlatModel(nn.Module):
    """flat 对照: 6 层因果处理全序列 (前缀+后缀), 同 prefix-LM 目标."""
    def __init__(self, variant, V=V, d=D):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        if variant == "rubik":
            self.stack = nn.ModuleList([RubikLayer(d, H, decay=True) for _ in range(2 * L)])
        else:
            mods = []
            for i in range(2 * L):
                mods.append(SWA32(d, H, win=64) if (i + 1) % 3 == 0 else RubikLayer(d, H, decay=True))
            self.stack = nn.ModuleList(mods)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = self.emb(x)
        for layer in self.stack:
            if isinstance(layer, RubikLayer):
                h, _ = layer(h)
            else:
                h = layer(h)
        return self.head(self.ln_f(h))


def get_batch(seqs, seed):
    rng = random.Random(seed)
    xs = []
    for _ in range(B):
        s = seqs[rng.randrange(len(seqs))][:P + Q]
        xs.append(F.pad(s, (0, P + Q - s.numel())))
    x = torch.stack(xs).to(DEV)
    # 目标: x[t+1], 对 t in [P-1, P+Q-2], 即后缀 32 个位置
    y = torch.full_like(x, 0)
    y[:, P - 1:P + Q - 1] = x[:, P:P + Q]
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, P - 1:P + Q - 1] = True
    return x, y, mask


@torch.no_grad()
def evaluate(m, name, val, seed=777, nb=30):
    m.eval()
    tot, tok = 0.0, 0
    for i in range(nb):
        x, y, mk = get_batch(val, seed + i)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = m(x)
        if name.startswith("ced"):
            ys = x[:, P + 1:P + Q]
            l = F.cross_entropy(logits[:, :-1].reshape(-1, V), ys.reshape(-1), reduction="sum")
            n = ys.numel()
        else:
            l = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="none") * mk.reshape(-1)
            l = l.sum(); n = int(mk.sum().item())
        tot += l.item(); tok += n
    m.train()
    return tot / tok


def main():
    torch.manual_seed(0)
    random.seed(0)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    val, train = seqs[:5000], seqs[5000:]
    print("train %d val %d" % (len(train), len(val)), flush=True)
    data = {}
    if os.path.exists(RES):
        with open(RES, encoding="utf-8") as f:
            data = json.load(f)

    runs = [("ced_rubik", lambda V: CEDModel("rubik", V)),
            ("ced_hybrid2", lambda V: CEDModel("hybrid2", V)),
            ("flat_rubik", lambda V: FlatModel("rubik", V)),
            ("flat_hybrid2", lambda V: FlatModel("hybrid2", V))]
    for name, ctor in runs:
        if name in data:
            continue
        torch.manual_seed(0)
        m = ctor(V).to(DEV)
        print("=== %s (%.2fM) ===" % (name, sum(p.numel() for p in m.parameters()) / 1e6), flush=True)
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
        t0 = time.time()
        for st in range(STEPS):
            m.train()
            x, y, mk = get_batch(train, 60_000 + st)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = m(x)
            if name.startswith("ced"):
                ys = x[:, P + 1:P + Q]
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), ys.reshape(-1))
            else:
                loss = (F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                                        reduction="none") * mk.reshape(-1)).sum() / mk.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if (st + 1) % 500 == 0:
                vl = evaluate(m, name, val)
                mem = torch.cuda.max_memory_allocated() / 2**30
                print("  %s %4d loss=%.4f val=%.4f mem=%.1fGB (%.0fs)" %
                      (name, st + 1, loss.item(), vl, mem, time.time() - t0), flush=True)
        v = evaluate(m, name, val)
        data[name] = dict(val_nll=round(v, 4), wall_s=round(time.time() - t0))
        with open(RES, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        print("%s FINAL val=%.4f" % (name, v), flush=True)
        del m, opt
        torch.cuda.empty_cache()

    print("\n=== CED 对比 (suffix NLL, 32/128 位置) ===")
    for k, r in data.items():
        print("  %-12s %.4f" % (k, r["val_nll"]))


if __name__ == "__main__":
    main()
