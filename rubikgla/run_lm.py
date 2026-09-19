#!/usr/bin/env python3
"""自然语言 LM 对比 (SPEC 阶段二的本地替代: minimind token 序列, OpenASHVoc).
模型: rubik-decay / gla / hybrid(Rubik+SWA 每3层) / tf
配置: 6 层 x d256 x H8(dh32), seq 256, B=64, 2500 步, 每250 步评测固定验证批.
增量落盘可续跑.
"""
import sys, os, time, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import RubikLayer, GLALayer

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "lm_results.json")
PT_CACHE = r"F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt"
DEV = "cuda"
D, L, B, STEPS = 128, 128, 64, 2500
V = 23005


class SWALayer(nn.Module):
    """滑窗因果注意力 (窗口 128), pre-norm."""
    def __init__(self, d, H=8, win=128):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.H, self.dh = H, d // H
        self.win = win
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, L, d = x.shape
        h = self.ln(x)
        qkv = self.qkv(h).view(B, L, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = q @ k.transpose(-1, -2) / self.dh ** 0.5
        i = torch.arange(L, device=x.device)
        causal = i[:, None] >= i[None, :]
        window = (i[:, None] - i[None, :]).abs() < self.win
        att = att.masked_fill(~(causal & window)[None, None], float("-inf"))
        o = torch.softmax(att, -1) @ v
        o = o.transpose(1, 2).reshape(B, L, d)
        return x + self.out(o)


class HybridLayer(nn.Module):
    """奇数位置 SWA / 其余 Rubik 的混合堆栈用."""
    def __init__(self, d, H=8, win=128):
        super().__init__()
        self.rubik = RubikLayer(d, H, decay=True)
        self.swa = SWALayer(d, H, win)

    def forward(self, x, state=None, use_swa=False):
        if use_swa:
            return self.swa(x), state
        return self.rubik(x, state)


class LM(nn.Module):
    def __init__(self, kind, layers=6, d=D, H=8):
        super().__init__()
        self.kind = kind
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(L, d) if kind == "tf" else None
        if kind == "rubik":
            self.stack = nn.ModuleList([RubikLayer(d, H, decay=True) for _ in range(layers)])
        elif kind == "gla":
            self.stack = nn.ModuleList([GLALayer(d, H) for _ in range(layers)])
        elif kind.startswith("hybrid"):
            period = int(kind.replace("hybrid", "") or 3)
            self.kinds = ["swa" if (i + 1) % period == 0 else "rubik"
                          for i in range(layers)]
            self.stack = nn.ModuleList([
                SWALayer(d, H) if k == "swa" else RubikLayer(d, H, decay=True)
                for k in self.kinds])
        elif kind == "tf":
            layer = nn.TransformerEncoderLayer(d, nhead=H, dim_feedforward=4 * d,
                                               batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, num_layers=layers)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, ids):
        x = self.emb(ids)
        if self.kind == "tf":
            p = self.pos(torch.arange(ids.shape[1], device=ids.device)).unsqueeze(0)
            x = x + p
            mask = torch.triu(torch.ones(ids.shape[1], ids.shape[1],
                                         device=ids.device, dtype=torch.bool), 1)
            x = self.tf(x, mask=mask)
        elif self.kind.startswith("hybrid"):
            state = None
            for i, layer in enumerate(self.stack):
                if self.kinds[i] == "swa":
                    x = layer(x)
                else:
                    x, state = layer(x, state)
        else:
            for layer in self.stack:
                x, _ = layer(x)
        return self.head(self.ln_f(x))


def get_batch(seqs, seed, B=B, S=L):
    rng = random.Random(seed)
    xs = []
    for _ in range(B):
        s = seqs[rng.randrange(len(seqs))][:S]
        xs.append(F.pad(s, (0, S - s.numel())))
    x = torch.stack(xs).to(DEV)
    y = x.clone(); y[:, :-1] = x[:, 1:]; y[:, -1] = 0
    return x, y


@torch.no_grad()
def evaluate(m, seqs, seed=777, nb=30):
    m.eval()
    tot, tok = 0.0, 0
    for i in range(nb):
        x, y = get_batch(seqs, seed + i)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = m(x)
        l = F.cross_entropy(logits[:, :-1].reshape(-1, V), y[:, :-1].reshape(-1),
                            ignore_index=0, reduction="sum")
        tot += l.item(); tok += (y[:, :-1] != 0).sum().item()
    m.train()
    return tot / tok


def main():
    torch.manual_seed(0)
    random.seed(0)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    n_val = 5000
    val, train = seqs[:n_val], seqs[n_val:]
    print("train %d val %d" % (len(train), len(val)), flush=True)
    data = {}
    if os.path.exists(RES):
        with open(RES, encoding="utf-8") as f:
            data = json.load(f)

    for kind in ("hybrid2", "hybrid6"):
        if kind in data:
            continue
        torch.manual_seed(0)
        m = LM(kind).to(DEV)
        print("=== %s (%.2fM) ===" % (kind, sum(p.numel() for p in m.parameters()) / 1e6), flush=True)
        opt = torch.optim.AdamW(m.parameters(), lr=6e-4, weight_decay=0.01)
        t0 = time.time()
        for st in range(STEPS):
            m.train()
            x, y = get_batch(train, 30_000 + st)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = m(x)
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, V),
                                       y[:, :-1].reshape(-1), ignore_index=0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if (st + 1) % 250 == 0:
                vl = evaluate(m, val)
                mem = torch.cuda.max_memory_allocated() / 2**30
                print("  %s %4d loss=%.4f val=%.4f mem=%.1fGB (%.0fs)" %
                      (kind, st + 1, loss.item(), vl, mem, time.time() - t0), flush=True)
        v = evaluate(m, val)
        data[kind] = dict(val_nll=round(v, 4), val_bpc=round(v / math.log(2), 4),
                          wall_s=round(time.time() - t0),
                          mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
        with open(RES, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        print("%s FINAL val NLL=%.4f bpc=%.4f" % (kind, v, v / math.log(2)), flush=True)
        del m, opt
        torch.cuda.empty_cache()

    print("\n=== 汇总 (val NLL / bpc) ===")
    for k, r in data.items():
        print("  %-7s %.4f / %.4f" % (k, r["val_nll"], r["val_bpc"]))


import math
if __name__ == "__main__":
    main()
