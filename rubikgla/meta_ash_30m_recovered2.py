#!/usr/bin/env python3
"""
30M 公平对比: 原版 ConvMaxStateSuper vs MetaMaxState-30M (逐头 ConvLinearT 参数化)
两者参数量对齐 (~29.7M), 同数据同种子 3000 步.
"""
import sys, json, time
import numpy as np
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_linear_triton_train import _ConvLinearFn

DEV = "cuda"
VOCAB, D, LAYERS, HEADS = 23005, 640, 16, 8
W_CONV, K_CONV = 64, 9
ETA, RHO = 0.02, 0.5


def convlinear_ref(x, kw, w_out, bias, act=False):
    """ConvLinearT 纯 torch 参考 (与 test_cl_grad.py 一致)."""
    b, s, h = x.shape
    k, w = kw.shape
    p = k // 2
    xp = F.pad(x, (p, p), mode="replicate")
    A = xp.unfold(2, k, 1)                                # [b,s,h,k]
    T = torch.einsum("bshk,kw->bshw", A, kw) + bias       # [b,s,h,w]
    if act:
        T = F.relu(T)
    return T @ w_out                                      # [b,s,h]


class CL(nn.Module):
    """ConvLinearT (Triton), 同参数量 (k*w + w + w = 704)."""
    def __init__(self, act=False):
        super().__init__()
        self.kw = nn.Parameter(torch.randn(K_CONV, W_CONV) * 0.02)
        self.w_out = nn.Parameter(torch.randn(W_CONV, 1) * 0.02)
        self.bias = nn.Parameter(torch.zeros(W_CONV))
        self.act = act

    def forward(self, x):
        return _ConvLinearFn.apply(x.contiguous(), self.kw,
                                   self.w_out.reshape(-1), self.bias, self.act)


class OrigMaxState(nn.Module):
    """原版 ConvMaxStateSuper (convash30.py 逐行复刻, 仅 br 换 torch 参考)."""
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.br = nn.ModuleList([CL() for _ in range(4)])
        self.head_linear = nn.Linear(heads * 5, heads, bias=False)
        self.alpha1 = nn.Parameter(torch.tensor(0.5))
        self.alpha2 = nn.Parameter(torch.tensor(0.5))
        self.alpha3 = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, state=None):
        b, s, d = x.shape
        H = self.heads
        dh = d // H
        o0 = self.br[0](x).view(b, s, H, dh)
        o1 = self.br[1](x).view(b, s, H, dh)
        o2 = self.br[2](x).view(b, s, H, dh)
        o3 = self.br[3](x).view(b, s, H, dh)
        combined4 = torch.stack([o0, o1, o2, o3], dim=2)

        if state is None:
            out4, _ = torch.cummax(o2, dim=1)
            state = out4[:, -1:]
        else:
            out4, _ = torch.cummax(torch.cat([state, o2], dim=1), dim=1)
            out4 = out4[:, 1:]
            state = out4[:, -1:]

        a1, a2, a3 = self.alpha1, self.alpha2, self.alpha3
        result = (o0 * o1 + a1 * o1 + a2 * o3
                  + o0 * (a3 * out4 + o3) + o1 * (o2 + out4) + o2 * out4)
        Wm = self.head_linear.weight.view(H, 5, H)
        cg = (torch.einsum("okh,bskhc->bsoc", Wm[:, :4], combined4)
              + torch.einsum("oh,bshc->bsoc", Wm[:, 4], out4))
        result = result + cg * out4
        return result.permute(0, 1, 3, 2).reshape(b, s, d), state


class MetaMaxState30(nn.Module):
    """MetaMaxState-30M: 原版全部机制 + MetaGRU 门控(也用 ConvLinearT, 参数等级一致) + 繁衍项 + 内稳态R."""
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.br = nn.ModuleList([CL() for _ in range(4)])
        self.gate_r = CL()   # MetaGRU 重置门 (ConvLinearT 参数化)
        self.gate_z = CL()   # MetaGRU 更新门
        self.head_linear = nn.Linear(heads * 5, heads, bias=False)
        self.alpha1 = nn.Parameter(torch.tensor(0.5))
        self.alpha2 = nn.Parameter(torch.tensor(0.5))
        self.alpha3 = nn.Parameter(torch.tensor(0.5))
        self.register_buffer("R", torch.full((1, 1, d), 2.5))
        self.eta = ETA
        self.rho = RHO

    def forward(self, x, state=None):
        b, s, d = x.shape
        H = self.heads
        dh = d // H
        o0 = self.br[0](x).view(b, s, H, dh)
        o1 = self.br[1](x).view(b, s, H, dh)
        o2 = self.br[2](x).view(b, s, H, dh)
        o3 = self.br[3](x).view(b, s, H, dh)
        combined4 = torch.stack([o0, o1, o2, o3], dim=2)

        if state is None:
            out4, _ = torch.cummax(o2, dim=1)
            state = out4[:, -1:]
        else:
            out4, _ = torch.cummax(torch.cat([state, o2], dim=1), dim=1)
            out4 = out4[:, 1:]
            state = out4[:, -1:]

        # 繁衍项 R·h(1-h) 注入 o1 (标量 R.data, 逐元素)
        r_mean = self.R.data.mean()
        h_sig = torch.sigmoid(o1)
        o1 = o1 + r_mean * h_sig * (1 - h_sig) * 0.1
        combined4 = torch.stack([o0, o1, o2, o3], dim=2)

        a1, a2, a3 = self.alpha1, self.alpha2, self.alpha3
        result = (o0 * o1 + a1 * o1 + a2 * o3
                  + o0 * (a3 * out4 + o3) + o1 * (o2 + out4) + o2 * out4)
        Wm = self.head_linear.weight.view(H, 5, H)
        cg = (torch.einsum("okh,bskhc->bsoc", Wm[:, :4], combined4)
              + torch.einsum("oh,bshc->bsoc", Wm[:, 4], out4))
        gen = result + cg * out4                            # [b,s,H,dh]
        gen = gen.permute(0, 1, 3, 2).reshape(b, s, d)

        # MetaGRU 门控
        rg = torch.sigmoid(self.gate_r(x))                  # [b,s,d]
        zg = torch.sigmoid(self.gate_z(x))                  # [b,s,d]
        out = zg * gen + (1 - zg) * rg * x

        # 内稳态 (detach, 不碰 autograd 版本)
        with torch.no_grad():
            self.R.data += self.eta * self.R.data * (self.rho - o1.detach().mean().item())
            self.R.data.clamp_(0.1, 4.0)
        return out, state


class ConvFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn1 = CL()
        self.gate = CL(act=True)
        self.ffn2 = CL()

    def forward(self, x):
        return self.ffn2(self.ffn1(x) * self.gate(x))


class CLayer(nn.Module):
    def __init__(self, attn_cls):
        super().__init__()
        self.attn = attn_cls(D, HEADS)
        self.ffn = ConvFFN()
        self.ln = nn.LayerNorm(D)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, state=None):
        x1, state = self.attn(x, state)
        x = self.ln(self.alpha * self.ffn(x1) + (1 - self.alpha) * x)
        return x, state


class CLM(nn.Module):
    def __init__(self, attn_cls):
        super().__init__()
        self.em = nn.Embedding(VOCAB, D, padding_idx=0)
        self.decoder_layers = nn.ModuleList([CLayer(attn_cls) for _ in range(LAYERS)])
        self.head = nn.Linear(D, VOCAB, bias=False)

    def forward(self, x, targets=None):
        x = self.em(x)
        state = [None] * len(self.decoder_layers)
        for i, layer in enumerate(self.decoder_layers):
            x1, state[i] = layer(x, state[i])
            x = x1 + x
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                   targets[:, 1:].reshape(-1), ignore_index=0)
        return logits, loss


def build_docs():
    from open_ash_voc import OpenASHVoc
    voc = OpenASHVoc(agent_voc_path=r"F:\OpenASH2605\open_ash_voc_agent.json")
    texts = []
    for p in [r"F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl",
              r"F:\OpenASH2605\minimind_data\sft_t2t_mini.jsonl"]:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    dd = json.loads(line)
                    t = dd.get("text", "")
                    if not t and "conversations" in dd:
                        t = "".join(x.get("content", "") for x in dd["conversations"])
                    if len(t) > 500:
                        texts.append(t.replace("\n", ""))
                except Exception:
                    pass
                if len(texts) >= 3000:
                    break
    seq_len = 256
    docs = []
    for txt in texts[:800]:
        ids = voc.encode(txt)[:seq_len]
        ids = [max(0, min(x, VOCAB - 1)) for x in ids]
        if len(ids) < seq_len:
            ids = ids + [0] * (seq_len - len(ids))
        docs.append(torch.tensor([ids], device=DEV))
    return docs


def train(model, docs, steps=3000, bs=8, seed=42, tag=""):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=3e-5)
    model.train()
    t0 = time.time()
    for st in range(steps):
        idx = np.random.randint(0, len(docs), bs)
        x = torch.cat([docs[i] for i in idx])
        logits, loss = model(x, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if st % 1000 == 0 or st == steps - 1:
            print("  [%s] step %d loss=%.4f (%.0fs)" % (tag, st, loss.item(), time.time() - t0), flush=True)
    return loss.item()


def main():
    print("build docs...", flush=True)
    docs = build_docs()
    print("docs:", len(docs), flush=True)

    for name, cls in [("ORIG", OrigMaxState), ("META", MetaMaxState30)]:
        m = CLM(cls).to(DEV)
        print("%s params: %.1fM" % (name, sum(p.numel() for p in m.parameters()) / 1e6), flush=True)
        final = train(m, docs, tag=name)
        print("%s FINAL LOSS = %.4f" % (name, final), flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
