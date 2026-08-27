"""ConvASH30: 全层 ConvLinear 化 OpenASH (不绑头), hidden=640x16, 剩余 ~29.7M.

每层: combined 4分支 = 4×ConvLinearT; FFN(ffn1/gate/ffn2) = 3×ConvLinearT
cummax + gen_model 与 OpenASH 语义一致 ([b,s,H,dh] 布局).
"""
import sys
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_linear_triton_train import _ConvLinearFn

VOCAB = 23005
D, LAYERS, HEADS = 640, 16, 8
W_CONV, K_CONV = 64, 9


class ConvLinearT(nn.Module):
    """Triton fwd+bwd 的 ConvLinear (Kw 参数化)."""

    def __init__(self, h, w=W_CONV, k=K_CONV, act=False):
        super().__init__()
        self.kw = nn.Parameter(torch.empty(k, w))
        self.w_out = nn.Parameter(torch.empty(w, 1))
        self.bias = nn.Parameter(torch.zeros(w))
        self.act = act
        nn.init.normal_(self.kw, 0.0, 0.02)
        nn.init.normal_(self.w_out, 0.0, 0.02)

    def forward(self, x):
        return _ConvLinearFn.apply(x.contiguous(), self.kw,
                                   self.w_out.reshape(-1), self.bias, self.act)


class ConvMaxStateSuper(nn.Module):
    def __init__(self, d, heads, w=W_CONV, k=K_CONV):
        super().__init__()
        self.heads = heads
        self.br = nn.ModuleList([ConvLinearT(d, w, k, act=False) for _ in range(4)])
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
        combined4 = torch.stack([o0, o1, o2, o3], dim=2)          # [b,s,4,H,dh]

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


class ConvFFN(nn.Module):
    def __init__(self, d, w=W_CONV, k=K_CONV):
        super().__init__()
        self.ffn1 = ConvLinearT(d, w, k, act=False)
        self.gate = ConvLinearT(d, w, k, act=True)
        self.ffn2 = ConvLinearT(d, w, k, act=False)

    def forward(self, x):
        return self.ffn2(self.ffn1(x) * self.gate(x))


class ConvLayer(nn.Module):
    def __init__(self, d, heads, w=W_CONV, k=K_CONV):
        super().__init__()
        self.attn = ConvMaxStateSuper(d, heads, w, k)
        self.ffn = ConvFFN(d, w, k)
        self.ln = nn.LayerNorm(d)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, state=None):
        x1, state = self.attn(x, state)
        x = self.ln(self.alpha * self.ffn(x1) + (1 - self.alpha) * x)
        return x, state


class ConvASH30(nn.Module):
    def __init__(self, voc=VOCAB, d=D, layers=LAYERS, heads=HEADS,
                 w=W_CONV, k=K_CONV):
        super().__init__()
        self.em = nn.Embedding(voc, d, padding_idx=0)
        self.decoder_layers = nn.ModuleList(
            [ConvLayer(d, heads, w, k) for _ in range(layers)])
        self.head = nn.Linear(d, voc, bias=False)     # 不绑头

    def forward(self, x, state=None):
        x = self.em(x)
        if state is None:
            state = [None] * len(self.decoder_layers)
        for i, layer in enumerate(self.decoder_layers):
            x1, state[i] = layer(x, state[i])
            x = x1 + x
        return self.head(x), state

    @torch.no_grad()
    def generate(self, prompt_ids, steps=60, temperature=0.8, top_k=30,
                 top_p=0.9, rep_penalty=1.15):
        """采样三件套: top_k 过滤 -> softmax -> 重复抑制(概率上 p/penalty 再归一化)
        -> top_p 核过滤 -> multinomial."""
        self.eval()
        dev = next(self.parameters()).device
        p = torch.tensor(prompt_ids, dtype=torch.long, device=dev).unsqueeze(0)
        out, state = self(p)
        for _ in range(steps):
            logits = out[0, -1].float() / temperature
            if top_k:
                v, _ = logits.topk(top_k)
                logits[logits < v[-1]] = -1e9
            probs = F.softmax(logits, -1)
            if rep_penalty and rep_penalty != 1.0:
                seen = p[0].unique()
                probs[seen] /= rep_penalty
                probs /= probs.sum()
            if top_p and top_p < 1.0:
                sp, si = probs.sort(descending=True)
                cum = sp.cumsum(-1)
                n_keep = int((cum < top_p).sum()) + 1
                probs[si[n_keep:]] = 0.0
                probs /= probs.sum()
            tok = torch.multinomial(probs, 1).unsqueeze(0)
            out, state = self(tok, state)
            p = torch.cat([p, tok], 1)
        return p
