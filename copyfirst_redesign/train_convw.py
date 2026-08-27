"""convk9 缩放实验: 网格宽度 w 与 h 感受野 k 放大, 对比 linear FFN.

配置: conv_w64 / conv_w128 / conv_w256 (k=9) + conv_k15_w64 / conv_k15_w128
协议同 train_deepconv.py: 3000 步 bs=16 cosine seed0, 只替换 ffn2.
"""
import os, sys, time, random
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_ash import OpenASH
from conv_linear_triton_train import _ConvLinearFn

DEV = "cuda"
PT_CACHE = r"F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt"
OUT = r"F:\OpenASH2605\copyfirst_redesign"
H, L, HEADS = 512, 6, 8
VOCAB = 23005


class ConvFFNX(nn.Module):
    """可配置 w/k 的单层宽 ConvLinear (Triton fwd+bwd)."""

    def __init__(self, w=64, k=9):
        super().__init__()
        self.kw = nn.Parameter(torch.empty(k, w))
        self.w_out = nn.Parameter(torch.empty(w, 1))
        self.bias = nn.Parameter(torch.zeros(w))
        nn.init.normal_(self.kw, 0.0, 0.02)
        nn.init.normal_(self.w_out, 0.0, 0.02)

    def forward(self, x):
        return _ConvLinearFn.apply(x.contiguous(), self.kw,
                                   self.w_out.reshape(-1), self.bias, False)


def apply_convw(m, w, k):
    dev = next(m.parameters()).device
    for layer in m.decoder_layers:
        layer.ffn.ffn2 = ConvFFNX(w, k).to(dev)
    return m


def train_pretrain(m, tag, steps=3000, bs=16, sl=256, lr=1e-3):
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)[:300000]
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for st in range(steps):
        m.train()
        xs = []
        for _ in range(bs):
            s = seqs[random.randrange(len(seqs))][:sl]
            xs.append(F.pad(s, (0, sl - s.numel())))
        x = torch.stack(xs).to(DEV)
        y = x.clone(); y[:, :-1] = x[:, 1:]; y[:, -1] = 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = m(x)
            loss = F.cross_entropy(out[:, :-1].reshape(-1, VOCAB), y[:, :-1].reshape(-1),
                                   ignore_index=0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if st % 500 == 0 or st == steps - 1:
            print("  %s step %d loss=%.3f (%.0fs, %.1fms/step)" %
                  (tag, st, loss.item(), time.time() - t0, (time.time() - t0) / (st + 1) * 1000),
                  flush=True)
    return m


def main():
    configs = [
        ("conv_w64", 64, 9),
        ("conv_w128", 128, 9),
        ("conv_w256", 256, 9),
        ("conv_w512", 512, 9),
        ("conv_w1024", 1024, 9),
        ("conv_k15_w64", 64, 15),
        ("conv_k15_w128", 128, 15),
    ]
    for tag, w, k in configs:
        ckpt = os.path.join(OUT, "ash30m_%s.pth" % tag)
        if os.path.exists(ckpt):
            print("=== %s 已存在, 跳过 ===" % tag, flush=True)
            continue
        torch.manual_seed(0); random.seed(0)
        m = apply_convw(OpenASH(voc_size=VOCAB, hidden_size=H,
                                num_heads=HEADS, num_layers=L).to(DEV), w, k)
        p = sum(t.numel() for t in m.parameters())
        print("=== %s (%.2fM, ffn2: %d params) ===" % (tag, p / 1e6, k * w + w + w), flush=True)
        m = train_pretrain(m, tag)
        torch.save(m.state_dict(), ckpt)
    print("完成")


if __name__ == "__main__":
    main()
