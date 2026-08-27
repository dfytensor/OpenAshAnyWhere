"""OpenASH + ConvLinear 30M 级: MiniMind 预训练 + SFT 对比 (原版 vs ConvLinear 替换).

模型: hidden=512, layers=6, heads=8 (~30M 级)
数据: minimind pretrain_cached (1.27M 样本) + sft_cached (905k)
对比: ① 原版 OpenASH  ② ffn 投影全部替换 ConvLinear(k=9, Triton GEMM)
"""
import os, sys, time, random
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_ash import OpenASH
from conv_linear_triton import convlinear_triton_dot

DEV = "cuda"
PT_CACHE = r"F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt"
SFT_CACHE = r"F:\rowcol_llm\sft_cached_256.pt"
OUT = r"F:\OpenASH2605\copyfirst_redesign"

H, L, HEADS = 512, 6, 8
VOCAB = 23005
W_CONV, K_CONV = 64, 9


def apply_conv_ffn(m):
    for layer in m.decoder_layers:
        cf = ConvFFN()
        layer.ffn.ffn2 = cf
    return m


class ConvFFN(nn.Module):
    """ffn2 替换为 ConvLinear — Triton 前向+反向 (训练全链 kernel 化)."""

    def __init__(self, h=H, w=W_CONV, k=K_CONV):
        super().__init__()
        self.w_in = nn.Parameter(torch.empty(1, w))
        self.w_out = nn.Parameter(torch.empty(w, 1))
        self.conv = nn.Conv2d(1, 1, k, padding=k // 2, bias=True)
        nn.init.normal_(self.w_in, 0.0, 0.02)
        nn.init.normal_(self.w_out, 0.0, 0.02)
        self._kw = None      # Kw 参数化缓存: 直接把 Kw 作为可训练参数 (等价且更简单)
        self.kw = nn.Parameter(torch.empty(k, w))
        nn.init.normal_(self.kw, 0.0, 0.02)
        self.bias = nn.Parameter(torch.zeros(w))

    def forward(self, x):
        from conv_linear_triton_train import _ConvLinearFn
        return _ConvLinearFn.apply(x.contiguous(), self.kw,
                                   self.w_out.reshape(-1), self.bias)


def make_model(conv=False):
    m = OpenASH(voc_size=VOCAB, hidden_size=H, num_heads=HEADS, num_layers=L).to(DEV)
    if conv:
        apply_conv_ffn(m)
        m.to(DEV)   # 替换后的 ConvFFN 参数搬 GPU
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
        if st % 500 == 0:
            print("  %s step %d loss=%.3f (%.0fs)" % (tag, st, loss.item(), time.time() - t0),
                  flush=True)
        sched.step()
    return m


def main():
    torch.manual_seed(0); random.seed(0)
    m0 = make_model(False)
    p0 = sum(p.numel() for p in m0.parameters())
    m1 = make_model(True)
    p1 = sum(p.numel() for p in m1.parameters())
    print("原版: %.1fM | ConvLinear版: %.1fM (%.1f%%)" % (p0 / 1e6, p1 / 1e6, p1 / p0 * 100))

    print("=== 原版预训练 ===")
    train_pretrain(m0, "原版", steps=3000)
    torch.save(m0.state_dict(), os.path.join(OUT, "ash30m_orig.pth"))

    print("=== ConvLinear版预训练 ===")
    train_pretrain(m1, "Conv", steps=3000)
    torch.save(m1.state_dict(), os.path.join(OUT, "ash30m_conv.pth"))
    print("完成")


if __name__ == "__main__":
    main()
