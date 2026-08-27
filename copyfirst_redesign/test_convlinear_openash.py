"""ConvLinear 应用到 OpenASH: 替换 FFN 的投影, 测功能(loss)与速度."""
import sys, time, random
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_ash import OpenASH
from conv_linear import ConvLinear

DEV = "cuda"
CACHE = r"F:\OpenASH2605\minimind_data\pretrain_cached_30000_384.pt"


def make_model(replace_ffn2=False, w=96, layers=4):
    m = OpenASH(voc_size=23005, hidden_size=768, num_heads=8, num_layers=layers).to(DEV)
    if replace_ffn2:
        for layer in m.decoder_layers:
            layer.ffn.ffn2 = ConvLinear(768, w=w, k=3).to(DEV)
    return m


def main():
    torch.manual_seed(0)
    random.seed(0)
    seqs = torch.load(CACHE, map_location="cpu", weights_only=True)
    print("数据: %d 条" % len(seqs))

    B, S = 8, 256

    def train(m, steps=300, tag=""):
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        t0 = time.time()
        losses = []
        for st in range(1, steps + 1):
            xs = []
            for _ in range(B):
                s = seqs[random.randrange(len(seqs))][:S]
                xs.append(torch.nn.functional.pad(s, (0, S - s.numel())))
            x = torch.stack(xs).to(DEV)
            y = x.clone(); y[:, :-1] = x[:, 1:]; y[:, -1] = 0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out, _ = m(x)
                loss = F.cross_entropy(out[:, :-1].reshape(-1, 23005),
                                       y[:, :-1].reshape(-1), ignore_index=0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if st % 100 == 0:
                losses.append(loss.item())
                print("  %s step %d loss=%.3f (%.0fs)" % (tag, st, loss.item(), time.time() - t0),
                      flush=True)
        return losses

    # 1) 参数对比
    m0 = make_model(False)
    m1 = make_model(True, w=96)
    p0 = sum(p.numel() for p in m0.parameters())
    p1 = sum(p.numel() for p in m1.parameters())
    print("原版 4 层: %.1fM | ConvLinear-ffn2(w=96): %.1fM (%.1f%%)" % (
        p0 / 1e6, p1 / 1e6, p1 / p0 * 100))

    # 2) 功能: loss 下降
    print("=== 原版训练 ===")
    l0 = train(m0, 300, "原版")
    print("=== ConvLinear-ffn2 训练 ===")
    l1 = train(m1, 300, "ConvLinear")

    # 3) 步时
    x = torch.randint(2, 23004, (B, S), device=DEV)
    def bench(fn, n=20):
        for _ in range(3): fn()
        torch.cuda.synchronize(); t = time.time()
        for _ in range(n): fn()
        torch.cuda.synchronize(); return (time.time() - t) / n * 1000
    def step(m):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = m(x)
            loss = out.sum()
        loss.backward()
    t0 = bench(lambda: step(m0))
    t1 = bench(lambda: step(m1))
    print("步时(fwd+bwd): 原版 %.1f ms | ConvLinear %.1f ms (%.2fx)" % (t0, t1, t1 / t0))


if __name__ == "__main__":
    main()
