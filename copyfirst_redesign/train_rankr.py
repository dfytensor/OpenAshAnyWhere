"""RankRLinear r=1/2/4/8 对比: 30M OpenASH (h=512, L=6, heads=8) MiniMind 预训练.

协议与 ash30m_train.py 一致: 3000 步, bs=16, S=256, cosine, lr=1e-3, seed 0.
对比: 原版 Linear FFN 与 r ∈ {1,2,4,8} 全 FFN 低秩替换.
"""
import os, sys, time, random
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import torch
import torch.nn.functional as F
from open_ash import OpenASH
from rankr import apply_rankr

DEV = "cuda"
PT_CACHE = r"F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt"
OUT = r"F:\OpenASH2605\copyfirst_redesign"
H, L, HEADS = 512, 6, 8
VOCAB = 23005


def train_pretrain(m, tag, steps=3000, bs=16, sl=256, lr=1e-3):
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)[:300000]
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    hist = []
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
        hist.append(loss.item())
    return m, hist


def main():
    results = {}
    for r in [None, 1, 2, 4, 8]:
        torch.manual_seed(0); random.seed(0)
        if r is None:
            tag = "linear"
            m = OpenASH(voc_size=VOCAB, hidden_size=H, num_heads=HEADS, num_layers=L).to(DEV)
        else:
            tag = "r=%d" % r
            m = apply_rankr(OpenASH(voc_size=VOCAB, hidden_size=H,
                                    num_heads=HEADS, num_layers=L).to(DEV), r)
        p = sum(p.numel() for p in m.parameters())
        print("=== %s (%.2fM) ===" % (tag, p / 1e6), flush=True)
        m, hist = train_pretrain(m, tag)
        results[tag] = hist
        torch.save(m.state_dict(), os.path.join(OUT, "ash30m_rankr_%s.pth" % tag.replace("=", "")))
    print("\n最终 loss:")
    for tag, hist in results.items():
        print("  %s: %.3f" % (tag, min(hist)))
    print("完成")


if __name__ == "__main__":
    main()
