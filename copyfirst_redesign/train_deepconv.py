"""DeepConv2D 多层卷积对比: 30M OpenASH MiniMind 预训练 (协议同 train_rankr.py).

配置: linear 基线 / ConvLinear(k9) / d2(1->2->1) / d2(1->4->2->1) / d2(1->8->4->1)
"""
import os, sys, time, random
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import torch
import torch.nn.functional as F
from open_ash import OpenASH
from deepconv import apply_d2
from deepconv_triton import apply_d2t

DEV = "cuda"
PT_CACHE = r"F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt"
OUT = r"F:\OpenASH2605\copyfirst_redesign"
H, L, HEADS = 512, 6, 8
VOCAB = 23005


def make_convlinear(m):
    from ash30m_train import apply_conv_ffn
    apply_conv_ffn(m)
    m.to(DEV)
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
    for tag, build in [
        ("linear", lambda: OpenASH(voc_size=VOCAB, hidden_size=H,
                                   num_heads=HEADS, num_layers=L).to(DEV)),
        ("convk9", lambda: make_convlinear(OpenASH(voc_size=VOCAB, hidden_size=H,
                                                   num_heads=HEADS, num_layers=L).to(DEV))),
        ("d2_1_2_1", lambda: apply_d2(OpenASH(voc_size=VOCAB, hidden_size=H,
                                              num_heads=HEADS, num_layers=L).to(DEV), (2,))),
        ("d2_1_4_2_1", lambda: apply_d2(OpenASH(voc_size=VOCAB, hidden_size=H,
                                                num_heads=HEADS, num_layers=L).to(DEV), (4, 2))),
        ("d2_1_8_4_1", lambda: apply_d2(OpenASH(voc_size=VOCAB, hidden_size=H,
                                                num_heads=HEADS, num_layers=L).to(DEV), (8, 4))),
        ("d2t_c2", lambda: apply_d2t(OpenASH(voc_size=VOCAB, hidden_size=H,
                                             num_heads=HEADS, num_layers=L).to(DEV), c=2)),
        ("d2t_c4", lambda: apply_d2t(OpenASH(voc_size=VOCAB, hidden_size=H,
                                             num_heads=HEADS, num_layers=L).to(DEV), c=4)),
    ]:
        ckpt = os.path.join(OUT, "ash30m_%s.pth" % tag)
        if os.path.exists(ckpt):
            print("=== %s 已存在, 跳过 ===" % tag, flush=True)
            continue
        torch.manual_seed(0); random.seed(0)
        m = build()
        p = sum(t.numel() for t in m.parameters())
        print("=== %s (%.2fM) ===" % (tag, p / 1e6), flush=True)
        m = train_pretrain(m, tag)
        torch.save(m.state_dict(), ckpt)
    print("完成")


if __name__ == "__main__":
    main()
