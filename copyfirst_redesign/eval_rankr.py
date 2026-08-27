"""固定评测集评估 rank-r 各配置最终模型: 100 batch 平均 CE (确定性)."""
import sys, random
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import torch
import torch.nn.functional as F
from open_ash import OpenASH
from rankr import apply_rankr

DEV = "cuda"
PT_CACHE = r"F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt"
VOCAB = 23005
H, L, HEADS = 512, 6, 8


@torch.no_grad()
def eval_model(m, seqs, n_batch=100, bs=16, sl=256):
    m.eval()
    tot = 0.0
    for _ in range(n_batch):
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
        tot += loss.item()
    return tot / n_batch


def main():
    torch.manual_seed(0); random.seed(0)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)[:100000]
    for r in [None, 1, 2, 4, 8]:
        tag = "linear" if r is None else "r%d" % r
        m = (OpenASH(voc_size=VOCAB, hidden_size=H, num_heads=HEADS, num_layers=L)
             if r is None else apply_rankr(OpenASH(voc_size=VOCAB, hidden_size=H,
                                                   num_heads=HEADS, num_layers=L), r)).to(DEV)
        sd = torch.load(r"F:\OpenASH2605\copyfirst_redesign\ash30m_rankr_%s.pth" % tag,
                        map_location="cpu", weights_only=True)
        m.load_state_dict(sd)
        p = sum(t.numel() for t in m.parameters()) / 1e6
        torch.manual_seed(0); random.seed(0)
        v = eval_model(m, seqs)
        print("%s (%.2fM): eval loss = %.3f" % (tag, p, v), flush=True)


if __name__ == "__main__":
    main()
