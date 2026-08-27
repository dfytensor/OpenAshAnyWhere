"""CLR 上真实 LM: 闭环 weight_decay 是否能打败固定 wd / cosine schedule?
  fixed  : AdamW wd=常数
  cosine : wd 从 wd_max 余弦降到 ~0 (标准 schedule)
  clr    : λ(t)=λ_max·compress(V_train)·gap_signal(val−train), 加进 loss; AdamW wd=0.
          —— 训练拟合后才开始压; 仅当出现过拟合 gap 时强压, gap 消失就放松.
模型: 干净 transformer (ARC_LLM use_sgr=False, 因为 SGR 已证无效).
任务: minimind pretrain (open_ash_voc, VOCAB=23005).
"""
import torch, torch.nn.functional as F, math, os, sys, csv, argparse, time
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_llm import ARC_LLM

DEV = 'cuda'; OUT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get('KINT_CACHE', r'F:\OpenASH2605\train_60m\cache\pt_cache_openash_512_openash.pt')
VOCAB = 23005


class DS(Dataset):
    def __init__(self, d, s): self.d, self.s = d, s
    def __len__(self): return len(self.d)
    def __getitem__(self, i): return self.d[i][:self.s + 1]
    @staticmethod
    def collate(items):
        from torch.nn.utils.rnn import pad_sequence
        p = pad_sequence(items, batch_first=True, padding_value=0)
        return p[:, :-1], p[:, 1:]


def run(cond, steps, seq_len, batch, lr, n_val, seed=0,
        wd_max=0.1, clr_lam=0.1, vfrac=0.7, tau=0.15, gap_margin=0.02, gap_tau=0.03):
    torch.manual_seed(seed)
    print(f'Loading cache...', flush=True)
    data = torch.load(CACHE, weights_only=False)
    val_data, tr_data = data[:n_val], data[n_val:]
    model = ARC_LLM(VOCAB, d=256, n_heads=8, n_layers=4, d_ffn=1024, r_low=64,
                    use_sgr=False, max_len=seq_len+8).to(DEV)
    print(f'[{cond}] params={sum(p.numel() for p in model.parameters()):,}', flush=True)

    # 全部用 AdamW 解耦 wd; 动态条件每步改 param_group['weight_decay'] (公平对比)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    tr_loader = DataLoader(DS(tr_data, seq_len), batch_size=batch, shuffle=True,
                           num_workers=0, collate_fn=DS.collate, drop_last=True)
    val_loader = DataLoader(DS(val_data, seq_len), batch_size=batch, shuffle=False,
                            num_workers=0, collate_fn=DS.collate, drop_last=False)

    # 初始 V0 (随机基线)
    model.eval()
    with torch.no_grad():
        xb, yb = next(iter(tr_loader))
        V0 = F.cross_entropy(model(xb.to(DEV))[0].reshape(-1, VOCAB), yb.to(DEV).reshape(-1), ignore_index=0).item()

    csv_path = os.path.join(OUT, f'log_clrrl_{cond}.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['step','train_loss','val_loss','val_ppl','gap','lambda'])

    def val_stats():
        model.eval(); t = 0.0; c = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEV), y.to(DEV)
                lo, _ = model(x)
                l = F.cross_entropy(lo.reshape(-1, VOCAB), y.reshape(-1), ignore_index=0, reduction='sum')
                t += float(l); c += int((y != 0).sum())
        model.train(); return t / c

    val_ema = V0; t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        x, y = next(iter(tr_loader)); x, y = x.to(DEV), y.to(DEV)
        lo, _ = model(x)
        V = F.cross_entropy(lo.reshape(-1, VOCAB), y.reshape(-1), ignore_index=0)

        # 决定本步 wd (解耦, 直接设 AdamW 的 weight_decay)
        if cond == 'fixed':
            lam = wd_max
        elif cond == 'cosine':
            lam = wd_max * 0.5 * (1 + math.cos(math.pi * step / steps))
        else:  # clr
            if step % 30 == 0:
                vl = val_stats()
                val_ema = 0.8 * val_ema + 0.2 * vl
            gap = max(0.0, val_ema - float(V.detach()))     # 过拟合 gap
            compress = float(torch.sigmoid(torch.tensor((V0 * vfrac - float(V.detach())) / tau)))
            gap_sig = float(torch.sigmoid(torch.tensor((gap - gap_margin) / gap_tau)))
            lam = clr_lam * compress * gap_sig
        for g in opt.param_groups: g['weight_decay'] = lam

        loss = V
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            vl = val_stats(); vp = math.exp(vl)
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([step, float(V.detach()), vl, vp, vl - float(V.detach()), lam])
            if step % 200 == 0 or step <= 100:
                print(f'  [{cond}] s{step} tr={float(V):.3f} val={vl:.3f} ppl={vp:.1f} '
                      f'gap={vl-float(V):+.3f} λ={lam:.3f}', flush=True)
    vl = val_stats()
    print(f'[{cond}] DONE val_ppl={math.exp(vl):.2f} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', default='clr', choices=['fixed','cosine','clr'])
    ap.add_argument('--steps', type=int, default=1200)
    ap.add_argument('--seq', type=int, default=256)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--n_val', type=int, default=2000)
    args = ap.parse_args()
    run(args.cond, args.steps, args.seq, args.batch, args.lr, args.n_val)
