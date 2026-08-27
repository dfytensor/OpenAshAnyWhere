"""ARC-LLM 真实 LM 测试: SGR 门控是否按 token 惊讶度路由?
任务: minimind pretrain (open_ash_voc 分词缓存, VOCAB=23005).
对照: vanilla transformer (无SGR) vs ARC-LLM (SGR). 同 wd, 比 val ppl.
关键测量: 训练后, 门控 g_t 与每 token 损失(惊讶度)的 Pearson 相关.
  - modular addition 上 r≈0 (门控恒定). 真实文本若 r>0 => SGR 真的在按惊讶路由.
"""
import torch, torch.nn.functional as F, math, os, sys, csv, argparse, numpy as np
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_llm import ARC_LLM

DEV = 'cuda'
OUT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get('KINT_CACHE', r'F:\OpenASH2605\train_60m\cache\pt_cache_openash_512_openash.pt')
VOCAB = 23005


class DS(Dataset):
    def __init__(self, data, seq_len):
        self.data = data; self.seq_len = seq_len
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i][:self.seq_len + 1]
    @staticmethod
    def collate(items):
        from torch.nn.utils.rnn import pad_sequence
        p = pad_sequence(items, batch_first=True, padding_value=0)
        return p[:, :-1], p[:, 1:]


def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = math.sqrt((x**2).sum() * (y**2).sum())
    return float((x*y).sum() / d) if d > 0 else 0.0


def run(cond, use_sgr, steps, seq_len, batch, lr, wd, n_val, seed=0):
    torch.manual_seed(seed)
    print(f'Loading cache {CACHE}...', flush=True)
    data = torch.load(CACHE, weights_only=False)
    val_data = data[:n_val]; tr_data = data[n_val:]
    print(f'  train={len(tr_data)} val={len(val_data)}', flush=True)

    model = ARC_LLM(VOCAB, d=256, n_heads=8, n_layers=4, d_ffn=1024, r_low=64,
                    use_sgr=use_sgr, max_len=seq_len+8).to(DEV)
    n = sum(p.numel() for p in model.parameters())
    print(f'[{cond}] params={n:,} use_sgr={use_sgr} wd={wd}', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    tr_loader = DataLoader(DS(tr_data, seq_len), batch_size=batch, shuffle=True,
                           num_workers=0, collate_fn=DS.collate, drop_last=True)
    val_loader = DataLoader(DS(val_data, seq_len), batch_size=batch, shuffle=False,
                            num_workers=0, collate_fn=DS.collate, drop_last=False)

    csv_path = os.path.join(OUT, f'log_real_{cond}.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['step','train_loss','val_loss','val_ppl','gate_mean'])

    def val_loss_ppl():
        model.eval(); tot = 0.0; cnt = 0; gs = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEV), y.to(DEV)
                lo, g = model(x)
                l = F.cross_entropy(lo.reshape(-1, VOCAB), y.reshape(-1), ignore_index=0, reduction='sum')
                tot += float(l); cnt += int((y != 0).sum())
                if g: gs.append(np.mean(g))
        return tot/cnt, math.exp(tot/cnt), (np.mean(gs) if gs else float('nan'))

    gs = 0; t0 = __import__('time').time()
    for step in range(1, steps+1):
        model.train()
        x, y = next(iter(tr_loader))
        x, y = x.to(DEV), y.to(DEV)
        lo, g = model(x)
        loss = F.cross_entropy(lo.reshape(-1, VOCAB), y.reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            vl, vp, gm = val_loss_ppl()
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([step, float(loss.detach()), vl, vp, gm])
            if step % 200 == 0 or step <= 100:
                print(f'  [{cond}] s{step} tr={float(loss):.3f} val={vl:.3f} ppl={vp:.1f} ḡ={gm:.3f}', flush=True)

    # 关键: gate-惊讶度相关性
    model.eval()
    all_g, all_l = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEV), y.to(DEV)
            lo, _ = model(x)
            if use_sgr:
                gt = model.gate_per_token(x)                 # (B,T)
                # 每 token 损失
                lt = F.cross_entropy(lo.reshape(-1, VOCAB), y.reshape(-1),
                                     ignore_index=0, reduction='none').reshape(x.shape)
                mask = (y != 0) & (lt < 15)                  # 排除 pad 与极端值
                all_g.append(gt[mask].float().cpu().numpy())
                all_l.append(lt[mask].float().cpu().numpy())
    if use_sgr:
        g = np.concatenate(all_g); l = np.concatenate(all_l)
        r = pearson(g, l)
        # 分桶: 高惊讶(top20% loss) vs 低惊讶(bottom20%) 的平均门控
        order = np.argsort(l); hi = g[order[-int(0.2*len(g)):]]; lo_ = g[order[:int(0.2*len(g))]]
        print(f'\n[{cond}] GATE-SURPRISE: Pearson r(g,loss)={r:+.4f}  '
              f'高惊讶token ḡ={hi.mean():.3f}  低惊讶token ḡ={lo_.mean():.3f}  '
             f'(差={hi.mean()-lo_.mean():+.3f})', flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([-1, r, hi.mean(), lo_.mean(), gm])
    else:
        print(f'\n[{cond}] (no SGR, 无门控)', flush=True)
    print(f'[{cond}] DONE val_ppl={vp:.1f} ({__import__("time").time()-t0:.0f}s)', flush=True)
    return vp


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', default='arc')
    ap.add_argument('--steps', type=int, default=1200)
    ap.add_argument('--seq', type=int, default=256)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=0.1)
    ap.add_argument('--n_val', type=int, default=2000)
    args = ap.parse_args()
    use_sgr = (args.cond == 'arc')
    run(args.cond, use_sgr, args.steps, args.seq, args.batch, args.lr, args.wd, args.n_val)
