"""ARC-LLM 的 LLM 指标评测: 参数对齐 vanilla vs ARC(SGR), 报 val ppl + next-token acc.
真实 LM 任务 (minimind pretrain, open_ash_voc, VOCAB=23005), seq=512.
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


def run(cond, steps, seq_len, batch, lr, wd, n_val, d_ffn=1024, seed=0):
    torch.manual_seed(seed)
    print(f'Loading cache...', flush=True)
    data = torch.load(CACHE, weights_only=False)
    val_data, tr_data = data[:n_val], data[n_val:]
    use_sgr = (cond.startswith('arc'))
    model = ARC_LLM(VOCAB, d=256, n_heads=8, n_layers=4, d_ffn=d_ffn, r_low=64,
                    use_sgr=use_sgr, max_len=seq_len+8).to(DEV)
    n = sum(p.numel() for p in model.parameters())
    print(f'[{cond}] params={n:,} use_sgr={use_sgr}', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    tr_loader = DataLoader(DS(tr_data, seq_len), batch_size=batch, shuffle=True,
                           num_workers=0, collate_fn=DS.collate, drop_last=True)
    val_loader = DataLoader(DS(val_data, seq_len), batch_size=batch, shuffle=False,
                            num_workers=0, collate_fn=DS.collate, drop_last=False)

    csv_path = os.path.join(OUT, f'log_lmeval_{cond}.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['step','train_loss','val_loss','val_ppl','val_top1','gate_mean'])

    def val_metrics():
        model.eval(); tot = 0.0; c = 0; correct = 0; nt = 0; gs = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEV), y.to(DEV)
                lo, g = model(x)
                l = F.cross_entropy(lo.reshape(-1, VOCAB), y.reshape(-1), ignore_index=0, reduction='sum')
                tot += float(l); c += int((y != 0).sum())
                pred = lo.reshape(-1, VOCAB).argmax(-1)
                yt = y.reshape(-1); m = yt != 0
                correct += int((pred[m] == yt[m]).sum()); nt += int(m.sum())
                if g: gs.append(sum(g)/len(g))
        model.train()
        return tot/c, math.exp(tot/c), correct/max(nt,1), (sum(gs)/len(gs) if gs else float('nan'))

    t0 = time.time()
    for step in range(1, steps+1):
        model.train()
        x, y = next(iter(tr_loader)); x, y = x.to(DEV), y.to(DEV)
        lo, _ = model(x)
        loss = F.cross_entropy(lo.reshape(-1, VOCAB), y.reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            vl, vp, vacc, gm = val_metrics()
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([step, float(loss.detach()), vl, vp, vacc, gm])
            if step % 500 == 0 or step <= 200:
                print(f'  [{cond}] s{step} tr={float(loss):.3f} val_ppl={vp:.2f} top1={vacc:.3f} ḡ={gm:.3f}', flush=True)
    vl, vp, vacc, gm = val_metrics()
    print(f'[{cond}] DONE val_ppl={vp:.2f} top1={vacc:.4f} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', default='arc')
    ap.add_argument('--d_ffn', type=int, default=1024)
    ap.add_argument('--steps', type=int, default=2500)
    ap.add_argument('--seq', type=int, default=512)
    ap.add_argument('--batch', type=int, default=24)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=0.1)
    ap.add_argument('--n_val', type=int, default=3000)
    args = ap.parse_args()
    run(args.cond, args.steps, args.seq, args.batch, args.lr, args.wd, args.n_val, d_ffn=args.d_ffn)
