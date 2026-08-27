"""K_int 实验: 从零训练 FRSMASH, 逐步记录 C(权重范数平方) 与 V(损失), 检验 C*V>=K_int.

用法:
  baseline:  python train_kint.py --cond baseline --max_steps 1500
  heated:    python train_kint.py --cond heated   --max_steps 1500
"""
import torch, torch.nn.functional as F, math, time, os, argparse, csv, sys
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
sys.path.insert(0, r'F:\rwkv\frsmash_v36')
from frsmash_v36 import FRSMASHv36

VOCAB = 23005
CACHE = r'F:\OpenASH2605\train_60m\cache\pt_cache_openash_512_openash.pt'
DEV = 'cuda'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


class CachedDS(Dataset):
    def __init__(self, seq_len, max_lines=None):
        print(f'Loading cache {CACHE}...', flush=True)
        self.data = torch.load(CACHE, weights_only=False)
        if max_lines: self.data = self.data[:max_lines]
        self.seq_len = seq_len
        print(f'  {len(self.data)} samples', flush=True)
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i][:self.seq_len + 1]
    @staticmethod
    def collate(items):
        p = pad_sequence(items, batch_first=True, padding_value=0)
        return p[:, :-1], p[:, 1:]


@torch.no_grad()
def weight_norm_sq(model):
    """C 代理: 全体参数平方 L2 范数 = Σ‖p‖² (Weight Norm²)."""
    s = 0.0
    for p in model.parameters():
        s += float((p.detach().float() ** 2).sum())
    return s


def safe_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    if os.path.exists(path): os.remove(path)
    os.rename(tmp, path)


def run(cond, max_steps, hidden, heads, layers, seq_len, batch, lr, wd, max_lines, seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    import random; random.seed(seed); import numpy as np; np.random.seed(seed)
    ds = CachedDS(seq_len, max_lines)
    model = FRSMASHv36(VOCAB, hidden, heads, layers, n_slots=4).to(DEV)
    n = sum(p.numel() for p in model.parameters())
    print(f'[{cond}] params={n:,} wd={wd} lr={lr} bs={batch} seq={seq_len}', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler()
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0,
                        collate_fn=CachedDS.collate, drop_last=True, pin_memory=True)
    spe = len(loader)

    csv_path = os.path.join(OUT_DIR, f'log_{cond}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['step', 'loss_V', 'weight_norm_sq_C', 'K_est', 'lr', 'cond'])

    gs = 0; t0 = time.time()
    model.train()
    total = max_steps
    while gs < total:
        for x, t in loader:
            if gs >= total: break
            x = x.clamp(0, VOCAB - 1).to(DEV); t = t.clamp(0, VOCAB - 1).to(DEV)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                o = model(x)
                loss = F.cross_entropy(o.reshape(-1, VOCAB), t.reshape(-1), ignore_index=0)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            cur_lr = lr * (0.1 + 0.45 * (1 + math.cos(math.pi * gs / total)))
            for pg in opt.param_groups: pg['lr'] = cur_lr
            gs += 1
            V = loss.item()
            if gs % 10 == 0 or gs <= 5:
                C = weight_norm_sq(model)
                K = C * V
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([gs, V, C, K, cur_lr, cond])
                if gs % 50 == 0:
                    el = time.time() - t0
                    tok = gs * batch * seq_len / el
                    print(f'  [{cond}] s{gs}/{total} V={V:.4f} C={C:.1f} K={K:.1f} lr={cur_lr:.2e} {tok:.0f}tok/s', flush=True)

    ckpt = os.path.join(OUT_DIR, f'kint_{cond}_final.pth')
    safe_save({'model': model.state_dict(), 'step': gs,
               'config': dict(hidden=hidden, heads=heads, layers=layers)}, ckpt)
    print(f'[{cond}] DONE -> {ckpt} ({time.time()-t0:.0f}s)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', default='baseline')          # 自由标签 (用作 csv/ckpt 名)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--wd', type=float, default=0.0)        # >0 时直接用, =0 时按 cond 推断
    ap.add_argument('--max_steps', type=int, default=1500)
    ap.add_argument('--hidden', type=int, default=432)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--seq', type=int, default=512)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--max_lines', type=int, default=0)
    args = ap.parse_args()
    wd = args.wd if args.wd > 0 else (0.01 if args.cond == 'baseline' else 0.10)
    run(args.cond, args.max_steps, args.hidden, args.heads, args.layers,
        args.seq, args.batch, args.lr, wd, args.max_lines or None, seed=args.seed)


if __name__ == '__main__':
    main()
