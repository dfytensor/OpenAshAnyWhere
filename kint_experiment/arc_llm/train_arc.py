"""ARC-LLM 训练: Modular Addition grokking, 4 条件对照.
  vanilla    : 标准 FFN + 固定 wd           (基线 grokking)
  arc_no_sgr : 标准 FFN + CLR-on-L2          (论文闭环 v4, 隔离 SGR)
  arc_no_clr : SGR + 固定 wd (门控自由)      (隔离 CLR)
  arc_full   : SGR + CLR-on-gate             (完整 ARC-LLM)

CLR: λ(t) = λ_max · σ((V0·0.3−V)/τ) · (1−gen_ema)
  arc_full/no_sgr 用闭环; 压缩对象: arc_full=门控均值 s̄ (压有效秩),
  arc_no_sgr=权重范数² (压 L2, 论文版). gen_ema 跟踪 test acc.
"""
import torch, torch.nn.functional as F, math, os, csv, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_llm import ARC_LLM, model_stable_rank

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = os.path.dirname(os.path.abspath(__file__))


def make_data(p=113, train_frac=0.3, seed=0):
    a = torch.arange(p)
    A, B = torch.meshgrid(a, a, indexing='ij')
    pairs = torch.stack([A.flatten(), B.flatten()], 1)
    Y = (pairs[:, 0] + pairs[:, 1]) % p
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(p * p, generator=g)
    ntr = int(train_frac * p * p); tr, te = perm[:ntr], perm[ntr:]
    EQ = p
    def seq(idx):
        n = idx.numel()
        return torch.cat([pairs[idx], torch.full((n, 1), EQ, dtype=torch.long)], 1)
    return seq(tr).to(DEV), Y[tr].to(DEV), seq(te).to(DEV), Y[te].to(DEV), p


def weight_norm_sq(m):
    s = 0.0
    for pp in m.parameters():
        s += float((pp.detach() ** 2).sum())
    return s


def run(cond, p, steps, lr, wd, lam_hi, vfrac, tau, seed, use_sgr, use_clr,
        d=128, heads=4, layers=3, d_ffn=512, r_low=32, train_frac=0.3):
    torch.manual_seed(seed)
    VOCAB = p + 2
    Xtr, Ytr, Xte, Yte, _ = make_data(p, train_frac, seed)
    model = ARC_LLM(VOCAB, d, heads, layers, d_ffn, r_low, use_sgr=use_sgr, max_len=16).to(DEV)
    n = sum(pp.numel() for pp in model.parameters())
    print(f'[{cond}] params={n:,} use_sgr={use_sgr} use_clr={use_clr} wd={wd} lam_hi={lam_hi}', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=(0.0 if use_clr else wd))

    model.eval()
    with torch.no_grad():
        lo, _ = model(Xtr[:256]); V0 = F.cross_entropy(lo[:, -1, :], Ytr[:256]).item()

    csv_path = os.path.join(OUT, f'log_{cond}.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['step','V','C_rank','K','acc_train','acc_test','lambda','gate_mean'])

    gen_ema = 0.0
    def quick_test():
        model.eval(); bs = 2048; c = 0; t = 0
        with torch.no_grad():
            for i in range(0, Xte.size(0), bs):
                lo, _ = model(Xte[i:i+bs]); c += int((lo[:, -1].argmax(-1) == Yte[i:i+bs]).sum()); t += lo.size(0)
        model.train(); return c / t

    for step in range(1, steps + 1):
        model.train()
        logits, gates = model(Xtr)
        V = F.cross_entropy(logits[:, -1, :], Ytr)
        # 闭环 λ
        if use_clr:
            if step % 30 == 0:
                gen_ema = 0.8 * gen_ema + 0.2 * quick_test()
            compress = torch.sigmoid(torch.tensor((V0 * vfrac - float(V.detach())) / tau))
            lam = lam_hi * float(compress) * (1.0 - gen_ema)
        else:
            lam = 0.0
        # 压缩项: CLR 一律作用于权重 (L2, 论文版). SGR 门控自由路由(不直接惩罚),
        # 只改变架构——隔离 SGR 贡献靠 arc_full vs arc_no_sgr 对比.
        if use_clr:
            loss = V + lam * sum((pp ** 2).sum() for pp in model.parameters())
        else:
            loss = V                                        # 固定 wd 已在 AdamW
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 100 == 0 or step <= 10:
            model.eval()
            with torch.no_grad():
                lt, _ = model(Xtr); le, _ = model(Xte)
                act = (lt[:, -1].argmax(-1) == Ytr).float().mean().item()
                ace = (le[:, -1].argmax(-1) == Yte).float().mean().item()
            C_rank = model_stable_rank(model)
            gm = sum(gates) / max(len(gates), 1) if gates else float('nan')
            K = C_rank * float(V.detach())
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([step, float(V.detach()), C_rank, K, act, ace, lam, gm])
            if step % 1000 == 0 or step <= 20:
                print(f'  [{cond}] s{step} V={float(V):.3f} srank={C_rank:.1f} '
                      f'tr={act:.3f} te={ace:.3f} λ={lam:.2e} ḡ={gm:.3f}', flush=True)
    print(f'[{cond}] DONE final_test_acc={ace:.3f}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', default='arc_full')
    ap.add_argument('--p', type=int, default=113)
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=0.1)
    ap.add_argument('--lam_hi', type=float, default=3.0)    # arc_full 门控惩罚峰值
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    cfg = dict(p=args.p, steps=args.steps, lr=args.lr, wd=args.wd,
               lam_hi=args.lam_hi, vfrac=0.3, tau=0.05, seed=args.seed)
    if args.cond == 'vanilla':
        run(args.cond, use_sgr=False, use_clr=False, **cfg)
    elif args.cond == 'arc_no_sgr':
        run(args.cond, use_sgr=False, use_clr=True, lam_hi=3e-4, **{k:v for k,v in cfg.items() if k!='lam_hi'})
    elif args.cond == 'arc_no_clr':
        run(args.cond, use_sgr=True, use_clr=False, **cfg)
    else:  # arc_full: SGR 自由路由 + CLR-on-L2 (隔离 SGR 贡献)
        run(args.cond, use_sgr=True, use_clr=True, lam_hi=3e-4, **{k:v for k,v in cfg.items() if k!='lam_hi'})
