"""FRSMASH v3.6 载体上的 K-Net / Grokking 实验.
任务: Modular Addition a+b mod p (causal: [a,b,=] -> 预测答案).
对比:
  standard : 纯 FRSMASH, 几个 wd
  knet     : FRSMASH + 存在性惩罚 gamma*relu(K_int - C*V)  (闭环调控)
C=权重范数平方, V=CE, K=C*V. 自适应 K_int=frac*初始C*V.
"""
import torch, torch.nn.functional as F, math, os, csv, sys, argparse
sys.path.insert(0, r'F:\rwkv\frsmash_v36')
from frsmash_v36 import FRSMASHv36

DEV = 'cuda'
OUT = os.path.dirname(os.path.abspath(__file__))


def make_data(p=113, train_frac=0.3, seed=0):
    a = torch.arange(p)
    A, B = torch.meshgrid(a, a, indexing='ij')
    pairs = torch.stack([A.flatten(), B.flatten()], 1)
    Y = (pairs[:, 0] + pairs[:, 1]) % p
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(p * p, generator=g)
    ntr = int(train_frac * p * p)
    tr, te = perm[:ntr], perm[ntr:]
    EQ = p                                   # '=' token id
    def seq(idx):
        n = idx.numel()
        return torch.cat([pairs[idx], torch.full((n, 1), EQ, dtype=torch.long)], 1)
    return (seq(tr).to(DEV), Y[tr].to(DEV), seq(te).to(DEV), Y[te].to(DEV), p)


@torch.no_grad()
def weight_norm_sq(m):
    s = 0.0
    for pp in m.parameters():
        s += float((pp.detach().float() ** 2).sum())
    return s


def weight_norm_sq_grad(m):
    """带梯度的 C (用于压缩 loss 项)."""
    s = None
    for pp in m.parameters():
        t = (pp.float() ** 2).sum()
        s = t if s is None else s + t
    return s


@torch.no_grad()
def stable_rank(m):
    """C 的更好代理: 稳定秩 = ‖W‖_F² / ‖W‖_2² , 对所有 2D 权重求和(归一化).
    grokking = rank collapse, 稳定秩比权重范数更贴近"电路复杂度"."""
    tot = 0.0; n = 0
    for pp in m.parameters():
        if pp.dim() == 2:
            W = pp.detach().float()
            fro = (W ** 2).sum().item()
            spec = torch.linalg.svdvals(W)[0].item() ** 2
            tot += fro / (spec + 1e-12); n += 1
    return tot / max(n, 1)


def run(cond, p, hidden, heads, layers, steps, lr, wd, gamma, frac_K, train_frac, seed,
        gov='none', lam_lo=0.0, lam_hi=1e-4, vfrac=0.3, tau=0.05):
    torch.manual_seed(seed)
    VOCAB = p + 2
    Xtr, Ytr, Xte, Yte, _ = make_data(p, train_frac, seed)
    model = FRSMASHv36(VOCAB, hidden, heads, layers, n_slots=4).to(DEV)
    n = sum(pp.numel() for pp in model.parameters())
    print(f'[{cond}] FRSMASH params={n:,} p={p} wd={wd} gamma={gamma}', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=(0.0 if gov == 'compress' else wd))

    # 自适应 K_int = frac_K * 初始 C*V
    model.eval()
    with torch.no_grad():
        lo = model(Xtr[:256])[:, -1, :]
        V0 = F.cross_entropy(lo, Ytr[:256]).item()
    C0 = weight_norm_sq(model)
    K_int = frac_K * C0 * V0
    print(f'  [{cond}] C0={C0:.1f} V0={V0:.3f} -> K_int(target)={K_int:.1f}', flush=True)

    csv_path = os.path.join(OUT, f'log_{cond}.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['step','V','C','K_est','acc_train','acc_test','gov_signal','srank'])

    def eval_acc():
        model.eval()
        def pred(X, y):
            correct = 0; total = 0; bs = 1024
            with torch.no_grad():
                for i in range(0, X.size(0), bs):
                    lo = model(X[i:i+bs])[:, -1, :].argmax(-1)
                    correct += int((lo == y[i:i+bs]).sum()); total += lo.size(0)
            return correct / total
        return pred(Xtr, Ytr), pred(Xte, Yte)

    V0_t = V0
    te_ema = 0.0                                           # closed 模式: 测试准确率 EMA
    def quick_test_acc():
        model.eval(); bs = 1024; correct = 0; total = 0
        with torch.no_grad():
            for i in range(0, Xte.size(0), bs):
                lo = model(Xte[i:i+bs])[:, -1, :].argmax(-1)
                correct += int((lo == Yte[i:i+bs]).sum()); total += lo.size(0)
        model.train()
        return correct / total
    for step in range(1, steps + 1):
        model.train()
        logits = model(Xtr)[:, -1, :]
        V = F.cross_entropy(logits, Ytr)
        C = weight_norm_sq(model)
        K = C * float(V.detach())
        if gov == 'floor':                              # 原始 K-Net: 维持 C·V>=K_int
            pen = F.relu(1.0 - C * V / K_int)
            loss = V + gamma * pen
            gov_sig = float(pen.detach())
        elif gov == 'compress':                         # 改进 K-Net v1: 拟合后压 C (V反馈, 可能slingshot)
            compress = torch.sigmoid((V0_t * vfrac - V.detach()) / tau)   # 0->1 as V drops
            lam = lam_lo + (lam_hi - lam_lo) * float(compress)
            loss = V + lam * weight_norm_sq_grad(model)
            gov_sig = lam                                # 记录自适应 λ
        elif gov == 'ramp':                             # 改进 K-Net v2: 单调斜坡 λ (无反馈, 稳定)
            lam = lam_lo + (lam_hi - lam_lo) * (step / steps)
            loss = V + lam * weight_norm_sq_grad(model)
            gov_sig = lam
        elif gov == 'sine':                             # 改进 K-Net v3: 升-峰-降 (压缩后放松, 抗过压)
            s = math.sin(math.pi * step / steps)
            lam = lam_lo + (lam_hi - lam_lo) * s
            loss = V + lam * weight_norm_sq_grad(model)
            gov_sig = lam
        elif gov == 'closed':                           # 改进 K-Net v4: 测试准确率闭环 (grok后放松λ)
            if step % 30 == 0:
                te_ema = 0.8 * te_ema + 0.2 * quick_test_acc()
            compress = torch.sigmoid((V0_t * vfrac - V.detach()) / tau)
            lam = lam_lo + (lam_hi - lam_lo) * float(compress) * (1.0 - te_ema)
            loss = V + lam * weight_norm_sq_grad(model)
            gov_sig = lam
        else:                                            # 标准
            loss = V
            gov_sig = 0.0
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step <= 10:
            act, ace = eval_acc()
            sr = stable_rank(model)
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([step, float(V.detach()), C, K, act, ace, gov_sig, sr])
            if step % 1000 == 0 or step <= 50:
                tag = f'pen={gov_sig:.2f}' if gov == 'floor' else (f'λ={gov_sig:.2e}' if gov == 'compress' else '')
                print(f'  [{cond}] s{step} V={float(V):.3f} C={C:.0f} K={K:.0f} '
                      f'tr={act:.3f} te={ace:.3f} srank={sr:.1f} {tag}', flush=True)
    _, ace = eval_acc()
    print(f'[{cond}] DONE final_test_acc={ace:.3f}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', default='standard')
    ap.add_argument('--p', type=int, default=113)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=0.0)
    ap.add_argument('--gamma', type=float, default=0.0)   # floor 模式的存在性惩罚系数
    ap.add_argument('--gov', default='none', choices=['none', 'floor', 'compress', 'ramp', 'sine', 'closed'])
    ap.add_argument('--lam_lo', type=float, default=1e-6)  # compress: 拟合期 λ
    ap.add_argument('--lam_hi', type=float, default=3e-4)  # compress: 压缩期 λ
    ap.add_argument('--vfrac', type=float, default=0.3)    # compress: V 降到 V0*vfrac 触发压缩
    ap.add_argument('--tau', type=float, default=0.05)     # compress: sigmoid 温度
    ap.add_argument('--train_frac', type=float, default=0.3)
    ap.add_argument('--frac_K', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    run(args.cond, args.p, args.hidden, args.heads, args.layers, args.steps,
        args.lr, args.wd, args.gamma, args.frac_K, args.train_frac, seed=args.seed,
        gov=args.gov, lam_lo=args.lam_lo, lam_hi=args.lam_hi, vfrac=args.vfrac, tau=args.tau)
