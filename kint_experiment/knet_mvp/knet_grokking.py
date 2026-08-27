"""K-Net MVP 验证 (链接 E 节): DCU + Governor + 存在性惩罚, 在 Modular Addition(Grokking) 上跑.
忠实实现 E.2 伪代码, 检验 E.3 四条可证伪预言.

条件:
  A1 standard_wd0   : 普通 MLP, wd=0     -> 预期: 记忆, 不 grokking, K 崩溃
  A2 standard_wd1e3 : 普通 MLP, wd=1e-3  -> 预期: grokking (已知结果, 正对照)
  B  knet_wd0       : K-Net(Governor+存在性惩罚), wd=0 -> 预期: grokking + K 平台 (核心)
  C  knet_no_gov    : K-Net 但 sigma 固定 0, penalty gamma=0 -> 预期: = A1, 过拟合
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, os, csv, math

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = os.path.dirname(os.path.abspath(__file__))


# ============ Modular Addition 数据 (Grokking 经典战场) ============
def make_data(p=97, train_frac=0.4, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(p * p, generator=g)
    a = (idx // p).long(); b = (idx % p).long()
    y = (a + b) % p
    n_tr = int(train_frac * p * p)
    return ((a[:n_tr], b[:n_tr], y[:n_tr]), (a[n_tr:], b[n_tr:], y[n_tr:]), p)


# ============ DCU (Dynamic Complexity Unit) ============
class DCU(nn.Module):
    """权重 = W_mu + noise*s;  s=softplus(alpha+sigma_gov);  C 贡献 = Σ log(1+s²).
    alpha 负初始化 => 初始噪声小, K-Net 起步≈标准 MLP (能学); Governor 危机时抬 sigma."""
    def __init__(self, d_in, d_out, alpha_init=-2.5):
        super().__init__()
        self.W_mu = nn.Parameter(torch.randn(d_in, d_out) * 0.02)
        self.alpha = nn.Parameter(torch.full((d_out,), float(alpha_init)))  # 负初始化: 小噪声
    def forward(self, x, sigma_gov):
        s = F.softplus(self.alpha + sigma_gov)             # (d_out,)
        if self.training:
            eps = torch.randn_like(self.W_mu)              # (d_in, d_out)
            W = self.W_mu + eps * s                        # per-output 噪声
        else:
            W = self.W_mu
        return x @ W, s


# ============ Governor (守恒调控器, E.2) ============
class Governor(nn.Module):
    def __init__(self, K_int, kappa=0.1):
        super().__init__()
        self.register_buffer('K_int', torch.tensor(float(K_int)))
        self.kappa = kappa
    def forward(self, C, V):
        CV = C * V
        crisis = (CV < self.K_int).float()                 # 跌破生存线
        sigma = self.kappa * (V / (self.K_int + 1e-6) - 1.0) * crisis
        return sigma, crisis


# ============ 三个模型 ============
class StandardMLP(nn.Module):
    """Power et al. 式 2 层 MLP (无 Governor)."""
    def __init__(self, p, dim=128, width=256):
        super().__init__()
        self.em = nn.Embedding(p, dim)
        self.fc1 = nn.Linear(dim * 2, width); self.fc2 = nn.Linear(width, width); self.hd = nn.Linear(width, p)
    def forward(self, a, b):
        x = torch.cat([self.em(a), self.em(b)], dim=-1)
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x))
        return self.hd(x)

class KNet(nn.Module):
    """K-Net: 嵌入 + 2 DCU + 输出 DCU, Governor 调 sigma."""
    def __init__(self, p, dim=128, width=256, K_int=8.0, kappa=0.1, use_governor=True, use_penalty=True):
        super().__init__()
        self.em = nn.Embedding(p, dim)
        self.d1 = DCU(dim*2, width); self.d2 = DCU(width, width); self.d3 = DCU(width, p)
        self.gov = Governor(K_int, kappa)
        self.use_governor = use_governor
        self.use_penalty = use_penalty
    def forward(self, a, b, V_for_gov=None):
        sigma = torch.zeros(1, device=a.device)
        crisis = torch.zeros(1, device=a.device)
        if self.use_governor and V_for_gov is not None and self.training:
            # 用上一步 V 估 C (近似, 见 train 里传 detach 的 V)
            C = self.C_proxy_detached()
            sigma, crisis = self.gov(C, V_for_gov)
        x = torch.cat([self.em(a), self.em(b)], dim=-1)
        x, s1 = self.d1(x, sigma); x = F.relu(x)
        x, s2 = self.d2(x, sigma); x = F.relu(x)
        logits, s3 = self.d3(x, sigma)
        return logits, (s1, s2, s3), sigma, crisis
    def C_proxy_detached(self):
        with torch.no_grad():
            s1 = F.softplus(self.d1.alpha).sum()
            s2 = F.softplus(self.d2.alpha).sum()
            s3 = F.softplus(self.d3.alpha).sum()
            return float(torch.log1p(F.softplus(self.d1.alpha)**2).sum() +
                         torch.log1p(F.softplus(self.d2.alpha)**2).sum() +
                         torch.log1p(F.softplus(self.d3.alpha)**2).sum())


def run_cond(cond, p, dim, width, steps, lr, wd, K_int=None, kappa=0.5,
             beta=0.0, gamma=0.5, frac_K=0.5, seed=0):
    torch.manual_seed(seed)
    (a_tr,b_tr,y_tr),(a_te,b_te,y_te),p = make_data(p=p, seed=seed)
    a_tr,b_tr,y_tr = a_tr.to(DEV),b_tr.to(DEV),y_tr.to(DEV)
    a_te,b_te,y_te = a_te.to(DEV),b_te.to(DEV),y_te.to(DEV)

    if cond.startswith('standard'):
        model = StandardMLP(p, dim, width).to(DEV)
        is_knet = False
    else:
        use_gov = (cond == 'knet_wd0')
        model = KNet(p, dim, width, K_int=1.0, kappa=kappa,
                     use_governor=use_gov, use_penalty=use_gov).to(DEV)
        is_knet = True
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # 自适应 K_int = frac_K * 初始(C*V), 使存在性惩罚在过拟合后期(V 下降)才激活
    if is_knet:
        with torch.no_grad():
            model.eval()
            _,_,_,_ = model(a_tr[:64], b_tr[:64])
        C0 = model.C_proxy_detached(); V0 = math.log(p)
        K_int_eff = (K_int if K_int is not None else frac_K * C0 * V0)
        model.gov.K_int.fill_(K_int_eff)
        print(f'  [{cond}] C0={C0:.3f} V0={V0:.3f} -> K_int(target)={K_int_eff:.3f}', flush=True)

    csv_path = os.path.join(OUT, f'log_{cond}.csv')
    with open(csv_path,'w',newline='') as f:
        csv.writer(f).writerow(['step','V_train','C','K_est','acc_train','acc_test','sigma_mean','crisis'])

    V_prev = torch.tensor(math.log(p), device=DEV)
    for step in range(1, steps+1):
        model.train()
        if is_knet:
            logits, ss, sigma, crisis = model(a_tr, b_tr, V_for_gov=V_prev.detach())
            V = F.cross_entropy(logits, y_tr)
            C = (torch.log1p(ss[0]**2).sum() + torch.log1p(ss[1]**2).sum() + torch.log1p(ss[2]**2).sum())
            loss = V + beta*C
            if model.use_penalty:
                loss = loss + gamma * F.relu(model.gov.K_int - C*V)   # 存在性惩罚 (E.2)
            sigma_mean = float(sigma.mean().detach()); cris = float(crisis.mean().detach())
            V_prev = V.detach()
            C_f = float(C.detach())
        else:
            logits = model(a_tr, b_tr)
            V = F.cross_entropy(logits, y_tr)
            C = 0.0
            for pp in model.parameters():
                C = C + (pp.detach()**2).sum()
            loss = V
            C_f = float(C); sigma_mean=0.0; cris=0.0
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 100 == 0 or step <= 20:
            model.eval()
            with torch.no_grad():
                if is_knet:
                    lt,_,_,_ = model(a_tr,b_tr); le,_,_,_ = model(a_te,b_te)
                else:
                    lt = model(a_tr,b_tr); le = model(a_te,b_te)
                act = (lt.argmax(-1)==y_tr).float().mean().item()
                ace = (le.argmax(-1)==y_te).float().mean().item()
            K = C_f * float(V.detach())
            with open(csv_path,'a',newline='') as f:
                csv.writer(f).writerow([step, float(V.detach()), C_f, K, act, ace, sigma_mean, cris])
            if step % 1000 == 0:
                print(f'  [{cond}] s{step} V={float(V):.3f} C={C_f:.2f} K={K:.2f} '
                      f'tr={act:.3f} te={ace:.3f} sig={sigma_mean:.3f} cris={cris:.2f}', flush=True)
    print(f'[{cond}] DONE final_test_acc={ace:.3f}', flush=True)
    return ace


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--p', type=int, default=59)
    ap.add_argument('--dim', type=int, default=64)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--conds', default='standard_wd0,standard_wd1e2,knet_wd0,knet_no_gov')
    args = ap.parse_args()
    cfg = dict(p=args.p, dim=args.dim, width=args.width, steps=args.steps, lr=args.lr)
    for c in args.conds.split(','):
        wd = 1e-2 if c == 'standard_wd1e2' else 0.0
        run_cond(c, **cfg, wd=wd)
