#!/usr/bin/env python3
"""v15: 纵向 — 收敛性攻击. 教绑定块'到达'不动点.
核心: L = CE + λ·(末步相对残差) — 模型必须同时 (a)预测好 (b)在第K步到达(f输出≈0)
快验: 3000 步, 对照 v5 (同架构同步数, 无收敛损失: 探针+0.385, 残差0.12-0.17, 从不早退)
看: 残差是否→小, 探针(残差-NLL相关)是否变强, k_used 是否分化"""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v5_30m import StdBlock, TiedBlock, get_batch, V, D, H, DEV, SEQ, PT_CACHE

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "deqlm15b.log")
OUT = os.path.join(HERE, "deqlm15b_results.json")
CKPT = os.path.join(HERE, "deqlm15b_quick.pth")
STEPS = 3000
LAMBDA_CONV = 0.01
TOL = 5e-3           # 早退容差 (比 v10 的观测残差 0.12-0.17 低一个量级)


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


class ConvDEQLM(nn.Module):
    """2 std + tied×30 + 2 std, forward 返回 (logits, 末步相对残差 per-sample)"""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Embedding(SEQ + 1, D)
        self.ln_emb = nn.LayerNorm(D)
        self.bottom = nn.ModuleList([StdBlock() for _ in range(2)])
        self.deq = TiedBlock(kmax=30)
        self.top = nn.ModuleList([StdBlock() for _ in range(2)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, V)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def bottom_h(self, ids):
        B, L = ids.shape
        h = self.emb(ids) + self.pos(torch.arange(L, device=ids.device)).unsqueeze(0)
        h = self.ln_emb(h)
        for layer in self.bottom:
            h = layer(h)
        return h

    def forward_full(self, ids):
        """返回 logits, per-sample 末步相对残差 (B,), 逐样本残差轨迹 (B, K)."""
        h = self.bottom_h(ids)
        alpha = self.deq.alpha()
        z = h
        znorm = h.norm(dim=-1).clamp(min=1e-6)          # (B, L)
        res_traj = []
        for k in range(30):
            z_new = z + alpha * self.deq._block_fn(z)
            rel = (z_new - z).norm(dim=-1) / znorm      # (B, L)
            res_traj.append(rel.mean(dim=1))            # (B,)
            z = z_new
        res_traj = torch.stack(res_traj, 1)             # (B, 30)
        hh = z
        for layer in self.top:
            hh = layer(hh)
        return self.head(self.ln_f(hh)), res_traj[:, -1], res_traj


@torch.no_grad()
def probe_eval(m, val, n=150, seed=7000):
    """残差探针 + tol 早退分布."""
    import numpy as np
    m.eval()
    residuals, nlls, k_used_all = [], [], []
    for i in range(n):
        x, y = get_batch(val, bs=1, device=DEV, rng=random.Random(seed + i))
        logits, res_final, traj = m.forward_full(x)
        mask = y != 0
        if mask.sum() < 8:
            continue
        l = F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1), reduction="none").reshape(y.shape)
        nlls.append(l[mask].mean().item())
        residuals.append(res_final.item())
        # tol 早退: 首次 rel_res < TOL 的 k
        tr = traj[0].cpu().numpy()
        ku = 30
        for k, r in enumerate(tr):
            if r < TOL:
                ku = k + 1
                break
        k_used_all.append(ku)
    r_, n_, k_ = np.array(residuals), np.array(nlls), np.array(k_used_all)
    corr_res = float(np.corrcoef(r_, n_)[0, 1]) if r_.std() > 1e-9 else 0.0
    corr_k = float(np.corrcoef(k_, n_)[0, 1]) if k_.std() > 1e-9 else 0.0
    m.train()
    return dict(corr_residual=round(corr_res, 3), corr_k_used=round(corr_k, 3),
                res_mean=round(float(r_.mean()), 5), res_std=round(float(r_.std()), 5),
                k_frac_early=round(float((k_ < 30).mean()), 3),
                k_mean=round(float(k_.mean()), 2))


def main():
    torch.manual_seed(0)
    random.seed(0)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    train, val = seqs[:int(0.95 * len(seqs))], seqs[int(0.95 * len(seqs)):]
    m = ConvDEQLM().to(DEV)
    log("v15 快验: 3000 步, λ_conv=%.2f, tol=%.0e (对照 v5: corr=+0.385, res 0.12-0.17)" % (LAMBDA_CONV, TOL))
    opt = optim.AdamW(m.parameters(), lr=4e-4, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=4e-5)

    t0 = time.time()
    rng = random.Random(42)
    for st in range(STEPS):
        m.train()
        x, y = get_batch(train, bs=32, device=DEV, rng=rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, res_final, _ = m.forward_full(x)
        ce = F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1), ignore_index=0)
        loss = ce + LAMBDA_CONV * res_final.mean()
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if st % 250 == 0:
            log("st%d ce=%.4f res=%.5f (%.0fms/步)" % (st, ce.item(), res_final.mean().item(), (time.time()-t0)/(st+1)*1000))

    torch.save(m.state_dict(), CKPT)
    log("\n=== 探针评测 (对照 v5: corr_res=+0.385) ===")
    r = probe_eval(m, val)
    log(json.dumps(r, ensure_ascii=False))
    # 快速 NLL 对照 (v5 3000步: 4.251 全序列)
    m.eval()
    tot, tok = 0.0, 0
    with torch.no_grad():
        for i in range(20):
            x, y = get_batch(val, bs=32, device=DEV, rng=random.Random(999 + i))
            lo, _, _ = m.forward_full(x)
            l = F.cross_entropy(lo.float().reshape(-1, V), y.reshape(-1), reduction="sum")
            tot += l.item(); tok += y.numel()
    log("全序列 NLL = %.4f (v5 对照 4.251)" % (tot / tok))
    r["full_nll"] = round(tot / tok, 4)
    json.dump(r, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
