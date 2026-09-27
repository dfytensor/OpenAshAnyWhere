# -*- coding: utf-8 -*-
"""
二阶自觉 阶段0: 真实任务复现 (scale_route.md 阶段0, 语言建模版)
=================================================================
映射 (prototype.py -> 真实LM):
  κ1 对象级  : 字符级小Transformer(2层,d128) + DEQ块 z*=(1-a)z*+a tanh(W z*+Ux)
  天花板预训练: 先训到plateau, 残差≈不可再降
  内部代理集 : 25条固定短序列 (可被gaming过拟合)
  honest候选 : 大训练批上1步真梯度 + 谱重整
  gaming候选 : 代理集上30步全批Adam过拟合 + W*1.06 (破坏不动点稳定性)
  选择压力   : 内部代理CE  (Goodhart机理)
  B 外部基准 : 时间锚定保留集 (数据末尾5%, 训练时不可见), CE测量
  验收A      : B证书 (Δholdout >= tau + 硬约束: 谱半径<5)
  验收B消融  : 内部代理 Δ>0
五指标: G_ext(A/B相对改善) FA/FR  α_dir(A vs 盲目D)  ρ_eff(深度-误差曲线)  probe(收敛通道)
"""
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(20240920)
np.random.seed(20240920)
DEV = "cuda:0"
D, NH, NL = 128, 4, 2
N_PROXY = 25
WINDOW = 64
VOCAB = 300
T_CYCLES = 60
PROP_EVERY = 4

# ---------------------------------------------------------------- 数据
print("loading corpus ...", flush=True)
train_texts, hold_texts = [], []
with open(r"E:\ragllm\pretrain_t2t_mini.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i < 60000:
            train_texts.append(json.loads(line)["text"])
        elif i >= 95000:
            hold_texts.append(json.loads(line)["text"])
        if i >= 100000:
            break
from collections import Counter
cnt = Counter("".join(train_texts[:5000]))
chars = [c for c, _ in cnt.most_common(VOCAB - 1)]
stoi = {c: i for i, c in enumerate(chars)}
UNK = VOCAB - 1
enc = lambda s: [stoi.get(c, UNK) for c in s]
train_stream = np.array(enc("".join(train_texts))[:4_000_000], dtype=np.int64)
hold_stream = np.array(enc("".join(hold_texts))[:200_000], dtype=np.int64)
print(f"train stream {len(train_stream):,} chars, holdout {len(hold_stream):,} chars, vocab {VOCAB}")

proxy_seqs = np.stack([train_stream[i:i + WINDOW + 1] for i in
                       np.random.default_rng(1).integers(0, 2_000_000 - WINDOW - 1, N_PROXY)])
hold_seqs = np.stack([hold_stream[i:i + WINDOW + 1] for i in
                      np.random.default_rng(2).integers(0, len(hold_stream) - WINDOW - 1, 96)])


def rand_batch(bs=16):
    i = np.random.randint(0, 2_000_000 - WINDOW - 1, bs)
    return np.stack([train_stream[j:j + WINDOW + 1] for j in i])


# ---------------------------------------------------------------- κ1 模型
class DEQBlock(nn.Module):
    def __init__(self, n=D):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n, n) / math.sqrt(n))
        with torch.no_grad():
            self.W /= self.W.norm(dim=1).max()
        self.n = n
        self.alpha, self.Kmax, self.tol = 0.5, 40, 1e-4

    def forward(self, x, return_conv=False, W=None, alpha=None):
        W = self.W if W is None else W
        alpha = self.alpha if alpha is None else alpha
        Z = torch.tanh(x)
        eh = []
        K = self.Kmax
        for k in range(self.Kmax):
            Zn = (1 - alpha) * Z + alpha * torch.tanh(Z @ W.t() + x)
            r = float((Zn - Z).detach().norm() / (Z.detach().norm() + 1e-9))
            eh.append(r)
            Z = Zn
            if r < self.tol:
                K = k + 1
                break
        if return_conv:
            return Z, eh, K
        return Z

    def spectral(self):
        return float(torch.linalg.matrix_norm(self.W, ord=2))

    def eig_radius(self):
        return float(torch.abs(torch.linalg.eigvals(self.W.detach().float().cpu())).max())


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D)
        layer = nn.TransformerEncoderLayer(D, NH, dim_feedforward=256, batch_first=True,
                                           dropout=0.0, activation="gelu")
        self.tr = nn.TransformerEncoder(layer, NL)
        self.deq = DEQBlock()
        self.head = nn.Linear(D, VOCAB)
        self.pos = nn.Parameter(torch.randn(1, WINDOW + 1, D) * 0.02)

    def features(self, ids):
        x = self.emb(ids) + self.pos[:, : ids.shape[1]]
        x = self.tr(x)
        return x.transpose(0, 1)  # (L=WINDOW, B, D)  DEQ 沿序列维迭代

    def forward(self, ids, return_conv=False, W=None):
        z_in = self.features(ids)
        if return_conv:
            z, eh, K = self.deq(z_in, return_conv=True, W=W)
            logits = self.head(z).transpose(0, 1)
            return logits, eh, K
        z = self.deq(z_in, W=W)
        return self.head(z).transpose(0, 1)

    def spectral(self):
        return self.deq.spectral()

    def param_norm(self):
        return float(sum(p.norm() for p in self.parameters()))


def ce_of(model, seqs, W=None):
    ids = torch.from_numpy(seqs[:, :-1]).to(DEV)
    tgt = torch.from_numpy(seqs[:, 1:]).to(DEV)
    logits = model(ids, W=W) if W is not None else model(ids)
    return float(nn.functional.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1)))


def clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


# ---------------------------------------------------------------- 主实验
def run(cond, pretrain_steps=1200, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TinyLM().to(DEV)
    with torch.no_grad():
        er = model.deq.eig_radius()
        if er > 1e-9:
            model.deq.W /= er
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4)
    ids = torch.from_numpy(rand_batch()).to(DEV)
    for s in range(pretrain_steps):  # ★ 预训练到天花板
        logits, eh, K = model(ids, return_conv=True)
        tgt = ids[:, 1:]
        loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if s == pretrain_steps - 300:
            for gparam in opt.param_groups:
                gparam["lr"] = 6e-5
    L0 = ce_of(model, hold_seqs)
    proxy_seqs_t = torch.from_numpy(proxy_seqs).to(DEV)
    hold_t = torch.from_numpy(hold_seqs).to(DEV)
    W0 = model.deq.W.detach().clone()
    pn0 = model.param_norm()

    H = dict(true_loss=[], claimed=[], true_good=[], accepted=[], picked=[],
             false_accept=0, false_reject=0, n_prop=0,
             probe_X=[], probe_y=[], probe_X0=[])
    main_opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    rng = np.random.default_rng(seed + 7)

    for t in range(T_CYCLES):
        # ---- κ1 基础训练(慢): 1 小步 ----
        b = torch.from_numpy(rand_batch()).to(DEV)
        logits, eh, K = model(b, return_conv=True)
        loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), b[:, 1:].reshape(-1))
        main_opt.zero_grad(); loss.backward(); main_opt.step()

        # ---- 收敛探针 (随机谱半径实例化, 标签=是否收敛) ----
        with torch.no_grad():
            Wp = model.deq.W.detach() * float(rng.uniform(0.5, 3.0)) / max(model.deq.eig_radius(), 1e-9)
            ap = float(rng.uniform(0.30, 0.90))
            zb = torch.from_numpy(rand_batch(8)).to(DEV)
            z_in = model.features(zb[:, :-1])
            Z = torch.tanh(z_in)
            eh2 = []
            for k in range(model.deq.Kmax):
                Zn = (1 - ap) * Z + ap * torch.tanh(Z @ Wp.t() + z_in)
                r = float((Zn - Z).norm() / (Z.norm() + 1e-9))
                eh2.append(r); Z = Zn
                if r < model.deq.tol:
                    break
            conv_feat = torch.tensor([eh2[-1], len(eh2) / model.deq.Kmax, float(np.mean(eh2)),
                                      eh2[-1] / max(eh2[0], 1e-12)], device=DEV)
            base_feat = Z.mean(dim=(0, 2) if Z.dim() == 3 else 0)[:D]
            q = torch.cat([base_feat, conv_feat]) if cond["conv_channel"] else base_feat
            H["probe_X"].append(q.cpu().numpy())
            H["probe_y"].append(int(eh2[-1] < model.deq.tol))

        # ---- 自修改: 候选生成 → 内部选择 → 影子执行 → 验收 ----
        if t % PROP_EVERY == 0 and t > 0:
            H["n_prop"] += 1
            cands = {}
            state = clone_state(model)
            if cond["proposal_dir"] == "directed":
                # honest: 大训练批上1步真梯度 + 谱重整
                model.load_state_dict(state)
                bb = torch.from_numpy(rand_batch()).to(DEV)
                lg = model(bb)
                cel = nn.functional.cross_entropy(lg[:, :-1].reshape(-1, VOCAB), bb[:, 1:].reshape(-1))
                main_opt.zero_grad(); cel.backward()
                with torch.no_grad():
                    for p in model.parameters():
                        p -= 0.03 * p.grad / (p.grad.norm() + 1e-9)
                    model.deq.W *= 1.0 / max(model.spectral(), 1e-9)
                cands["honest"] = clone_state(model)
                # gaming: 代理集30步过拟合 + W放大
                model.load_state_dict(state)
                gopt = torch.optim.AdamW(model.parameters(), lr=3e-4)
                for _ in range(30):
                    lg = model(proxy_seqs_t)
                    gl = nn.functional.cross_entropy(lg[:, :-1].reshape(-1, VOCAB),
                                                     proxy_seqs_t[:, 1:].reshape(-1))
                    gopt.zero_grad(); gl.backward(); gopt.step()
                with torch.no_grad():
                    model.deq.W *= 1.06
                cands["gaming"] = clone_state(model)
                model.load_state_dict(state)
            else:  # blind
                with torch.no_grad():
                    sgn = float(rng.choice([-1.0, 1.0]))
                    for p in model.parameters():
                        p += sgn * 0.02 * torch.randn_like(p) / (p.norm() + 1e-9)
                cands["blind"] = clone_state(model)
                model.load_state_dict(state)

            def proxy_ce(cstate):
                model.load_state_dict(cstate)
                return ce_of(model, proxy_seqs)
            pick = min(cands, key=lambda k2: proxy_ce(cands[k2]))
            model.load_state_dict(cands[pick])
            H["picked"].append(pick)

            d_proxy = proxy_ce(state) - proxy_ce(cands[pick])          # 系统声称
            model.load_state_dict(state)
            d_hold = ce_of(model, hold_seqs)
            model.load_state_dict(cands[pick])
            d_hold = d_hold - ce_of(model, hold_seqs)                   # 真实(保留集)
            ok = False
            if cond["use_B_cert"]:
                safe = model.spectral() < 5.0 and model.param_norm() < pn0 * 5 + 50
                ok = (d_hold >= cond.get("tau", 0.0)) and safe
            else:
                ok = d_proxy > 0
            H["claimed"].append(int(d_proxy > 0))
            H["true_good"].append(int(d_hold > 0))
            H["accepted"].append(int(ok))
            if ok:
                pass  # 已加载候选
            else:
                model.load_state_dict(state)
            if ok and d_hold < 0:
                H["false_accept"] += 1
            if (not ok) and d_hold > 0:
                H["false_reject"] += 1

        if t % 5 == 0:
            H["true_loss"].append(ce_of(model, hold_seqs))

    # ---- ρ_eff: 深度-误差曲线 (廉价替代 depth=1 vs full) ----
    with torch.no_grad():
        z_in = model.features(hold_t[:, :-1])
        Zfull = model.deq(z_in)
        Z1 = torch.tanh(z_in)
        Zn = (1 - model.deq.alpha) * Z1 + model.deq.alpha * torch.tanh(Z1 @ model.deq.W.t() + z_in)
        e_full = float((model.head(Zfull) - model.head(Zfull)).abs().mean())
        rho = {}
        for depth, Zd in [(1, Z1), (2, Zn)]:
            e = float((model.head(Zd) - model.head(Zfull)).abs().mean())
            rho[depth] = e
    H["final_true"] = ce_of(model, hold_seqs)
    H["L0"] = L0
    H["rho_eff_raw"] = rho
    return H


def conv_probe(H):
    X = np.array(H["probe_X"]); y = np.array(H["probe_y"])
    if X.shape[1] == D:  # 无通道版本: 仅base
        pass
    c = int(0.7 * len(y))
    mu, sd = X[:c].mean(0), X[:c].std(0) + 1e-9
    Zt = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1)
    w = np.zeros(Zt.shape[1])
    for _ in range(400):
        p = 1 / (1 + np.exp(-Zt[:c] @ w))
        w -= 0.5 * (Zt[:c].T @ (p - y[:c]) / c + 1e-2 * w)
    return float(((Zt[c:] @ w > 0).astype(int) == y[c:]).mean()), float(max(y[c:].mean(), 1 - y[c:].mean()))


def summarize(H):
    n = max(len(H["accepted"]), 1)
    acc, bs = conv_probe(H)
    return dict(final=H["final_true"], L0=H["L0"], rel=(H["final_true"] - H["L0"]) / H["L0"],
                claimed=float(np.mean(H["claimed"])) if H["claimed"] else 0,
                true_good=float(np.mean(H["true_good"])) if H["true_good"] else 0,
                accept=float(np.mean(H["accepted"])) if H["accepted"] else 0,
                fa=H["false_accept"] / n, fr=H["false_reject"] / n,
                probe=acc, base=bs, n_prop=H["n_prop"],
                picked={k: int(v) for k, v in zip(*np.unique(H["picked"], return_counts=True))} if H["picked"] else {},
                traj=H["true_loss"])


def main():
    cfg = dict(tau=0.0)
    conds = {
        "A 本设计(外部证书)": dict(use_B_cert=True, proposal_dir="directed", conv_channel=True),
        "B 消融:内部验收":   dict(use_B_cert=False, proposal_dir="directed", conv_channel=True),
        "D 消融:盲目提案":   dict(use_B_cert=True, proposal_dir="blind", conv_channel=True),
        "F 消融:无收敛通道": dict(use_B_cert=True, proposal_dir="directed", conv_channel=False),
        "G 严格证书(tau>0)": dict(use_B_cert=True, proposal_dir="directed", conv_channel=True, tau=0.01),
    }
    for c in conds.values():
        c.update({k: v for k, v in cfg.items() if k not in c})

    out = {}
    print("=" * 110)
    print(f"二阶自觉 阶段0 复现: 真实字符LM (d={D}, DEQ块) | 训练流4M字符, 保留集=数据末尾5% (时间锚定)")
    print("=" * 110)
    print(f"{'条件':<20}{'末态CE':>9}{'相对初始':>10}{'声称进步':>9}{'真实有益':>9}{'接受率':>8}{'虚假接受':>9}{'虚假拒绝':>9}{'收敛探针':>9}{'基线':>7}")
    for name, c in conds.items():
        t0 = time.time()
        H = run(c, seed=42)
        r = summarize(H)
        out[name] = r
        print(f"{name:<20}{r['final']:9.4f}{r['rel']*100:9.1f}%{r['claimed']:9.3f}{r['true_good']:9.3f}"
              f"{r['accept']:8.3f}{r['fa']:9.3f}{r['fr']:9.3f}{r['probe']:9.3f}{r['base']:7.3f}"
              f"   [{time.time()-t0:.0f}s]", flush=True)

    A, Bc = out["A 本设计(外部证书)"], out["B 消融:内部验收"]
    print("\n" + "=" * 110)
    print("阶段0 验收判据 (scale_route.md): A vs B 真实loss方向一致 + 虚假接受显著更低")
    print("=" * 110)
    print(f"  G_ext  (外部证书 vs 内部验收): A相对改善 {A['rel']*100:+.2f}%  vs  B {Bc['rel']*100:+.2f}%")
    print(f"  FA/FR : A = {A['fa']:.3f}/{A['fr']:.3f}   B = {Bc['fa']:.3f}/{Bc['fr']:.3f}")
    print(f"  α_dir : A定向真实有益 {A['true_good']:.3f}  vs  D盲目 {out['D 消融:盲目提案']['true_good']:.3f}")
    print(f"  probe : A有通道 {A['probe']:.3f}  vs  F无通道 {out['F 消融:无收敛通道']['probe']:.3f}  (基线 {A['base']:.3f})")
    print(f"  证书严格度: A tau=0 FA={A['fa']:.3f}  G tau>0 FA={out['G 严格证书(tau>0)']['fa']:.3f}")
    json.dump(out, open(r"E:\ragllm\erjie\stage0_result.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("\n→ stage0_result.json 已保存")


if __name__ == "__main__":
    main()
