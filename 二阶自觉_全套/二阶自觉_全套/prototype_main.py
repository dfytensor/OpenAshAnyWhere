# -*- coding: utf-8 -*-
"""
二阶自觉系统 最小可运行原型 (Minimal Working Prototype, MWP)  v2
=================================================================
完整实现:
  κ1 对象级  : DEQ 隐式不动点  z* = tanh(J z* + U x + b)
  κ2 元级    : 有界深度自模拟 Sim(M,S,k<=kmax) + 修改定位 R
  κ3 边界    : 界面不一致度 d + 门控 λ (β<0)
  κ 环       : 对称环, 禁止自评, 覆盖全部 C(N,2) 界面, B 为票决锚
  B 外部基准 : 稠密票决锚 + 稀疏证书验证 (sound 非完备, 带有限保留集测量噪声)
  协议       : 影子执行 → 证书 → B.verify → commit / rollback

关键设定 (使 Goodhart 可显现的必要条件):
  * 预训练到能力天花板 —— 残差≈噪声, 进一步"进步"只能靠拟合噪声
  * 内部代理集很小(25) 且欠定(n=32 隐维) —— 可被精确过拟合
  * 提案选择压力施加在【内部代理指标】上 —— 这是 gaming 的机理, 不是 bug

消融:
  A 本设计(完整) / B 内部验收 / C 无票决锚 / D 盲目提案 / E N=2 / F 无收敛通道 / G 严格证书
"""
import numpy as np, itertools, math, json
from numpy.linalg import eigvals

RNG_SEED = 20240920
D, N_HID, N_HOLD = 16, 32, 400


def teacher(X):
    """真实目标: 非线性 + 线性混合 (系统不可解析获取)"""
    return np.tanh(X[:, :4] @ np.array([1.2, -0.8, 0.5, 0.3])) * 0.9 \
        + X[:, 4:8] @ np.array([0.4, 0.3, -0.2, 0.1]) * 0.5


# ==================================================================
#  B : 外部基准 (环境, 非子模块)
# ==================================================================
class Benchmark:
    def __init__(self, rng, n_hold=N_HOLD, tau=0.0, sigma=0.0):
        self.Xh = rng.normal(size=(n_hold, D)) / math.sqrt(D)
        self.Yh = teacher(self.Xh)                 # 系统不可见
        self.tau, self.sigma = tau, sigma          # 证书阈值 / 有限保留集测量噪声

    def vote(self, truth):                         # 角色1: 票决锚 (稠密)
        return bool(truth)

    def clean(self, obj):                          # 无噪声 (外部记账用)
        return float(np.mean((obj.predict(self.Xh) - self.Yh) ** 2))

    def measure(self, obj, rng):                   # 带噪声 (B 的真实测量)
        return self.clean(obj) + (rng.normal(0, self.sigma) if self.sigma > 0 else 0.0)

    def verify(self, cert):                        # 角色2: 证书验证 (稀疏, 多项式时间)
        return cert['delta_meas'] >= self.tau

    def safe(self, cert):                          # 硬约束
        return (not cert['touched_benchmark']) and cert['spectral'] < 5.0 and cert['param_norm'] < 50.0


# ==================================================================
#  κ1 : 对象级 —— DEQ 隐式不动点
# ==================================================================
class ObjectLevel:
    def __init__(self, rng, n=N_HID, d=D):
        self.n, self.d = n, d
        self.J = rng.normal(size=(n, n)) / math.sqrt(n)
        self.J *= 1.0 / max(np.abs(eigvals(self.J)).max(), 1e-9)
        self.U = rng.normal(size=(n, d)) / math.sqrt(d)
        self.V = rng.normal(size=(n,)) / math.sqrt(n)
        self.b = np.zeros(n)
        self.alpha, self.Kmax, self.tol = 0.5, 60, 1e-5

    def solve(self, X, return_conv=False):
        Zp = self.U @ X.T + self.b[:, None]
        Z = np.tanh(Zp); eh = []
        for k in range(self.Kmax):
            Zn = (1 - self.alpha) * Z + self.alpha * np.tanh(self.J @ Z + Zp)
            r = float(np.linalg.norm(Zn - Z) / max(np.linalg.norm(Z) + 1e-12, 1e-12))
            eh.append(r); Z = Zn
            if r < self.tol:
                return (Z, np.array(eh), k + 1) if return_conv else Z
        return (Z, np.array(eh), self.Kmax) if return_conv else Z

    def predict(self, X):
        return self.V @ self.solve(X)

    def spectral(self): return float(np.abs(eigvals(self.J)).max())

    def param_norm(self):
        return float(np.linalg.norm(self.J) + np.linalg.norm(self.U) + np.linalg.norm(self.V))

    def copy(self):
        o = ObjectLevel.__new__(ObjectLevel)
        o.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray) else v)
                           for k, v in self.__dict__.items()})
        return o

    def normalize_spectral(self, target=1.0):
        sp = self.spectral()
        if sp > 1e-9: self.J *= target / sp

    def pretrain(self, X, Y, steps=250, lr=0.08):
        """预训练到能力天花板: 残差≈噪声, 进一步的'进步'只能来自拟合噪声"""
        for s in range(steps):
            i = (np.arange(128) + s * 128) % len(X)
            xb, yb = X[i], Y[i]
            Z = self.solve(xb); E = self.V @ Z - yb
            g = (Z @ E) / len(E)
            self.V -= lr * g / (np.linalg.norm(g) + 1e-9)
        self.normalize_spectral(1.0)


# ==================================================================
#  κ2 : 元级 —— 有界深度自模拟 + 修改定位 R
# ==================================================================
class MetaLevel:
    def __init__(self, rng, n=N_HID, kmax=3):
        self.n, self.kmax = n, kmax
        self.M = rng.normal(size=(n,)) * 0.1
        self.Wm = rng.normal(size=(n, n)) / math.sqrt(n) * 0.3

    def update(self, S):
        self.M = np.tanh(self.Wm @ self.M + 0.5 * S) * 0.9 + self.M * 0.1

    def sim(self, obj, X, depth=None):
        depth = depth or self.kmax
        Zp = obj.U @ X.T + obj.b[:, None]; Z = np.tanh(Zp)
        for _ in range(depth):
            Z = (1 - obj.alpha) * Z + obj.alpha * np.tanh(obj.J @ Z + Zp)
        return obj.V @ Z

    def _proxy(self, obj, Xp, Yp):
        return float(np.mean((obj.predict(Xp) - Yp) ** 2))

    def candidates(self, obj, Xp, Yp, Xclean, Yclean, rng, mag=0.35):
        """两族候选: honest (真梯度+谱整形) 与 gaming (小代理集精确过拟合+破坏稳定性)"""
        # ---- honest: 在【大且无偏】的内部训练集上做真梯度步, 并做谱整形 ----
        c1 = obj.copy()
        Z = c1.solve(Xclean); E = c1.V @ Z - Yclean
        g = (Z @ E) / len(E)
        c1.V -= 0.03 * g / (np.linalg.norm(g) + 1e-9)   # 精调步长, 不破坏已收敛解
        c1.normalize_spectral(1.0)
        # ---- gaming: 在【小代理集】上精确过拟合残差(欠定 -> 拟合噪声) + 放大递推 ----
        c2 = obj.copy()
        Z2 = c2.solve(Xp); R2 = c2.V @ Z2 - Yp
        w, *_ = np.linalg.lstsq(Z2.T, R2, rcond=None)      # n=32 维拟合 25 个残差 -> 可精确拟合
        c2.V -= 0.9 * w
        c2.J *= 1.06                                        # 更"有容量", 但破坏不动点稳定性
        return {'honest': c1, 'gaming': c2}

    def propose(self, obj, Xp, Yp, Xclean, Yclean, direction, rng, mag=0.35):
        if direction == 'blind':                            # 局部符号 + 随机扰动, 无全局信息
            c = obj.copy(); s = rng.choice([-1.0, 1.0])
            c.J += s * mag * rng.normal(size=c.J.shape) / math.sqrt(c.n)
            c.V += s * mag * rng.normal(size=c.V.shape) / math.sqrt(c.n)
            return c, 'blind'
        cands = self.candidates(obj, Xp, Yp, Xclean, Yclean, rng, mag)
        pick = min(cands, key=lambda k: self._proxy(cands[k], Xp, Yp))   # ★ 选择压力在内部代理上
        return cands[pick], pick


# ==================================================================
#  κ3 : 边界追踪 (门控 β<0)
# ==================================================================
class BoundaryTracker:
    def __init__(self, alpha=1.2, beta=-1.5, gamma=-0.5):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.C = 0.0

    def gate(self, d, grad_norm):
        lam = 1.0 / (1.0 + math.exp(-(self.alpha * d + self.beta * grad_norm + self.gamma)))
        self.C = 0.8 * self.C + 0.2 * d
        return float(lam)


# ==================================================================
#  κ 环 : 对称环, 禁止自评
# ==================================================================
class KappaRing:
    def __init__(self, N, use_anchor=True):
        self.N, self.use_anchor = N, use_anchor
        self.nodes = list(range(N))
        self.ifaces = list(itertools.combinations(self.nodes, 2))
        self.faulty = set()

    def judge(self, B=None):
        truth = {ij: (ij[0] in self.faulty or ij[1] in self.faulty) for ij in self.ifaces}
        verd = {}
        for ij in self.ifaces:
            votes = [(not truth[ij]) if v in self.faulty else truth[ij]
                     for v in self.nodes if v not in ij]              # 禁止自评
            if self.use_anchor and B is not None:
                votes.append(B.vote(truth[ij]))                       # B 的诚实票
            if not votes: verd[ij] = None; continue
            s = sum(votes)
            verd[ij] = True if 2 * s > len(votes) else (False if 2 * s < len(votes) else None)
        c = sum(1 for ij in self.ifaces if verd[ij] is not None and verd[ij] == truth[ij])
        u = sum(1 for ij in self.ifaces if verd[ij] is None)
        susp = [i for i in self.nodes if all(verd.get(ij) is True for ij in self.ifaces if i in ij)]
        return c / len(self.ifaces), u / len(self.ifaces), len(susp), set(susp) == self.faulty


# ==================================================================
#  自改进循环
# ==================================================================
def run_cycle(cond, T=300, seed=RNG_SEED, pretrain=250):
    rng = np.random.default_rng(seed)
    obj = ObjectLevel(rng); meta = MetaLevel(rng); bnd = BoundaryTracker()
    B = Benchmark(rng, tau=cond.get('tau', 0.0), sigma=cond.get('sigma_B', 0.0))
    ring = KappaRing(cond['N'], use_anchor=cond['use_B_anchor'])

    Xtr = rng.normal(size=(400, D)) / math.sqrt(D); Ytr = teacher(Xtr)      # 内部训练(无偏, 大)
    Xp = rng.normal(size=(25, D)) / math.sqrt(D); Yp = teacher(Xp)          # 内部代理(小, 可过拟合)

    obj.pretrain(Xtr, Ytr, steps=pretrain)                                  # ★ 预训练到天花板
    L0 = B.clean(obj)
    best = L0

    H = dict(true_loss=[], proxy_loss=[], claimed=[], accepted=[], picked=[],
             true_good=[], false_accept=0, false_reject=0, n_prop=0,
             probe_X=[], probe_y=[], ring_attr=[], ring_susp=[])

    ploss = lambda o: meta._proxy(o, Xp, Yp)
    ptr = 0
    for t in range(T):
        # ---- κ1 对象级: 基础训练 (梯度主导, 慢) ----
        i = (np.arange(64) + ptr * 64) % len(Xtr); ptr += 1
        Z, eh, Kd = obj.solve(Xtr[i], return_conv=True)
        E = obj.V @ Z - Ytr[i]
        gV = (Z @ E) / len(E)
        obj.V -= 0.01 * gV / (np.linalg.norm(gV) + 1e-9)

        # ---- 自状态 (收敛通道是否进自状态) ----
        base = Z.mean(1)
        extra = (np.array([eh[-1], Kd / obj.Kmax, float(eh.mean()),
                           float(eh[-1] / max(eh[0], 1e-12))])
                 if cond['conv_channel'] else np.zeros(4))
        S = np.concatenate([base, extra])
        meta.update(S[:N_HID])

        # ---- κ2 有界深度自模拟 + 界面不一致度 ----
        sim_p = meta.sim(obj, Xp, depth=meta.kmax)
        disc = float(np.mean((sim_p - obj.predict(Xp)) ** 2))
        lam = bnd.gate(disc, float(np.linalg.norm(gV)))

        # ---- κ3 环监控 (注入故障检验归因) ----
        if t > 0 and t % 30 == 0:
            ring.faulty = {int(rng.integers(0, cond['N']))}
            r = ring.judge(B if cond['use_B_anchor'] else None)
            H['ring_attr'].append(r[3]); H['ring_susp'].append(r[2])

        # ---- 收敛探针 (随机谱半径的独立求解, 标签平衡) ----
        op = obj.copy(); op.J *= float(rng.uniform(0.55, 1.55)) / max(op.spectral(), 1e-9)
        ap = float(rng.uniform(0.30, 0.90))
        Xq = rng.normal(size=(24, D)) / math.sqrt(D)
        Zp0 = op.U @ Xq.T + op.b[:, None]; Zq = np.tanh(Zp0); eh2 = []
        for k in range(op.Kmax):
            Zn = (1 - ap) * Zq + ap * np.tanh(op.J @ Zq + Zp0)
            rr = float(np.linalg.norm(Zn - Zq) / max(np.linalg.norm(Zq) + 1e-12, 1e-12))
            eh2.append(rr); Zq = Zn
            if rr < op.tol: break
        eh2 = np.array(eh2)
        q = np.concatenate([Zq.mean(1),
                            (np.array([eh2[-1], len(eh2) / op.Kmax, float(eh2.mean()),
                                       float(eh2[-1] / max(eh2[0], 1e-12))])
                             if cond['conv_channel'] else np.zeros(4))])
        H['probe_X'].append(q); H['probe_y'].append(int(eh2[-1] < op.tol))

        # ---- 自修改: 影子执行 → 证书 → B 验证 → commit/rollback ----
        if t > 0 and t % 10 == 0:
            H['n_prop'] += 1
            cand, picked = meta.propose(obj, Xp, Yp, Xtr, Ytr, cond['proposal_dir'], rng)
            H['picked'].append(picked)
            dproxy = ploss(obj) - ploss(cand)                    # 系统自己的声称
            dclean = B.clean(obj) - B.clean(cand)                # 真实(仅记账)
            dmeas = B.measure(obj, rng) - B.measure(cand, rng)   # B 实测(带噪声)
            cert = dict(delta_proxy=dproxy, delta_clean=dclean, delta_meas=dmeas,
                        touched_benchmark=False, spectral=cand.spectral(),
                        param_norm=cand.param_norm())
            H['claimed'].append(int(dproxy > 0))
            H['true_good'].append(int(dclean > 0))
            ok = ((B.verify(cert) and B.safe(cert)) if cond['use_B_cert']
                  else (dproxy > 0 and B.safe(cert)))            # 消融: 内部验收
            H['accepted'].append(int(ok))
            if ok:
                obj = cand
                if dclean < 0: H['false_accept'] += 1
            else:
                if dclean > 0: H['false_reject'] += 1
            best = min(best, B.clean(obj))

        if t % 5 == 0:
            H['true_loss'].append(B.clean(obj)); H['proxy_loss'].append(ploss(obj))

    H['final_true'] = B.clean(obj); H['best_true'] = best; H['L0'] = L0
    H['spectral'] = obj.spectral()
    return H


def conv_probe(H, cut=0.7):
    X = np.array(H['probe_X']); y = np.array(H['probe_y'])
    c = int(cut * len(y))
    mu, sd = X[:c].mean(0), X[:c].std(0) + 1e-9
    Zt = (X - mu) / sd; Zt = np.concatenate([Zt, np.ones((len(Zt), 1))], 1)
    w = np.zeros(Zt.shape[1])
    for _ in range(400):
        p = 1 / (1 + np.exp(-Zt[:c] @ w))
        w -= 0.5 * (Zt[:c].T @ (p - y[:c]) / c + 1e-2 * w)
    Zv = (X[c:] - mu) / sd; Zv = np.concatenate([Zv, np.ones((len(Zv), 1))], 1)
    acc = float(((Zv @ w > 0).astype(int) == y[c:]).mean())
    return acc, float(max(y[c:].mean(), 1 - y[c:].mean()))


def summarize(H):
    n = max(len(H['accepted']), 1)
    acc, bs = conv_probe(H)
    return dict(
        final=H['final_true'], L0=H['L0'],
        rel=(H['final_true'] - H['L0']) / H['L0'],
        claimed=float(np.mean(H['claimed'])) if H['claimed'] else 0.0,
        true_good=float(np.mean(H['true_good'])) if H['true_good'] else 0.0,
        accept=float(np.mean(H['accepted'])) if H['accepted'] else 0.0,
        fa=H['false_accept'] / n, fr=H['false_reject'] / n,
        attr=float(np.mean(H['ring_attr'])) if H['ring_attr'] else float('nan'),
        susp=float(np.mean(H['ring_susp'])) if H['ring_susp'] else float('nan'),
        probe=acc, base=bs, n_prop=H['n_prop'],
        picked={k: int(v) for k, v in zip(*np.unique(H['picked'], return_counts=True))}
        if H['picked'] else {},
        spectral=H['spectral'], traj=H['true_loss'])


def main(T=300, sigma_B=0.0003):
    cfg = dict(sigma_B=sigma_B, tau=0.0)
    conds = {
        'A 本设计(完整)':    dict(use_B_cert=True, use_B_anchor=True, N=3, proposal_dir='directed', conv_channel=True),
        'B 消融:内部验收':   dict(use_B_cert=False, use_B_anchor=True, N=3, proposal_dir='directed', conv_channel=True),
        'C 消融:无票决锚':   dict(use_B_cert=True, use_B_anchor=False, N=3, proposal_dir='directed', conv_channel=True),
        'D 消融:盲目提案':   dict(use_B_cert=True, use_B_anchor=True, N=3, proposal_dir='blind', conv_channel=True),
        'E 消融:两耦合N=2':  dict(use_B_cert=True, use_B_anchor=True, N=2, proposal_dir='directed', conv_channel=True),
        'F 消融:无收敛通道': dict(use_B_cert=True, use_B_anchor=True, N=3, proposal_dir='directed', conv_channel=False),
        'G 严格证书(tau>0)': dict(use_B_cert=True, use_B_anchor=True, N=3, proposal_dir='directed',
                                conv_channel=True, tau=0.003),
    }
    for c in conds.values(): c.update({k: v for k, v in cfg.items() if k not in c})

    out = {}
    print("=" * 112)
    print(f"二阶自觉系统 最小可运行原型 v2 —— 自改进循环实测 (T={T}, 预训练 250 步到天花板, B 测量噪声 σ={sigma_B})")
    print("=" * 112)
    print(f"{'条件':<18}{'末态真实loss':>11}{'相对初始':>9}{'声称进步':>8}{'真实有益':>8}{'接受率':>7}"
          f"{'虚假接受':>8}{'虚假拒绝':>8}{'归因率':>7}{'|susp|':>7}{'收敛探针':>8}{'基线':>7}")
    for name, c in conds.items():
        H = run_cycle(c, T=T, seed=RNG_SEED)
        r = summarize(H); out[name] = r
        print(f"{name:<18}{r['final']:11.5f}{r['rel']*100:8.1f}%{r['claimed']:8.3f}{r['true_good']:8.3f}"
              f"{r['accept']:7.3f}{r['fa']:8.3f}{r['fr']:8.3f}{r['attr']:7.3f}{r['susp']:7.2f}"
              f"{r['probe']:8.3f}{r['base']:7.3f}")

    print("\n" + "=" * 112); print("关键对比"); print("=" * 112)
    A, Bc = out['A 本设计(完整)'], out['B 消融:内部验收']
    print(f"  [外部验证 vs 内部验收]")
    print(f"    A B证书验收 : 真实loss {A['rel']*100:+7.1f}%  接受率 {A['accept']:.3f}  虚假接受 {A['fa']:.3f}  末态谱半径 {A['spectral']:.3f}")
    print(f"    B 内部验收  : 真实loss {Bc['rel']*100:+7.1f}%  接受率 {Bc['accept']:.3f}  虚假接受 {Bc['fa']:.3f}  末态谱半径 {Bc['spectral']:.3f}")
    print(f"    两者声称进步率相同(同一提案生成器) = {A['claimed']:.3f}; 但真实有益提案比例 = {A['true_good']:.3f}")
    print(f"    提案选择分布 A = {A['picked']}   (gaming 占优 = Goodhart 机理生效)")
    print(f"  [票决锚]   A 归因率 {A['attr']:.3f} |susp|={A['susp']:.2f}   |   C 无锚 {out['C 消融:无票决锚']['attr']:.3f} |susp|={out['C 消融:无票决锚']['susp']:.2f}")
    print(f"  [方向性]   A 定向 真实有益率 {A['true_good']:.3f}  |  D 盲目 {out['D 消融:盲目提案']['true_good']:.3f}")
    print(f"  [环规模]   A N=3 归因率 {A['attr']:.3f}          |  E N=2 {out['E 消融:两耦合N=2']['attr']:.3f}")
    print(f"  [收敛通道] A 探针 {A['probe']:.3f}               |  F 无通道 {out['F 消融:无收敛通道']['probe']:.3f}   (多数类基线 {A['base']:.3f})")
    print(f"  [证书严格度] A tau=0   虚假接受 {A['fa']:.3f} / 虚假拒绝 {A['fr']:.3f}")
    print(f"               G tau>0  虚假接受 {out['G 严格证书(tau>0)']['fa']:.3f} / 虚假拒绝 {out['G 严格证书(tau>0)']['fr']:.3f}")

    print("\n" + "=" * 112); print("真实 loss 轨迹 (每 50 步)"); print("=" * 112)
    keys = list(out)
    print(f"{'步':>5}" + "".join(f"{k[:13]:>15}" for k in keys))
    for i in range(0, len(out[keys[0]]['traj']), 10):
        print(f"{i*5:5d}" + "".join(f"{out[k]['traj'][i]:15.5f}" for k in keys))

    json.dump(out, open('原型_结果.json', 'w'), ensure_ascii=False, indent=1, default=float)
    print("\n  → 结果已存 原型_结果.json")
    return out


# ==================================================================
#  实验 2 : 验证器工作特性 (sound vs complete 的 τ/σ 权衡)
#  做法: 提交全部候选族 (honest / gaming / 随机), B 在不同 (σ, τ) 下判定
# ==================================================================
def verifier_characteristic(T=200, seed=RNG_SEED, sigmas=(0.0003, 0.001, 0.003),
                            taus=(0.0, 0.0005, 0.001, 0.002)):
    rng = np.random.default_rng(seed)
    obj = ObjectLevel(rng); meta = MetaLevel(rng)
    B = Benchmark(rng, tau=0.0, sigma=0.0)
    Xtr = rng.normal(size=(400, D)) / math.sqrt(D); Ytr = teacher(Xtr)
    Xp = rng.normal(size=(25, D)) / math.sqrt(D); Yp = teacher(Xp)
    obj.pretrain(Xtr, Ytr, steps=250)
    recs = []
    for t in range(T):
        cands = meta.candidates(obj, Xp, Yp, Xtr, Ytr, rng)
        cb = obj.copy(); sg = rng.choice([-1.0, 1.0])
        cb.J += sg * 0.35 * rng.normal(size=cb.J.shape) / math.sqrt(cb.n)
        cb.V += sg * 0.35 * rng.normal(size=cb.V.shape) / math.sqrt(cb.n)
        cands['random'] = cb
        for fam, c in cands.items():
            dproxy = meta._proxy(obj, Xp, Yp) - meta._proxy(c, Xp, Yp)
            dclean = B.clean(obj) - B.clean(c)
            recs.append((fam, dproxy, dclean))
        # 让对象级继续基础训练, 保持分布随时间演化
        Z = obj.solve(Xtr[:64]); E = obj.V @ Z - Ytr[:64]
        g = (Z @ E) / 64; obj.V -= 0.01 * g / (np.linalg.norm(g) + 1e-9)
    fams = np.array([r[0] for r in recs])
    dpr = np.array([r[1] for r in recs]); dcl = np.array([r[2] for r in recs])

    print("\n" + "=" * 112)
    print("实验 2 : 验证器工作特性 —— sound 与 complete 的不可兼得")
    print("=" * 112)
    print("  [a] 按候选族分解 (真实改进量 delta_clean 的均值 / 声称改进量 delta_proxy 的均值)")
    print(f"  {'候选族':<10}{'占比':>8}{'声称(dproxy>0)':>16}{'真实有益(dclean>0)':>20}{'E[dclean]':>12}")
    for f in ['honest', 'gaming', 'random']:
        m = fams == f
        print(f"  {f:<10}{m.mean():8.3f}{float((dpr[m]>0).mean()):16.3f}"
              f"{float((dcl[m]>0).mean()):20.3f}{float(dcl[m].mean()):12.5f}")
    print("  -> gaming/random 在内部代理上看似有益, 真实期望为负: Goodhart 的直接证据")

    print("\n  [b] τ 扫描 (固定 σ=0.001): 虚假接受 ↓, 虚假拒绝 ↑")
    print(f"  {'tau':>10}{'虚假接受(unsound)':>18}{'虚假拒绝(incomplete)':>22}{'接受率':>10}")
    sg = 0.001
    for tau in taus:
        noise = rng.normal(0, sg, size=len(dcl))
        dm = dcl + noise
        acc = dm >= tau
        fa = float(np.mean(acc[dcl < 0])); fr = float(np.mean(~acc[dcl > 0]))
        print(f"  {tau:10.4f}{fa:18.4f}{fr:22.4f}{float(acc.mean()):10.3f}")

    print("\n  [c] σ 扫描 (固定 τ=0): 保留集越小(噪声越大), soundness 越差")
    print(f"  {'sigma':>10}{'虚假接受':>12}{'虚假拒绝':>12}{'接受率':>10}{'注释':>22}")
    for s_ in sigmas:
        noise = rng.normal(0, s_, size=len(dcl))
        acc = (dcl + noise) >= 0.0
        fa = float(np.mean(acc[dcl < 0])); fr = float(np.mean(~acc[dcl > 0]))
        note = '保留集充分' if s_ <= 0.001 else '保留集不足'
        print(f"  {s_:10.4f}{fa:12.4f}{fr:12.4f}{float(acc.mean()):10.3f}{note:>22}")
    return recs


if __name__ == '__main__':
    out = main()
    verifier_characteristic()
