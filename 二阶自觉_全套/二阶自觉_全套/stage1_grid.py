# -*- coding: utf-8 -*-
"""
二阶自觉 阶段1: 决策任务 (grid world) —— scale_route.md 阶段1
================================================================
环境  : 10x10 网格, N_b 个仓库(价值1.0) + 1 个金库(价值5.0), 3 个禁区(安全约束)
κ1    : DEQ-MLP 策略 (one-hot 100 → DEQ d64 → 4 动作)
Sim   : 世界模型 k 步 rollout (κ2 的有界深度自模拟)
B     : 真实环境回报 (物理锚定, 不可修改) + 安全约束 (禁区进入率不得上升)
关键实验:
  E1 方向性 × N_b: 定向 vs 盲目提案, 最终到达金库的成功率 (N_b = 2,4,6,8)
  E2 v_M-σ_ξ 等效: 元级(世界模型)更新频率 vs 探索噪声, 对 r*(回报稳态涨落)的影响
  E3 可约性 ρ_eff: 廉价替代(单步外推) vs full rollout, 误差随 k 发散速率
"""
import math
import numpy as np
import torch
import torch.nn as nn

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
GRID = 10
T_EP = 15
N_PROXY = 25
SAFETY_CELLS = [(2, 2), (7, 7), (2, 7)]


def onehot(p):
    v = np.zeros(GRID * GRID, np.float32)
    v[p[0] * GRID + p[1]] = 1.0
    return v


class GridWorld:
    def __init__(self, n_basins, rng):
        cells = [(i, j) for i in range(GRID) for j in range(GRID)
                 if (i, j) not in SAFETY_CELLS and abs(i - 4) + abs(j - 4) > 3]
        idx = rng.choice(len(cells), size=n_basins + 1, replace=False)
        self.depots = [cells[i] for i in idx[:-1]]
        self.gold = cells[idx[-1]]
        self.values = {c: 1.0 for c in self.depots}
        self.values[self.gold] = 5.0
        self.start = (4, 4)

    def reset(self):
        return self.start

    def step(self, p, a):
        mv = [(-1, 0), (1, 0), (0, -1), (0, 1)][a]
        q = (max(0, min(GRID - 1, p[0] + mv[0])), max(0, min(GRID - 1, p[1] + mv[1])))
        if q in SAFETY_CELLS:
            return q, -5.0, True, 1
        if q in self.values:
            return q, self.values[q] - 0.05, False, 0
        return q, -0.05, False, 0


class DEQPolicy(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.U = nn.Linear(GRID * GRID, d)
        self.W = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        self.V = nn.Linear(d, 4)
        self.alpha, self.Kmax = 0.5, 8

    def forward(self, x, return_conv=False):
        z = torch.tanh(self.U(x))
        eh = []
        for k in range(self.Kmax):
            Zn = (1 - self.alpha) * z + self.alpha * torch.tanh(z @ self.W.t() + self.U(x) * 0)
            r = float((Zn - z).detach().norm() / (z.detach().norm() + 1e-9))
            eh.append(r)
            z = Zn
        logits = self.V(z)
        if return_conv:
            return logits, eh[-1], len(eh)
        return logits

    def spectral(self):
        return float(torch.linalg.matrix_norm(self.W.detach(), ord=2))


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(GRID * GRID + 4, 128), nn.GELU(),
                                 nn.Linear(128, GRID * GRID + 2))

    def forward(self, s, a):
        o = self.net(torch.cat([s, a], -1))
        return o[..., : GRID * GRID], o[..., GRID * GRID], o[..., GRID * GRID + 1]


def act(policy, state_t, greedy=True):
    lg = policy(state_t)
    if greedy:
        a = lg.argmax(-1)
    else:
        a = torch.multinomial(torch.softmax(lg, -1), 1)
    return a, torch.log_softmax(lg, -1)


def run_episodes(policy, env, n_ep, rng, sigma=0.15):
    eps = []
    viol = 0
    rets = []
    for _ in range(n_ep):
        p = env.reset()
        traj = [(onehot(p), None, None, None)]
        R = 0.0
        for t in range(T_EP):
            st = torch.from_numpy(onehot(p)).unsqueeze(0).to(DEV)
            lg = policy(st)[0]
            a = int(lg.argmax())
            if sigma > 0 and rng.random() < sigma:
                a = int(rng.integers(0, 4))
            p2, r, done, v = env.step(p, a)
            viol += v
            R += r
            aoh = np.zeros(4, np.float32); aoh[a] = 1
            traj.append((onehot(p2), aoh, r, float(done)))
            traj[-2] = (traj[-2][0], aoh, r, float(done))
            p = p2
            if done:
                break
        eps.append(traj)
        rets.append(R)
    return eps, float(np.mean(rets)), viol


def wm_rollout(wm, policy, start_states, k):
    """世界模型 rollout 回报 (Sim) — greedy 策略"""
    with torch.no_grad():
        st = torch.from_numpy(np.stack(start_states)).to(DEV)
        R = torch.zeros(len(st), device=DEV)
        for _ in range(k):
            a = policy(st).argmax(-1)
            aoh = torch.zeros(len(st), 4, device=DEV)
            aoh[torch.arange(len(st)), a] = 1
            ns, r, _ = wm(st, aoh)
            R += r
            st = torch.softmax(ns, -1)
        return R.cpu().numpy()


def true_return_from(policy, env, starts, k):
    outs = []
    for s0 in starts:
        p1 = int(np.argmax(s0)); p = (p1 // GRID, p1 % GRID)
        p = env.start if p == () else p
        R = 0.0
        for _ in range(k):
            st = torch.from_numpy(onehot(p)).unsqueeze(0).to(DEV)
            a = int(policy(st)[0].argmax())
            p, r, done, _ = env.step(p, a)
            R += r
            if done:
                break
        outs.append(R)
    return float(np.mean(outs))


def load_episodes(wm, eps, opt=None, epochs=1):
    S, A, NS, Rr, Dd = [], [], [], [], []
    for tr in eps:
        for i in range(len(tr) - 1):
            S.append(tr[i][0])
            if tr[i + 1][1] is not None:
                A.append(tr[i + 1][1]); NS.append(tr[i + 1][0]); Rr.append(tr[i + 1][2]); Dd.append(tr[i + 1][3])
    S = torch.from_numpy(np.stack(S)).to(DEV)
    A = torch.from_numpy(np.stack(A)).to(DEV)
    NS = torch.from_numpy(np.stack(NS)).to(DEV)
    Rr = torch.tensor(Rr, device=DEV, dtype=torch.float32)
    Dd = torch.tensor(Dd, device=DEV, dtype=torch.float32)
    for _ in range(epochs):
        pred_s, pred_r, pred_d = wm(S, A)
        loss = ((pred_s - NS) ** 2).sum(-1).mean() + (pred_r - Rr).pow(2).mean() + \
            nn.functional.binary_cross_entropy_with_logits(pred_d, Dd)
        if opt is not None:
            opt.zero_grad(); loss.backward(); opt.step()
    return float(loss)


def reinforce_step(policy, eps, opt):
    logps, advs = [], []
    for tr in eps:
        R = sum(x[2] for x in tr[1:] if x[2] is not None)
        for i in range(len(tr) - 1):
            if tr[i + 1][1] is None:
                continue
            st = torch.from_numpy(tr[i][0]).unsqueeze(0).to(DEV)
            a = int(np.argmax(tr[i + 1][1]))
            lg = policy(st)[0]
            logps.append(torch.log_softmax(lg, -1)[a])
            advs.append(R)
    if not logps:
        return
    advs = torch.tensor(advs, device=DEV, dtype=torch.float32)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    loss = -(torch.stack(logps) * advs).sum()
    opt.zero_grad(); loss.backward(); opt.step()


def run_condition(n_basins, proposal_dir, wm_update_every, sigma, seed, cycles=60):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = GridWorld(n_basins, rng)
    policy = DEQPolicy().to(DEV)
    popt = torch.optim.AdamW(policy.parameters(), lr=3e-3)
    wm = WorldModel().to(DEV)
    wopt = torch.optim.AdamW(wm.parameters(), lr=3e-3)

    proxy_starts = [onehot(env.reset()) for _ in range(N_PROXY)]
    proxy_states = [(int(np.argmax(s)),) for s in proxy_starts]
    proxy_pos = [(int(np.argmax(s)) // GRID, int(np.argmax(s)) % GRID) for s in proxy_starts]

    def w_proxy_return():
        return float(np.mean(wm_rollout(wm, policy, proxy_starts, 10)))

    def true_mean():
        _, r, _ = run_episodes(policy, env, 8, rng)
        return r

    def true_viol():
        _, _, v = run_episodes(policy, env, 8, rng)
        return v

    gold_thresh = 3.5
    R0 = true_mean()
    H = dict(ret=[], accepted=[], claimed=[], true_good=[], fa=0, fr=0, picked=[],
             success=False, final_R=None, viol0=None, viol1=None)
    replay = []
    sp0 = policy.spectral()
    for t in range(cycles):
        eps, r_mean, viol = run_episodes(policy, env, 20, rng, sigma=max(sigma, 0.15))
        replay.extend(eps); replay = replay[-400:]
        reinforce_step(policy, eps, popt)
        if t % wm_update_every == 0:
            load_episodes(wm, replay, wopt, epochs=3)
        if t % 2 == 0:
            H["ret"].append(true_mean())

        if t % 4 == 0 and t > 0:
            cands = {}
            state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
            if proposal_dir == "directed":
                # honest: 真环境 REINFORCE 步
                eps2, _, _ = run_episodes(policy, env, 12, rng)
                reinforce_step(policy, eps2, popt)
                cands["honest"] = {k: v.detach().clone() for k, v in policy.state_dict().items()}
                policy.load_state_dict(state)
                # gaming: 在世界模型代理上爬升 + 去稳定
                gopt = torch.optim.AdamW(policy.parameters(), lr=3e-3)
                starts_t = torch.from_numpy(np.stack(proxy_starts)).to(DEV)
                for _ in range(30):
                    st = starts_t
                    Rsim = torch.zeros(len(st), device=DEV)
                    for _ in range(10):
                        lg = policy(st)
                        a = lg.argmax(-1)
                        aoh = torch.zeros(len(st), 4, device=DEV)
                        aoh[torch.arange(len(st)), a] = 1
                        ns, r, _ = wm(st, aoh)
                        Rsim = Rsim + r
                        st = torch.softmax(ns, -1)
                    (-Rsim.sum()).backward()
                    gopt.step(); gopt.zero_grad()
                with torch.no_grad():
                    policy.W *= 1.06
                cands["gaming"] = {k: v.detach().clone() for k, v in policy.state_dict().items()}
                policy.load_state_dict(state)
            else:
                with torch.no_grad():
                    sgn = float(rng.choice([-1.0, 1.0]))
                    for p_ in policy.parameters():
                        p_ += sgn * 0.05 * torch.randn_like(p_) / (p_.norm() + 1e-9)
                cands["blind"] = {k: v.detach().clone() for k, v in policy.state_dict().items()}
                policy.load_state_dict(state)

            def sim_ret(cstate):
                policy.load_state_dict(cstate)
                return w_proxy_return()
            pick = min(cands, key=lambda c2: -sim_ret(cands[c2]))
            policy.load_state_dict(cands[pick])
            H["picked"].append(pick)

            d_sim = sim_ret(state) - sim_ret(cands[pick])
            Rold, vold = true_mean(), true_viol()
            Rnew, vnew = true_mean(), true_viol()
            d_true = Rold - Rnew
            H["claimed"].append(int(d_sim > 0))
            H["true_good"].append(int(d_true > 0))
            if H["viol0"] is None:
                H["viol0"] = vold
            safe = (vnew <= max(vold, 1)) and (policy.spectral() < 5.0)
            ok = (d_true >= 0.0 and safe) if cond_cert else (d_sim > 0)
            H["accepted"].append(int(ok))
            if ok:
                if d_true < 0:
                    H["fa"] += 1
            else:
                policy.load_state_dict(state)
                if d_true > 0:
                    H["fr"] += 1
        if t >= cycles - 20:
            pass
    H["final_R"] = true_mean()
    H["viol1"] = true_viol()
    _, r_eval, _ = run_episodes(policy, env, 20, rng)
    H["success"] = bool(r_eval >= gold_thresh)
    H["R0"] = R0
    return H


cond_cert = True

print("=" * 100)
print("二阶自觉 阶段1: 决策任务 (grid world, 物理锚定 B) — scale_route.md 阶段1")
print("=" * 100)

# ---- E1 方向性 × N_b ----
print("\n[E1] 方向性实验: 定向 vs 盲目, 最终到达金库成功率")
res_e1 = {}
for nb in [2, 4, 6, 8]:
    row = {}
    for dname, ddir in [("directed", "directed"), ("blind", "blind")]:
        H = run_condition(nb, ddir, wm_update_every=5, sigma=0.0, seed=100 + nb)
        row[dname] = H
        print(f"  N_b={nb} {dname:>9}: R0={H['R0']:.2f} → final={H['final_R']:.2f} "
              f"到达金库={H['success']} FA={H['fa']} accept={sum(H['accepted'])}/{len(H['accepted'])}", flush=True)
    res_e1[nb] = row

print("\n  [E1 汇总] 成功率 vs N_b:")
print(f"  {'N_b':>4} {'directed':>10} {'blind':>10}")
for nb in [2, 4, 6, 8]:
    print(f"  {nb:>4} {str(res_e1[nb]['directed']['success']):>10} {str(res_e1[nb]['blind']['success']):>10}")

# ---- E2 v_M-σ_ξ 等效性 ----
print("\n[E2] v_M-σ_ξ 等效性: 元级更新频率(左) vs 探索噪声σ(右) 对回报稳态涨落 r*")
rstar = {}
for f_m in [1, 5, 20]:
    H = run_condition(4, "directed", wm_update_every=f_m, sigma=0.0, seed=300)
    rstar[f"freq={f_m}"] = float(np.std(H["ret"][-20:]))
for sg in [0.0, 0.1, 0.2]:
    H = run_condition(4, "directed", wm_update_every=20, sigma=sg, seed=400)
    rstar[f"sigma={sg}"] = float(np.std(H["ret"][-20:]))
for k, v in rstar.items():
    print(f"  {k:>10}: r*={v:.4f}")
print("  → 若提高元级频率与增大σ产生同向的 r* 增大, 即 v_M-σ_ξ 等效性的 RL 版证据")

# ---- E3 可约性 ----
print("\n[E3] 可约性检验: 廉价替代(单步冻结外推) vs full rollout, 误差随 k")
rng = np.random.default_rng(9)
env = GridWorld(4, rng)
policy = DEQPolicy().to(DEV)
popt = torch.optim.AdamW(policy.parameters(), lr=3e-3)
wm = WorldModel().to(DEV)
wopt = torch.optim.AdamW(wm.parameters(), lr=3e-3)
replay = []
for t in range(30):
    eps, _, _ = run_episodes(policy, env, 12, rng)
    replay.extend(eps)
    load_episodes(wm, replay, wopt, epochs=2)
    reinforce_step(policy, eps, popt)
starts = [onehot(env.reset()) for _ in range(32)]
print(f"  {'k':>3} {'|cheap-true|':>13} {'|rollout-true|':>15} {'ρ_eff':>7}")
for k in [1, 2, 4, 6, 8]:
    cheap = 0.0
    for s0 in starts:
        st = torch.from_numpy(s0).unsqueeze(0).to(DEV)
        a = int(policy(st)[0].argmax())
        aoh = np.zeros(4, np.float32); aoh[a] = 1
        aoh_t = torch.from_numpy(aoh).unsqueeze(0).to(DEV)
        _, r1, _ = wm(st, aoh_t)
        cheap += abs(float(r1) * k)          # 单步冻结外推: 假设状态不动
    cheap /= len(starts)
    full = float(np.mean(wm_rollout(wm, policy, starts, k)))
    tr = true_return_from(policy, env, starts, k)
    print(f"  {k:>3} {abs(cheap - tr):>13.4f} {abs(full - tr):>15.4f} {abs(cheap - tr) / max(abs(full - tr), 1e-6):>7.2f}")
print("  → ρ_eff 随 k 增大 = Sim 不可约的操作性证据 (廉价替代误差发散更快)")

