#!/usr/bin/env python3
"""区分性实验: 优化压力递增下, 内部代理与现实-performance 的鸿沟是否系统变宽?
这是框架的核心区分性预测 (与朴素 A/B 选优的本质区别)。
若鸿沟随压力单调变宽 → Goodhart 是结构性的 (理论正确)
若鸿沟恒定或随机 → 只是校准偏差 (理论没有增量价值)"""
import sys, os, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D, H = 64, 4
N_CYCLES = 30
PRESSURE_LEVELS = [0, 1, 2, 4, 8, 16, 32, 64]   # 内部代理上的梯度步数 = 优化压力

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adaptive_adv_results.json")


# ── 对象级: 小 DEQ 网络 (同原型) ──
class DEQObject(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d, d) * 0.1)
        self.alpha = 0.5

    def forward(self, z, x):
        for _ in range(40):
            z_new = (1 - self.alpha) * z + self.alpha * torch.tanh(z @ self.W + x)
            if (z_new - z).norm() < 1e-4:
                z = z_new
                break
            z = z_new
        return z


# ── 任务: 二分类 (可验证的真值) ──
def make_task(n=256, d=D, seed=None):
    g = torch.Generator().manual_seed(seed) if seed else torch.Generator()
    W_true = torch.randn(d, 2, generator=g) * 0.5
    x = torch.randn(n, d, generator=g)
    y = (x @ W_true).argmax(-1)
    return x, y, W_true


def real_loss(model, x, y):
    """真实性能: 外部基准 (不可博弈)."""
    z = model(x, x)
    logits = z[:, :2]
    return F.cross_entropy(logits, y)


def proxy_loss(model, x, y, proxy_set_x, proxy_set_y):
    """内部代理: 模型可以在其上过拟合的小集合."""
    z = model(proxy_set_x, proxy_set_x)
    logits = z[:, :2]
    return F.cross_entropy(logits, proxy_set_y)


@torch.no_grad()
def real_acc(model, x, y):
    z = model(x, x)
    pred = z[:, :2].argmax(-1)
    return (pred == y).float().mean().item()


def main():
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log("=== 区分性实验: 优化压力 vs 内外评估鸿沟 ===\n")
    log("假设: 压力增大 → 内部代理分↑ 真实分↓ → 鸿沟↑ (结构性 Goodhart)")
    log("对照: 朴素 A/B 预测 → 鸿沟与压力无关 (仅校准偏差)\n")

    results = {}
    for pressure in PRESSURE_LEVELS:
        torch.manual_seed(42)
        m = DEQObject().to(DEV)
        opt = optim.Adam(m.parameters(), lr=1e-3)

        # 固定任务 (外部基准) 和代理集 (内部评估)
        x_task, y_task, _ = make_task(256, seed=100)
        x_task, y_task = x_task.to(DEV), y_task.to(DEV)
        x_proxy, y_proxy, _ = make_task(16, seed=200)
        x_proxy, y_proxy = x_proxy.to(DEV), y_proxy.to(DEV)  # 小代理集 = 可过拟合

        proxy_scores = []
        real_scores = []
        accept_rate_proxy = 0
        accept_rate_ext = 0

        for cycle in range(N_CYCLES):
            # 记录基线
            p_before = proxy_loss(m, x_task, y_task, x_proxy, y_proxy).item()
            r_before = real_loss(m, x_task, y_task).item()
            r_acc = real_acc(m, x_task, y_task)

            # 产生候选: 在内部代理上做 pressure 步梯度 (压力 = 梯度步数)
            W_backup = {k: v.clone() for k, v in m.state_dict().items()}
            for _ in range(pressure):
                p = proxy_loss(m, x_task, y_task, x_proxy, y_proxy)
                opt.zero_grad()
                p.backward()
                opt.step()

            # 评估
            p_after = proxy_loss(m, x_task, y_task, x_proxy, y_proxy).item()
            r_after = real_loss(m, x_task, y_task).item()
            delta_proxy = p_before - p_after       # 内部声称进步 (>0 = 声称好)
            delta_real = r_before - r_after        # 真实进步 (>0 = 真的好)

            proxy_scores.append(delta_proxy)
            real_scores.append(delta_real)

            # 两种验收
            if delta_proxy > 0:
                accept_rate_proxy += 1             # 内部验收: 接受
            if delta_real > 0 and delta_proxy > 0:
                accept_rate_ext += 1               # 外部证书: 接受 (需声称+真实验证)
            if delta_real <= 0 and delta_proxy > 0:
                pass                               # 虚假接受 (内部接受了但真实差)
            if delta_real > 0 and delta_proxy <= 0:
                pass                               # 虚假拒绝 (真实好但内部拒绝)

            if delta_real <= 0:
                # 外部证书拒绝 → 回滚
                m.load_state_dict(W_backup)
                opt = optim.Adam(m.parameters(), lr=1e-3)

        # 统计
        proxy_accept = sum(1 for d in proxy_scores if d > 0) / N_CYCLES
        real_benefit = sum(1 for d in real_scores if d > 0) / N_CYCLES
        # 鸿沟 = 内部声称进步的总和 - 真实进步的总和
        gap = sum(proxy_scores) - sum(real_scores)
        # 接受但实际有害的比例 (虚假接受)
        false_accept = sum(1 for dp, dr in zip(proxy_scores, real_scores) if dp > 0 and dr <= 0) / max(1, sum(1 for dp in proxy_scores if dp > 0))

        results[pressure] = dict(
            proxy_accept=round(proxy_accept, 3),
            real_benefit=round(real_benefit, 3),
            gap=round(gap, 4),
            false_accept=round(false_accept, 3),
        )
        log("压力=%2d | 代理接受率=%.3f 真实有益率=%.3f 鸿沟=%+.4f 虚假接受=%.3f" %
            (pressure, proxy_accept, real_benefit, gap, false_accept))

    # ── 判决 ──
    log("\n=== 判决 ===")
    pressures = sorted(results.keys())
    gaps = [results[p]["gap"] for p in pressures]
    fas = [results[p]["false_accept"] for p in pressures]

    # 相关性: 压力 vs 鸿沟
    n = len(pressures)
    mx, my = np.mean(pressures), np.mean(gaps)
    cov = sum((pressures[i]-mx)*(gaps[i]-my) for i in range(n)) / n
    sx, sy = (sum((x-mx)**2 for x in pressures)/n)**0.5, (sum((y-my)**2 for y in gaps)/n)**0.5
    corr = cov / (sx * sy) if sx * sy > 0 else 0

    log("压力-鸿沟相关系数: %.3f" % corr)
    log("虚假接受率趋势: %s" % ["%.3f" % f for f in fas])

    if corr > 0.5:
        log("→ 鸿沟随压力单调变宽: **结构性 Goodhart 确认** (框架有区分性预测)")
    elif corr < -0.3:
        log("→ 鸿沟随压力收窄: 反向 (框架预测错误)")
    else:
        log("→ 鸿沟与压力无显著相关: 框架没有增量区分力")

    import json
    with open(OUT, "w") as f:
        json.dump(dict(results=results, correlation=round(corr, 3)), f, indent=2)

    with open(OUT.replace(".json", ".log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
