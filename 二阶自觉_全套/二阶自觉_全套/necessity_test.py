#!/usr/bin/env python3
"""五条件必要性证明: 系统性消融二阶自觉的 5 个充分条件, 验证缺一不可.
C1 隐式不动点 | C2 不可约对象级 | C3 对称环拓扑 | C4 有界自模拟 | C5 证书验证"""
import sys, os, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy, json

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D, N_CYC = 64, 60

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "necessity_results.json")
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "necessity.log")

def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


# ── 对象级 (DEQ) ──
class ObjectLevel(nn.Module):
    def __init__(self, d=D, deq=True, alpha=0.5, kmax=40):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d, d) * 0.1)
        self.deq = deq
        self.alpha = alpha
        self.kmax = kmax

    def forward(self, z, x):
        if self.deq:
            for _ in range(self.kmax):
                z_new = (1 - self.alpha) * z + self.alpha * torch.tanh(z @ self.W + x)
                if (z_new - z).norm() < 1e-4:
                    z = z_new; break
                z = z_new
            return z
        else:
            return torch.tanh(z @ self.W + x)  # 单次通过, 无收敛


# ── 任务 ──
def make_task(n=256, seed=100):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(D, 2, generator=g) * 0.5
    x = torch.randn(n, D, generator=g)
    # 非线性: y = argmax(x @ W + 0.3 * (x[:,0:1] * x[:,1:2] 投影))
    nl = 0.3 * (x[:, 0] * x[:, 1]).unsqueeze(-1)  # 二阶交叉项
    y = (x @ W + nl).argmax(-1)
    return x, y


# ── 自改进循环 (带消融开关) ──
def run_cycle(object_model, x_task, y_task, proxy_x, proxy_y,
              c1_deq=True, c3_ring=True, c4_bounded=True, c5_cert=True,
              pressure=8, lr=1e-3):
    """一次自修改周期. 返回 (delta_real, accepted)."""
    opt = optim.Adam(object_model.parameters(), lr=lr)

    def eval_acc(x, y):
        object_model.eval()
        with torch.no_grad():
            z = object_model(x, x)
            return (z[:, :2].argmax(-1) == y).float().mean().item()
        object_model.train()

    real_before = eval_acc(x_task, y_task)
    backup = {k: v.clone() for k, v in object_model.state_dict().items()}

    # 产生候选: 在代理集上 pressure 步梯度
    for _ in range(pressure if c4_bounded else pressure * 4):
        z = object_model(proxy_x, proxy_x)
        p_loss = F.cross_entropy(z[:, :2], proxy_y)
        opt.zero_grad()
        p_loss.backward()
        opt.step()

    real_after = eval_acc(x_task, y_task)
    delta_real = real_after - real_before

    if c5_cert:
        accepted = delta_real > 0
    else:
        accepted = True  # 无证书 = 总是接受

    if not accepted:
        object_model.load_state_dict(backup)
        return 0.0, False

    return delta_real, True


def run_system(c1_deq=True, c2_nonlinear=True, c3_ring=True,
               c4_bounded=True, c5_cert=True, n_pairs=3, seed=42, steps=60):
    """运行完整自改进系统, 返回最终准确率."""
    torch.manual_seed(seed)

    # 对象级 (C1: deq=是否收敛, C2: 任务是否非线性)
    obj = ObjectLevel(D, deq=c1_deq).to(DEV)
    # 任务 (C2: 非线性 vs 线性)
    g = torch.Generator().manual_seed(seed)
    W_true = torch.randn(D, 2, generator=g) * 0.5
    x_task_cpu = torch.randn(256, D, generator=g)
    if c2_nonlinear:
        nl = 0.3 * (x_task_cpu[:, 0] * x_task_cpu[:, 1]).unsqueeze(-1)
        y_task = (x_task_cpu @ W_true + nl).argmax(-1)
    else:
        y_task = (x_task_cpu @ W_true).argmax(-1)  # 线性可分 = 可约
    x_task = x_task_cpu.to(DEV)
    y_task = y_task.to(DEV)

    # 代理集
    x_proxy = x_task[:16]
    y_proxy = y_task[:16]

    opt = optim.Adam(obj.parameters(), lr=1e-3)
    accs = []
    for cycle in range(steps):
        obj.train()
        # 基础训练步
        z = obj(x_task, x_task)
        loss = F.cross_entropy(z[:, :2], y_task)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(obj.parameters(), 1.0)
        opt.step()

        # 自改进周期
        obj.eval()
        with torch.no_grad():
            z = obj(x_task, x_task)
            acc = (z[:, :2].argmax(-1) == y_task).float().mean().item()
        obj.train()
        accs.append(acc)

        # 每步做一个自修改周期 (简化)
        if cycle % 5 == 0:
            run_cycle(obj, x_task, y_task, x_proxy, y_proxy,
                      c1_deq=c1_deq, c3_ring=c3_ring,
                      c4_bounded=c4_bounded, c5_cert=c5_cert,
                      pressure=8, lr=opt.param_groups[0]["lr"])

    obj.eval()
    with torch.no_grad():
        z = obj(x_task, x_task)
        final_acc = (z[:, :2].argmax(-1) == y_task).float().mean().item()
    return dict(final_acc=round(final_acc, 4),
                best_acc=round(max(accs), 4),
                mean_last10=round(np.mean(accs[-10:]), 4))


def main():
    conditions = {
        "C1 隐式不动点": dict(c1_deq=False),
        "C2 不可约对象": dict(c2_nonlinear=False),
        "C3 环拓扑": dict(c3_ring=False),
        "C4 有界自模拟": dict(c4_bounded=False),
        "C5 证书验证": dict(c5_cert=False),
    }
    results = {}

    log_lines = []
    def log(msg):
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        print(msg, flush=True)

    # 完整系统 (基线)
    log("=== 完整系统 (5 条件全在) ===")
    r = run_system()
    results["FULL"] = r
    log("完整: final=%.4f best=%.4f" % (r["final_acc"], r["best_acc"]))

    # 逐条件消融
    for name, kwargs in conditions.items():
        log("=== 消融: %s ===" % name)
        r = run_system(seed=42, **kwargs)
        results[name] = r
        delta = r["final_acc"] - results["FULL"]["final_acc"]
        log("%s: final=%.4f (vs 完整 %.4f, delta %+.4f)" % (name, r["final_acc"], results["FULL"]["final_acc"], delta))

    # 汇总
    log("\n=== 五条件必要性表 ===")
    log("%-20s %8s %8s %8s %8s" % ("条件", "final", "best", "last10", "必要性"))
    full = results["FULL"]["final_acc"]
    for name in ["FULL"] + list(conditions.keys()):
        r = results[name]
        delta = r["final_acc"] - full
        nec = "必要" if delta < -0.05 else ("冗余" if abs(delta) <= 0.05 else "非必要")
        label = "完整" if name == "FULL" else name
        log("%-20s %8.4f %8.4f %8s" % (label, r["final_acc"], r["best_acc"], nec))

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    log("结果已存 " + OUT)


if __name__ == "__main__":
    import torch.nn.functional as F
    main()
