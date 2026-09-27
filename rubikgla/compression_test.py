#!/usr/bin/env python3
"""压缩证明: DEQ 不动点是否真正压缩了迭代计算?
测试:
  T1 收敛曲线: 不同迭代次数 k 的准确率 — DEQ 应在 k<<Kmax 时达到平台 (压缩 = 少数步即可)
  T2 收敛探针预测力: "是否收敛" 能否预测 "该样本是否正确" (自模型携带真实信息)
  T3 对照: 同深度的非权重绑定网络 (无 DEQ 压缩) — 是否同样早期平台?"""
import sys, os, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math, json

DEV = "cuda" if torch.cuda.is_available() else "cpu"
V = 300
D, H = 128, 4
KMAX = 40
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compression_results.json")
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compression.log")

def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


# ── 模型 ──
class DEQClassifier(nn.Module):
    """DEQ: 权重绑定的迭代精化."""
    def __init__(self, V=V, d=D, H=H, kmax=KMAX, alpha=0.5):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(512, d)
        self.W_deq = nn.Linear(d, d)
        self.alpha = alpha
        self.kmax = kmax
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)

    def deq_step(self, z, x):
        return (1 - self.alpha) * z + self.alpha * torch.tanh(self.W_deq(z) + x)

    def forward(self, ids, k=None):
        B, L = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(L, device=ids.device)).unsqueeze(0)
        x = self.ln(x)
        z = torch.zeros_like(x)
        K = k if k is not None else self.kmax
        for _ in range(K):
            z = self.deq_step(z, x)
        return self.head(self.ln(z))

    def forward_with_trace(self, ids, ks=(1, 2, 4, 8, 16, 32, 40)):
        """返回多个 k 值的 logits."""
        B, L = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(L, device=ids.device)).unsqueeze(0)
        x = self.ln(x)
        z = torch.zeros_like(x)
        results = {}
        ki = 0
        for step in range(1, max(ks) + 1):
            z = self.deq_step(z, x)
            if step in ks:
                results[step] = self.head(self.ln(z))
        return results

    def convergence_status(self, ids, tol=0.01):
        """返回每个样本的 (是否收敛, 收敛迭代数)."""
        B, L = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(L, device=ids.device)).unsqueeze(0)
        x = self.ln(x)
        z = torch.zeros_like(x)
        converged = torch.zeros(B, dtype=torch.bool, device=DEV)
        conv_iter = torch.full((B,), float(KMAX), device=DEV)
        for step in range(1, KMAX + 1):
            z_new = self.deq_step(z, x)
            diff = (z_new - z).norm(dim=-1).mean(dim=-1)  # (B,)
            newly_conv = (~converged) & (diff < tol)
            conv_iter[newly_conv] = step
            converged = converged | newly_conv
            z = z_new
        return converged, conv_iter


class FlatClassifier(nn.Module):
    """对照: 同深度非绑定网络 (无 DEQ 压缩)."""
    def __init__(self, V=V, d=D, H=H, layers=KMAX):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(512, d)
        layer = nn.TransformerEncoderLayer(d, nhead=H, dim_feedforward=4 * d,
                                           batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, num_layers=min(layers, 6))  # 上限 6 层防爆炸
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)

    def forward(self, ids, k=None):
        B, L = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(L, device=ids.device)).unsqueeze(0)
        mask = torch.triu(torch.ones(L, L, device=ids.device, dtype=torch.bool), 1)
        x = self.tf(x, mask=mask)
        return self.head(self.ln(x))


# ── 数据: 字符级语言建模 ──
def make_data(text, train_frac=0.9):
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([c2i[c] for c in text[:200000]])
    split = int(len(data) * train_frac)
    return data[:split], data[split:], len(chars), c2i


def get_lm_batch(data, bs=32, sl=64, device=DEV):
    ix = torch.randint(0, len(data) - sl - 1, (bs,))
    x = torch.stack([data[i:i + sl] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + sl + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def eval_accuracy(m, data, ks=None, sl=64, bs=32):
    """按 k 值测准确率."""
    m.eval()
    ix = torch.randint(0, len(data) - sl - 1, (bs,))
    x = torch.stack([data[i:i + sl] for i in ix]).to(DEV)
    y = torch.stack([data[i + 1:i + sl + 1] for i in ix]).to(DEV)
    if hasattr(m, 'forward_with_trace') and ks:
        results = m.forward_with_trace(x, ks=ks)
        accs = {}
        for k, logits in results.items():
            accs[k] = (logits.argmax(-1) == y).float().mean().item()
        return accs
    else:
        logits = m(x)
        return (logits.argmax(-1) == y).float().mean().item()


def main():
    torch.manual_seed(0)
    # 数据
    text_path = r"F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl"
    text = ""
    with open(text_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            try:
                import json as j
                d = j.loads(line)
                t = d.get("text", "")
                text += t.replace("\n", "")
            except:
                pass
    text = text[:100000]
    train_data, val_data, actual_V, c2i = make_data(text)
    log("语料 %d 字符, 词表 %d, 训练 %d 验证 %d" % (len(text), actual_V, len(train_data), len(val_data)))

    # ── Phase 1: 训练 DEQ 和 Flat ──
    deq = DEQClassifier(V=actual_V).to(DEV)
    flat = FlatClassifier(V=actual_V, layers=KMAX).to(DEV)

    EPOCHS = 800
    opt_d = optim.Adam(deq.parameters(), lr=6e-4)
    opt_f = optim.Adam(flat.parameters(), lr=6e-4)

    log("=== 训练 ===")
    for st in range(EPOCHS):
        for m, opt in [(deq, opt_d), (flat, opt_f)]:
            m.train()
            x, y = get_lm_batch(train_data)
            logits = m(x)
            loss = F.cross_entropy(logits.reshape(-1, actual_V), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        if (st + 1) % 200 == 0:
            deq.eval(); flat.eval()
            with torch.no_grad():
                ld = eval_accuracy(deq, val_data, sl=64)
                lf = eval_accuracy(flat, val_data, sl=64)
            log("st%d DEQ_acc=%.4f Flat_acc=%.4f" % (st + 1, ld, lf))

    # ── T1: 收敛曲线 (DEQ vs Flat, 多 k 值) ──
    log("\n=== T1: 收敛曲线 ===")
    ks = [1, 2, 4, 8, 16, 32, 40]
    deq.eval(); flat.eval()
    with torch.no_grad():
        deq_accs = eval_accuracy(deq, val_data, ks=ks, sl=64)
        flat_acc = eval_accuracy(flat, val_data, sl=64)
    for k in ks:
        log("  k=%2d: DEQ_acc=%.4f" % (k, deq_accs[k]))
    log("  Flat(%d层): acc=%.4f" % (KMAX, flat_acc))

    # ── T2: 收敛探针预测力 ──
    log("\n=== T2: 收敛探针预测力 ===")
    deq.eval()
    ix = torch.randint(0, len(val_data) - 65, (128,))
    x = torch.stack([val_data[i:i + 64] for i in ix]).to(DEV)
    y = torch.stack([val_data[i + 1:i + 65] for i in ix]).to(DEV)
    conv, conv_iter = deq.convergence_status(x)
    with torch.no_grad():
        logits = deq(x)
        correct = (logits.argmax(-1) == y).float().mean(dim=-1)  # (B,)
    conv_np = conv.cpu().numpy().astype(int)
    correct_np = correct.cpu().numpy()
    # 分组
    acc_conv = correct_np[conv_np == 1].mean() if (conv_np == 1).sum() > 0 else -1
    acc_not_conv = correct_np[conv_np == 0].mean() if (conv_np == 0).sum() > 0 else -1
    log("收敛样本: %d/%d, acc=%.4f" % (conv_np.sum(), len(conv_np), acc_conv))
    log("未收敛样本: %d/%d, acc=%.4f" % ((1-conv_np).sum(), len(conv_np), acc_not_conv))
    # 相关性
    import numpy as np
    corr = np.corrcoef(conv_np.astype(float), correct_np)[0, 1] if len(set(conv_np)) > 1 else 0
    log("收敛-正确率相关系数: %.3f" % corr)

    # ── T3: DEQ vs Flat 效率 ──
    log("\n=== T3: 效率对比 ===")
    # DEQ 达到 90% of final acc 的最小 k
    final_acc = deq_accs[max(ks)]
    target = final_acc * 0.95
    min_k = next((k for k in ks if deq_accs[k] >= target), KMAX)
    log("DEQ 95%平台最小 k = %d (Kmax=%d, 压缩率 %.1f%%)" % (min_k, KMAX, 100 * (1 - min_k / KMAX)))
    log("Flat 同深度 (%d 层): acc=%.4f" % (KMAX, flat_acc))

    # 保存
    results = dict(
        deq_convergence={str(k): round(v, 4) for k, v in deq_accs.items()},
        flat_acc=round(flat_acc, 4),
        conv_probe=dict(
            acc_converged=round(acc_conv, 4),
            acc_not_converged=round(acc_not_converged, 4),
            correlation=round(corr, 3),
            n_converged=int(conv_np.sum()),
        ),
        compression=dict(
            min_k_95pct=min_k,
            kmax=KMAX,
            compression_pct=round(100 * (1 - min_k / KMAX), 1),
        ),
    )
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    log("结果已存 " + OUT)
    log("END")


if __name__ == "__main__":
    import torch.optim as optim
    main()
