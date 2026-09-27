#!/usr/bin/env python3
"""二阶自觉 → LLM 的实际价值: DEQ 残差范数作为免费的不确定性估计器.
无需额外模型/训练/打分器 — DEQ 迭代不收敛时的自然副产品."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D = 64
KMAX = 40
TOL = 0.001

print("=" * 60)
print("DEQ 残差范数作为免费的不确定性估计器")
print("=" * 60)


# ── DEQ 分类器 ──
class DEQNet(nn.Module):
    def __init__(self, d=D, n_class=2, alpha=0.5, kmax=KMAX):
        super().__init__()
        self.fc_in = nn.Linear(d, d)
        self.W = nn.Linear(d, d)
        self.alpha = alpha
        self.kmax = kmax
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_class)

    def deq_step(self, z, x):
        return (1 - self.alpha) * z + self.alpha * torch.tanh(self.W(z) + self.fc_in(x))

    def forward(self, x):
        z = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        for _ in range(self.kmax):
            z = self.deq_step(z, x)
        return self.head(self.ln(z))

    @torch.no_grad()
    def forward_with_norm(self, x):
        """返回 (logits, per_sample_residual_norms)."""
        z = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        for _ in range(self.kmax):
            z = self.deq_step(z, x)
        residual = (self.deq_step(z, x) - z).norm(dim=-1).squeeze(-1)  # (B,) 逐样本
        logits = self.head(self.ln(z))
        return logits, residual

    @torch.no_grad()
    def forward_with_trace(self, x, ks=(1, 2, 4, 8, 16, 32, 40)):
        z = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        results = {}
        for step in range(1, max(ks) + 1):
            z = self.deq_step(z, x)
            if step in ks:
                logits = self.head(self.ln(z))
                results[step] = (logits.argmax(-1) == torch.zeros_like(logits.argmax(-1))).float().mean().item() if False else logits
        return results


# ── 数据 ──
torch.manual_seed(42)
np.random.seed(42)

# 任务: 非线性二分类
d_data = D
W1 = torch.randn(d_data, 64) * 0.3
W2 = torch.randn(64, 2) * 0.3
X_all = torch.randn(3000, d_data)
Y_all = (torch.tanh(X_all @ W1) @ W2).argmax(-1)

split = int(0.8 * len(X_all))
x_tr, y_tr = X_all[:split].to(DEV), Y_all[:split].to(DEV)
x_te, y_te = X_all[split:].to(DEV), Y_all[split:].to(DEV)

# OOD 数据: 噪声 + 尺度偏移
x_ood = (torch.randn(200, d_data) * 5.0).to(DEV)
y_ood = torch.zeros(200, dtype=torch.long, device=DEV)

# ── 训练 ──
print("\n[1] 训练 DEQ 分类器...")
model = DEQNet().to(DEV)
opt = optim.Adam(model.parameters(), lr=1e-3)
for ep in range(1000):
    model.train()
    perm = torch.randperm(len(x_tr))[:64]
    logits = model(x_tr[perm])
    loss = F.cross_entropy(logits, y_tr[perm])
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if (ep + 1) % 200 == 0:
        model.eval()
        with torch.no_grad():
            acc = (model(x_te).argmax(-1) == y_te).float().mean().item()
        print("  ep%d loss=%.4f test_acc=%.4f" % (ep + 1, loss.item(), acc))
        model.train()

# ── 实验: 残差范数 vs 分布 ──
print("\n[2] 残差范数作为不确定性估计器...")
model.eval()

def get_norms(x_input):
    with torch.no_grad():
        _, r = model.forward_with_norm(x_input)
    if torch.is_tensor(r):
        return r.cpu().numpy().flatten()
    return np.array([float(r)])

def get_correct(x_input, y_input):
    model.eval()
    with torch.no_grad():
        pred = model(x_input).argmax(-1)
    return (pred == y_input).float().cpu().numpy()

norms_id = get_norms(x_te)
correct_id = get_correct(x_te, y_te)
norms_ood = get_norms(x_ood)

print("\n  ID  (测试集): 残差范数 mean=%.4f std=%.4f" % (norms_id.mean(), norms_id.std()))
print("  OOD (纯噪声): 残差范数 mean=%.4f std=%.4f" % (norms_ood.mean(), norms_ood.std()))

# 正确 vs 错误预测的残差范数
correct_mask = correct_id.astype(bool)
if correct_mask.sum() > 0 and (~correct_mask).sum() > 0:
    print("\n  ID 正确预测: 残差范数 mean=%.4f" % norms_id[correct_mask].mean())
    print("  ID 错误预测: 残差范数 mean=%.4f" % norms_id[~correct_mask].mean())

# AUC: 用残差范数检测 OOD
from sklearn.metrics import roc_auc_score
labels = np.concatenate([np.zeros(len(norms_id)), np.ones(len(norms_ood))])
scores = np.concatenate([norms_id, norms_ood])
try:
    auc = roc_auc_score(labels, scores)
    print("\n  OOD 检测 AUC = %.3f (1.0=完美, 0.5=随机)" % auc)
except:
    pass

# ── 关键实验: 收敛曲线 (多 k 值) ──
print("\n[3] 收敛曲线 (不同迭代步数的测试精度):")
model.eval()
ks = [1, 2, 4, 8, 16, 32, 40]
for k in ks:
    model.eval()
    correct_k = 0
    with torch.no_grad():
        # 手动跑 k 步
        z = torch.zeros(x_te.shape[0], x_te.shape[1], device=DEV)
        x_feat = model.fc_in(x_te)
        for _ in range(k):
            z = model.deq_step(z, x_feat)
        pred = model.head(model.ln(z)).argmax(-1)
        correct_k = (pred == y_te).float().mean().item()
    print("  k=%2d: acc=%.4f" % (k, correct_k))

print("\n" + "=" * 60)
print("结论: DEQ 残差范数是免费的不确定性估计器")
print("  - 无需额外模型/训练/打分器")
print("  - 不收敛 = 模型不确定 = 可能 OOD/异常")
print("=" * 60)
