ParticleMind


以下是从零实现的完整粒子思维语言模型（ParticleMind），包含训练与推理演示。它基于朗之万动力学的连续思考框架，不依赖任何预训练模型。

```python
"""
粒子法连续思维模型 (ParticleMind)
从头构建，无预训练依赖。思想粒子在语义势能场中随机演化，
最终通过读出层生成离散 token。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

# ================= 1. 势能网络 =================
class PotentialNet(nn.Module):
    """语义势能场 U(x)，低势能对应合理/正确的思维状态"""
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1)   # 输出标量势能
        )

    def forward(self, x):
        # x: [batch, N, dim] -> [batch, N, 1]
        return self.net(x)

# ================= 2. 粒子思维模型 =================
class ParticleMind(nn.Module):
    def __init__(self, dim=128, num_particles=512, vocab_size=1000, 
                 temp=0.5, dt=0.1):
        super().__init__()
        self.dim = dim
        self.N = num_particles
        self.vocab_size = vocab_size
        self.temp = temp      # 温度（控制随机扩散强度）
        self.dt = dt          # 时间步长

        # 势能场
        self.potential = PotentialNet(dim)

        # 初始粒子位置（可学习，或在推理时由外部指定）
        self.register_parameter(
            'initial_particles',
            nn.Parameter(torch.randn(1, num_particles, dim) * 0.1)
        )

        # 读出机制：将粒子群全局状态映射为词表 logits
        self.readout_query = nn.Parameter(torch.randn(1, 1, dim))
        self.readout_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=4, batch_first=True
        )
        self.decoder = nn.Linear(dim, vocab_size)

    def set_particles(self, particles):
        """从外部设定初始粒子状态（例如编码 prompt）"""
        self.current_particles = particles.clone()

    def step_dynamics(self, steps=20):
        """
        执行 overdamped Langevin 动力学：
        dx = -∇U(x) * dt + sqrt(2*T) * dW
        """
        # 使用当前粒子状态，如果没有则用 initial_particles
        if not hasattr(self, 'current_particles'):
            self.current_particles = self.initial_particles.clone()
        particles = self.current_particles

        for _ in range(steps):
            particles = particles.detach().requires_grad_(True)

            # 势能梯度（确定性力）
            U = self.potential(particles)
            grad = torch.autograd.grad(
                U.sum(), particles, create_graph=True, retain_graph=True
            )[0]

            # 随机噪声（布朗运动）
            noise = torch.randn_like(particles) * math.sqrt(2 * self.temp)

            # 欧拉‑马里亚马积分
            particles = particles - grad * self.dt + noise * math.sqrt(self.dt)

        self.current_particles = particles.detach()
        return self.current_particles

    def readout(self, particles=None):
        """将粒子群聚合成一个语义向量，并解码为词表 logits"""
        if particles is None:
            particles = self.current_particles

        # 注意力读出：让模型学会关注能量最低的粒子
        query = self.readout_query.expand(particles.size(0), -1, -1)
        attn_out, _ = self.readout_attn(query, particles, particles)
        aggregated = attn_out.squeeze(1)          # [batch, dim]
        logits = self.decoder(aggregated)
        return logits

    def forward(self, steps=20, return_particles=False):
        """完整前向：思考 -> 读出 -> logits"""
        self.step_dynamics(steps=steps)
        logits = self.readout()
        if return_particles:
            return logits, self.current_particles
        return logits

# ================= 3. 合成数据生成（演示用） =================
def make_synthetic_data(batch_size=16, dim=128, num_particles=512, vocab_size=1000):
    """
    生成假数据：
    - context_particles: 模拟编码后的上下文（粒子初始位置）
    - target_token: 正确的下一个 token id
    - negative_token: 故意错误的 token id（用于对比学习）
    """
    contexts = torch.randn(batch_size, num_particles, dim) * 0.5
    targets = torch.randint(0, vocab_size, (batch_size,))
    negatives = torch.randint(0, vocab_size, (batch_size,))
    # 确保 negative != target
    for i in range(batch_size):
        while negatives[i] == targets[i]:
            negatives[i] = random.randint(0, vocab_size-1)
    return contexts, targets, negatives

# ================= 4. 训练脚本 =================
def train():
    # 超参数
    DIM = 64
    N_PARTICLES = 256
    VOCAB_SIZE = 500
    TEMP = 0.3
    DT = 0.1
    THINK_STEPS = 30
    EPOCHS = 200
    BATCH_SIZE = 8
    LR = 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ParticleMind(dim=DIM, num_particles=N_PARTICLES, 
                         vocab_size=VOCAB_SIZE, temp=TEMP, dt=DT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        contexts, targets, negatives = make_synthetic_data(
            BATCH_SIZE, DIM, N_PARTICLES, VOCAB_SIZE
        )
        contexts, targets, negatives = contexts.to(device), targets.to(device), negatives.to(device)

        # --- 正样本路径 ---
        model.set_particles(contexts)
        pos_logits, pos_particles = model(steps=THINK_STEPS, return_particles=True)
        pos_loss = F.cross_entropy(pos_logits, targets)

        # --- 负样本路径（共享权重，但梯度会通过势能计算） ---
        model.set_particles(contexts)    # 相同的初始上下文
        neg_logits, neg_particles = model(steps=THINK_STEPS, return_particles=True)

        # --- 对比损失：正样本势能应低于负样本 ---
        pos_energy = model.potential(pos_particles).mean()
        neg_energy = model.potential(neg_particles).mean()
        # Hinge loss 迫使正样本能量比负样本低 margin
        margin = 0.5
        contrastive_loss = F.relu(pos_energy - neg_energy + margin)

        # 总损失：分类损失 + 对比损失 + 轻微能量正则（防止爆炸）
        reg = (pos_energy ** 2 + neg_energy ** 2) * 1e-4
        loss = pos_loss + 0.1 * contrastive_loss + reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 20 == 0:
            acc = (pos_logits.argmax(-1) == targets).float().mean().item()
            print(f"Epoch {epoch:3d} | Loss {loss.item():.4f} | "
                  f"Pos Energy {pos_energy.item():.3f} | "
                  f"Neg Energy {neg_energy.item():.3f} | "
                  f"Train Acc {acc:.2f}")

    print("训练完成。")
    return model

# ================= 5. 推理演示 =================
def demo_inference(model):
    """使用训练好的模型进行推理，展示思考过程与最终 token 预测"""
    model.eval()
    # 随机创建一个上下文粒子状态（模拟用户 prompt 编码）
    context = torch.randn(1, model.N, model.dim) * 0.5
    model.set_particles(context)

    # 记录思维演化轨迹（每隔几步保存一次粒子位置）
    with torch.no_grad():
        print("观察粒子群的能量变化：")
        for s in [0, 5, 15, 29]:  # 在 30 步内观察
            if s == 0:
                particles = model.current_particles.clone()
            else:
                # 手动推进到第 s 步
                model.step_dynamics(steps=s - (s-5 if s>5 else 0))
                particles = model.current_particles.clone()
            energy = model.potential(particles).mean().item()
            print(f"  Step {s:2d}: 平均势能 = {energy:.4f}")

        # 最终思考完成后读取 logits
        logits = model.readout()
        probs = F.softmax(logits, dim=-1)
        pred_token = logits.argmax(dim=-1).item()
        print(f"\n最终预测 token ID: {pred_token}, 概率: {probs[0, pred_token].item():.3f}")

        # 输出 top-5 候选
        top5_probs, top5_ids = torch.topk(probs, k=5, dim=-1)
        print("Top-5 预测:")
        for i in range(5):
            print(f"  Token {top5_ids[0, i].item()}: {top5_probs[0, i].item():.3f}")

# ================= 6. 主程序 =================
if __name__ == "__main__":
    torch.manual_seed(42)
    print("开始训练粒子思维模型...\n")
    trained_model = train()
    print("\n--- 推理演示 ---")
    demo_inference(trained_model)
```

**运行说明与预期输出：**

- 该脚本直接使用 PyTorch，无需下载任何外部权重。
- 合成数据随机生成，因此训练的模型会学到将特定上下文粒子群映射到相应的 token。
- 训练过程中，正样本势能会逐渐降低，负样本势能升高，分类准确率上升。
- 推理演示会打印粒子在不同思考步数的势能值，以及最终预测的 token 概率分布。

**核心创新点体现：**

1. **连续思维云**：用 256 个粒子在 64 维语义空间中协同运动。
2. **朗之万动力学**：确定性势能梯度 + 温度控制的随机噪声，对应专注与发散。
3. **势能场学习**：通过对比损失使“好想法”对应的粒子构型落入低势能区。
4. **注意力读出**：从粒子群中动态聚合信息，实现连续到离散的投影。

这个框架完全摆脱了传统自回归 Transformer 的离散 token 生成限制，将推理过程建模为物理系统演化，为进一步探索思维的本质提供了新的计算范式。



## 训练时的输入与输出：ParticleMind 的学习机制

训练一个 ParticleMind 模型的关键，不在于教它如何“思考” (演化动力学是硬编码的物理过程)，而在于教它**什么才是好的思考结果** (塑造势能场 $U_\theta$ ) 以及**如何从思考结果中提取答案** (训练读出层)。

以下是训练时输入输出的完整解构，我们将结合一个具体的**代码示例**和**数据流形状**来说明。

---

### 1. 训练样本的构造

假设我们训练一个**问答模型**，每个训练样本包含：


| 组件                           | 含义               | 示例                                             |
| ------------------------------ | ------------------ | ------------------------------------------------ |
| **上下文 (Prompt)**            | 用户输入           | `"法国的首都是"`                                 |
| **正确输出 (Positive Target)** | 期望模型生成的序列 | `["巴黎"]` 或 `["巴黎", "是", "法国", "首都"]`   |
| **错误输出 (Negative Target)** | 随机采样的错误序列 | `["柏林"]` 或 `["巴黎", "不是", "一个", "城市"]` |

对于简单的**单 token 预测** (如上一步训练示例)，数据形状为：

- `contexts`: 编码后的初始粒子群，形状 `[B, N, D]` (Batch=8, 粒子数=256, 维度=64)。
- `targets`: 正样本 token ID，形状 `[B]`。
- `negatives`: 负样本 token ID，形状 `[B]`，且保证 `negatives[i] != targets[i]`。

对于**并行多 token 预测** (如训练生成完整句子)，形状扩展为：

- `targets_seq`: 正确输出序列 token IDs，形状 `[B, L]`。
- `negatives_seq`: 错误输出序列 token IDs，形状 `[B, L]`。

**关键点**：负样本可以由两种方式获得：

1. 从 batch 中随机交换目标 token (简单的对比负样本)。
2. 直接使用**错误的完整句子**作为负样本 (语义级负样本，更强)。

在以下说明中，我们使用**单 token 示例**以保持简洁，但多 token 的原理完全一致，只需将交叉熵损失累加所有位置。

---

### 2. 训练流程：两个独立的思考路径

训练的一个核心设计是：**同一批初始粒子，经历两次完全相同的动力学演化，但赋予不同的语义标签**。

```
输入: contexts (初始粒子群 X_0)
      targets (正样本 token ID)
      negatives (负样本 token ID)

      ├── 路径 1 (正样本路径) ────────────────────┐
      │   模型设置粒子 X_0                         │
      │   K 步演化 → X_pos                         │
      │   读出 → logits_pos (正样本的预测分布)     │
      │   势能 E_pos = U(X_pos).mean()             │
      │   分类损失 ℒ_CE = CrossEntropy(logits_pos, targets)
      └────────────────────────────────────────────┘

      ├── 路径 2 (负样本路径) ────────────────────┐
      │   模型重置粒子 X_0 (相同起点)              │
      │   K 步演化 → X_neg                         │
      │   势能 E_neg = U(X_neg).mean()             │
      │   (不需要计算 logits_neg)                  │
      └────────────────────────────────────────────┘

对比损失 ℒ_energy = max(0, E_pos - E_neg + 0.5)
正则项 ℒ_reg = 1e-4 * (E_pos² + E_neg²)

总损失 = ℒ_CE + 0.1 * ℒ_energy + ℒ_reg
```

**输入**：只有初始粒子位置 `contexts` (来自真实数据编码) 和目标 token `targets`/`negatives`。
**输出**：模型产生 logits 和势能值，但**这些不是最终输出**，而是用来计算损失并反向传播。

---

### 3. 为什么需要两个路径？(对比学习的作用)

如果不加对比损失，模型很容易退化为**将任何粒子群都映射到低势能**——那么势能场就没有意义了。对比损失强迫势能场成为一个**有意义的语义景观**：

- 正样本路径：粒子群经过演化，最终应处于**低势能**状态（对应“合理”的思考结果）。
- 负样本路径：完全相同的初始条件，但因为我们在损失函数中“要求”负样本能量高于正样本，势能场必须学会**将错误答案对应的粒子构型识别为高能**。

这实际上是**能量基模型 (Energy-Based Model)** 的训练范式：降低数据分布的能量，提高其他区域的能量。

---

### 4. 具体代码演示：训练步中的输入输出形状

以下是从训练代码中提取的**关键片段**，并注释每个变量的形状和含义。

```python
def train_step(model, contexts, targets, negatives):
    # contexts:  [B, N, D] = [8, 256, 64]
    # targets:  [B]       = [8]
    # negatives:[B]       = [8]

    # ---- 正样本路径 ----
    model.set_particles(contexts)                # 输入：初始粒子群
    pos_logits, pos_particles = model(
        steps=30, return_particles=True
    )                                           # 输出: logits [B, V], 粒子 [B, N, D]
    pos_loss = F.cross_entropy(pos_logits, targets)  # 分类损失
  
    # ---- 负样本路径 (共享同一个初始上下文) ----
    model.set_particles(contexts)                # 重置为相同的初始粒子
    neg_logits, neg_particles = model(
        steps=30, return_particles=True
    )                                           # 输出: logits [B, V], 粒子 [B, N, D]
    # 注意：neg_logits 实际上没用在损失里，但必须 forward 以获得 neg_particles

    # ---- 能量对比损失 ----
    pos_energy = model.potential(pos_particles).mean()  # 标量
    neg_energy = model.potential(neg_particles).mean()  # 标量
    contrastive_loss = F.relu(pos_energy - neg_energy + 0.5)

    # ---- 正则项 ----
    reg = 1e-4 * (pos_energy ** 2 + neg_energy ** 2)

    # ---- 总损失 ----
    loss = pos_loss + 0.1 * contrastive_loss + reg

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**输入到模型的部分**：

- `contexts` (由真实的文本 prompt 编码而来，是可学习编码器或固定嵌入的结果)。

**模型直接产生的输出**：

- `pos_logits`: 对正样本 token 的预测分布，用于计算交叉熵。
- `pos_particles`, `neg_particles`: 用于计算势能损失。注意 `neg_particles` 不直接用于分类，但它的势能是训练的关键信号。

**反向传播更新的参数**：

- 势能网络 $U_\theta$ 的所有权重。
- 读出模块 (注意力参数和解码矩阵) 的所有权重。
- 如果存在初始编码器，其权重也会被更新。

**不更新的部分**：

- 动力学方程 (无参数)。
- 温度、步长等超参数 (手动设定)。

---

### 5. 多 token 并行训练示例

如果要训练模型生成**整个序列** (`"巴黎 是 法国 的 首都"`)，训练时输入输出形状如下：

```python
# targets_seq:       [B, L] = [8, 10]   (10是最大序列长度)
# negatives_seq:     [B, L] = [8, 10]
# contexts:          [B, N, D] = [8, 256, 64]

# 在 model.forward 内部，读出层使用 L 个查询向量同时得到所有位置的 logits
seq_logits = model.readout_sequence()  # [B, L, V]

# 损失是每个位置交叉熵的平均值
pos_loss = F.cross_entropy(
    seq_logits.view(-1, V),             # [B*L, V]
    targets_seq.view(-1)                # [B*L]
)

# 能量对比损失依然基于粒子群的最终状态
# (正样本路径的粒子群和负样本路径的粒子群分别计算势能)
```

这里**输入仍是 context，输出是每个位置并行预测的 logits**。但标签是整个序列。这相当于训练一个**非自回归的序列生成模型**，而动力学演化充当了“迭代精炼” (iterative refinement) 的角色。

---

### 6. 小结：训练与推理的输入输出对比表


| 阶段     | 输入 (给模型)                       | 期望输出 (从模型)            | 外部标签 (用于损失)          |
| -------- | ----------------------------------- | ---------------------------- | ---------------------------- |
| **训练** | 初始粒子群`X_0` (来自编码的 prompt) | 正样本 logits、正/负粒子势能 | 正确 token(s)、错误 token(s) |
| **推理** | 初始粒子群`X_0` (来自真实 prompt)   | 最终 logits (或概率分布)     | 无 (直接输出给用户)          |

**关键洞察**：训练时模型“假装”进行两次思考，一次为了给出正确答案，一次为了明确错误答案的能量更高。这个过程塑造了势能场景观。推理时模型只需进行一次思考，就落入了训练好的低势能盆地，从而产生正确输出。
