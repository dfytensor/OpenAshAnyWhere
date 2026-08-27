# ARC-LLM: Adaptive Rank-Controlled LLM

基于 K-Net 论文的实证结论,设计的一个新 LLM 框架。把论文里"闭环复杂度调控"从**训练
schedule**搬进**前向传播**——让模型按每个 token 的"惊讶度"动态分配有效秩。

## 论文给出的三条可操作结论 → ARC-LLM 的三个机制

| 论文结论(经实证) | ARC-LLM 机制 |
|---|---|
| 真正的复杂度度量是**有效秩**(stable rank 47→2.5),不是 C·V | **R1 秩原生复杂度**:C = Σ_l stable_rank(W_l),正则与监控都用秩 |
| Governor 方向反了:维持 C·V 下界是反 grokking;应"先拟合→压缩→泛化即收手" | **R2 闭环秩调控器(CLR)**:λ(t)=λ_max·compress(V)·(1−gen_ema),泛化后 λ 自动退场 |
| 闭环才是解(开环必在"压不够/压过头"间震荡) | **R3 惊讶门控秩路由(SGR)**:把闭环搬进前向——每 token 按惊讶度选高/低秩路径 |

## 核心创新:Surprise-Gated Rank routing (SGR)

每个 transformer 子层,不再用固定秩的 FFN,而是**每 token 动态混合两条路径**:

```
s_t = sigmoid(w_s · x_t)                    # token 惊讶度 ∈ (0,1), 由前向自发算出
y_full = FFN_full(x)                        # 全秩路径 (d_ffn)
y_low  = FFN_low(x)   = (x W_down) W_up     # 低秩路径 (r << d_ffn)
y = s_t · y_full + (1 − s_t) · y_low        # 惊讶→全秩深思; 自信→低秩压缩
```

- **直觉**:模型对"有把握"的 token 用低秩(=压缩=泛化电路),对"意外"的 token 用全秩
  (=记忆=精确拟合)。这把论文的"fit-then-compress"变成**逐 token、逐层的自调度**。
- **可证伪预言**:grokking 后,SGR 的平均门控 s̄ 应**下降**(模型整体变"自信",更多走低秩);
  且惊讶 token 的 gate 显著高于平凡 token。

## 训练目标:Closed-Loop Rank regulator (CLR)

$$\mathcal{L} = V + \lambda(t)\cdot \textstyle\sum_l \text{stable\_rank}(W_l), \quad
\lambda(t)=\lambda_{\max}\cdot\sigma\!\left(\tfrac{V_0\cdot0.3-V}{\tau}\right)\cdot(1-\overline{acc}_{val})$$

- 第一项拟合;第二项压**秩**(不是 L2 范数)。
- `(1−acc_val)` 是闭环:一旦泛化,λ→0,压缩收手(论文证明这是消除 slingshot 的关键)。

## 与论文 K-Net 的区别
| | 论文 K-Net(闭环版) | ARC-LLM |
|---|---|---|
| 复杂度货币 | 权重范数²(L2) | **有效秩**(R1) |
| 闭环对象 | 全局 λ(标量) | 全局 λ **+ 逐 token 门控**(R3) |
| 压缩作用点 | 全体权重均匀 | **按惊讶度分配**(架构级) |
| 退场信号 | test acc EMA | test acc EMA(同) |

## 实验(本目录)
- 任务:Modular Addition a+b mod p (p=113),grokking 标准战场
- 4 条件对照:
  - `vanilla` : 标准 transformer + weight decay
  - `arc_full`: SGR + CLR(完整 ARC-LLM)
  - `arc_no_sgr`: 只 CLR(隔离惊讶门控的贡献)
  - `arc_no_clr`: 只 SGR + 固定 wd(隔离闭环的贡献)
- 假设:ARC-LLM grok 更快更稳;grok 后有效秩塌缩 + 门控均值下降。

## 文件
- arc_llm.py   — ARC 模型(transformer + SGR + stable_rank)
- train_arc.py — CLR 训练 + 4 条件
- analyze_arc.py — grok 曲线 / 秩轨迹 / 门控行为
- DESIGN.md    — 本文件
