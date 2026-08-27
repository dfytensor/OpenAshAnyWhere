# ARC-LLM 实验结果 (诚实版)

基于 K-Net 论文的 3 条结论设计的新框架 **ARC-LLM**(transformer + Surprise-Gated Rank 路由 + 闭环秩调控 CLR),
在 Modular Addition (p=113) 上做 4 条件对照实验。

## 设计回顾
- **R1 秩原生复杂度**: 用 stable rank 度量/监控
- **R2 CLR 闭环调控器**: λ(t)=λ_max·compress(V)·(1−gen_ema), 泛化后 λ 退场
- **R3 SGR 惊讶门控路由**: 每 token 用门控 s_t=sigmoid(w·x) 在全秩/低秩 FFN 间路由

## 结果

| 条件 | grok 步 | 末段 test | 记忆后最大 V | V>1 尖峰数 |
|------|---------|-----------|-------------|-----------|
| 标准 wd=0.1 | **不 grok** | 0.012 | 4.22 | 10 |
| SGR + 固定 wd | **不 grok** | 0.012 | 4.68 | 11 |
| CLR(无 SGR) | s4600 | 0.999 | 5.03 | 15 |
| **ARC-LLM(SGR+CLR)** | s4200 | 1.000 | 4.73 | 16 |

## 诚实裁定

### ✅ CLR 是有效组件 (复现论文)
固定 weight_decay 的两个条件(vanilla、arc_no_clr)在 12000 步内**都不 grok**;
两个 CLR 条件**都 grok**(~s4200–4600)。闭环调控器取代固定 wd 诱发 grokking——
在 transformer 上再次验证了论文的核心结论。

### ❌ SGR(新颖架构)无 grokking 收益
arc_full vs arc_no_sgr 唯一差别是 SGR 路由:
- grok 步: 4200 vs 4600(相近, 在噪声内)
- slingshot 尖峰数: 16 vs 15(**SGR 不消除 slingshot**)
- 记忆后最大 V: 4.73 vs 5.03(相近)

**SGR 既不加速 grokking, 也不稳定训练。**

### ❌ 门控并未真正"按惊讶路由"
尖峰期 ḡ=0.453 vs 基线 ḡ=0.449——**门控几乎恒定(~0.45), 没有系统性地在惊讶时升高**。
"surprise-gated" 的设计意图未实现: 模型学到一个固定的全秩/低秩混合比, 而非逐 token 自适应路由。
(可能因 modular addition 的 token 同质化, 缺少足够的 per-token 惊讶差异来驱动路由; 真实 LM 数据上待测。)

### ⚠ SGR 唯一可能的边际价值: 推理算力
ḡ~0.47 意味收敛后约 53% 的 FFN 计算走低秩路径(rank 32 vs 512)。若推理期硬路由,
可省约一半 FFN 算力——但**质量是否保持未验证**, 且训练期两条路径都要算, 无训练加速。

## 失败设计的教训
- **原版 gate-penalty(arc_full v1)**: 用 λ·ḡ 当正则 → 退化为强制 ḡ→0 → 模型塌缩到低秩、
  失去容量、不 grok。**直接惩罚门控均值是错的**(平凡最小化)。
- **自由门控(arc_full v2)**: 门控不惩罚、CLR 改压权重 → 能 grok, 但 SGR 中性。

## 一句话
> ARC-LLM 的**有效部分(CLR)就是论文已经验证的闭环调控器**; **新颖部分(SGR 惊讶门控路由)在
> modular addition 上无 measurable 收益**——门控退化成固定混合比, 既不加速也不稳定 grokking。
> 把"闭环"从训练 schedule 搬进前向传播(SGR)这条路, 在本任务上**没走通**。CLR 留在训练侧才有效。

## 边界与下一步
- SGR 可能在**真实 LM**(token 难度差异大)上才有用, 本任务(modular addition, token 同质)不是好的试金石。
- 真正要让 SGR 工作, 门控信号可能需来自**外部惊讶**(如预测熵/残差), 而非纯学习的 w·x。
- 这些均**未测**, 不宜外推。

## 产出
- arc_llm.py (ARC-LLM 模型: transformer + SGR-FFN + stable_rank)
- train_arc.py (CLR 训练 + 4 条件)
- analyze_arc.py / arc_analysis.png
- log_{vanilla, arc_no_sgr, arc_no_clr, arc_full}.csv

---

## 补做：真实 LM 数据上的 SGR 测试 (决定性)

SGR 在 modular addition 上 r≈0(门控恒定)。假设是: 真实文本 token 难度差异大, SGR 才会按惊讶路由。
在 minimind pretrain (open_ash_voc 分词, VOCAB=23005) 上训练 ARC-LLM (15M 参数, d=256/L=4) 1200 步:

### 结果
| 条件 | val ppl | 门控-surprise Pearson r | 高惊讶 ḡ | 低惊讶 ḡ |
|------|---------|------------------------|----------|----------|
| vanilla (无SGR) | **104.4** | — | — | — |
| ARC-LLM (SGR) | 105.7 | **+0.104** | 0.922 | 0.887 |

### 诚实裁定
- **方向性正确**: 真实文本上 SGR 门控**确实**按惊讶弱相关(r=+0.10, 而 modular addition 上 ≈0)。
  这证实了"SGR 需要 token 难度差异才能路由"的假设——modular addition 不是好的试金石。
- **但效应太弱、无收益**:
  - 门控全程 ~0.9(模型几乎总走全秩), 低秩路径(rank 64 vs 全秩 1024)**太弱**, 模型不愿用它;
  - val ppl 没改善(105.7 vs 104.4, 略差);
  - 高/低惊讶的门控差仅 0.034。

### 结论
> SGR 的"惊讶路由"信号**真实存在**(r=+0.10), 但**太弱、且低秩路径不够竞争力**, 导致模型默认全秩、
> 没有 perplexity 收益。要让 SGR 真正有用, 需重新设计: 低秩路径要"够用"(非 rank 64 vs 1024 的悬殊),
> 或门控来自外部惊讶信号(预测熵/残差)而非纯学习。当前 ARC-LLM 的 SGR 部分**仍是中性偏负**。
>
> 一句话: **闭环(CLR)留训练侧有效; 把闭环搬进前向(SGR)即使在真实 LM 上也只得到弱信号、无收益。**

### 产出 (真实 LM)
- arc_real_lm.py (含 gate_per_token + gate-surprise 相关性分析)
- log_real_{vanilla, arc}.csv

