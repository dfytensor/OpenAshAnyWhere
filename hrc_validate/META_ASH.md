# Meta-ASH：MetaGRU 门控注入 ConvASH 的 30M 参数对齐验证报告

> **公开披露版** · 2026-09-09 · 代码：`meta_ash_30m.py` / `meta_ash_v2.py` / `meta_ash_ablation.py`
> 一句话结论：在参数严格对齐（29.6M vs 29.6M）的对照中，向 ConvASH 的
> ConvMaxStateSuper 注入 MetaGRU 门控 + 繁衍项 + 内稳态，
> **final loss 3.657 → 2.870（−0.79 nats，21.5% 相对提升）**。

---

## 1. 背景

### 1.1 ConvASH 底座

ConvASH30 是全 ConvLinear 化的 OpenASH 因果 RNN 语言模型（vocab 23005, 16 层 ×
640d, 不绑头, ~29.6M）。其核心注意模块 **ConvMaxStateSuper** 的全部机制：

- **br[0..3]**：4 条并行分支，每条是 ConvLinearT（时序卷积 k=9 × 中间维 w=64，
  仅 ~704 参数/分支，逐 token 窗口混合，输出 [b,s,H,dh] 逐头布局）；
- **cummax 状态**：对 o2 分支做 `torch.cummax(dim=1)` 得 out4，携带运行最大值状态，
  支持 O(1) 增量推理；
- **固定交互公式**：
  `result = o0*o1 + a1*o1 + a2*o3 + o0*(a3*out4 + o3) + o1*(o2+out4) + o2*out4`
  （a1/a2/a3 为可学习标量门）；
- **head_linear 5 路交互**：`[o0,o1,o2,o3,out4] × 5` 经 `Linear(H*5, H)` 生成 cg，
  `gen = result + cg*out4`。

### 1.2 MetaRU 前史（本仓库 mxgru.py 系列）

| 版本 | 机制 | 结果 |
|---|---|---|
| MetaRU v1 | 繁衍项 + 内稳态直接替换 GRU 更新 | ❌ 失败（差于随机） |
| **MetaRU v2** | **门控注入**（保留 GRU 主干，繁衍项经 gate 注入）+ 关闭内稳态 | ✅ 救活至 GRU 水平，不超基线 |
| MXGRU 四条合并路线 | MetaGRU × XGRU 融合 | ❌ 全部证伪（互斥） |

MetaRU v2 的教训是：**繁衍项 R·h(1−h) 不能替换主干更新，只能作为门控旁路注入**。
Meta-ASH 的设计问题因此是：这个结论能否迁移到 ConvASH 的 cummax 状态体系上，
以及在参数受限时能否产生超出"救活"的正增益。

## 2. 方法

### 2.1 MetaMaxState30：保留全部 + 新增三项

**保留（ConvASH 全系机制，逐行不动）**：br 4 分支、cummax 状态、固定交互公式、
head_linear 5 路交互、alpha1/2/3。

**新增**：

```python
# ① 繁衍项：注入 o1 分支（MetaRU v2 的门控旁路思路, 系数 0.1）
h_sig = sigmoid(o1)
o1 = o1 + R_mean * h_sig * (1 - h_sig) * 0.1

# ② MetaGRU 门控：门以 ConvLinearT 参数化（与 br 同参数等级, 每个 ~704 参数）
rg = sigmoid(gate_r(x))            # [b,s,d] 重置门
zg = sigmoid(gate_z(x))            # [b,s,d] 更新门
out = zg * gen + (1 - zg) * rg * x # 门控混合: gen=原版交互输出

# ③ 内稳态 R：无梯度 buffer（防 autograd 版本冲突, 用 .data 更新）
R += eta * R * (rho - mean(o1));  R.clamp(0.1, 4.0)   # eta=0.02, rho=0.5
```

### 2.2 严格对齐的对照设置

| 控制变量 | 设置 |
|---|---|
| 参数量 | 29.6M vs 29.6M（门用 ConvLinearT 而非 dense Linear 是对齐关键：dense r/z 门会 +13M） |
| 数据 | minimind 800 docs × 256 tok（OpenASHVoc 23005 编码） |
| 训练 | 3000 步, lr 3e-4 cosine→3e-5, BS=8, clip 1.0, wd 0.01 |
| 种子 | torch/np 双 42 |
| 硬件/内核 | 同一 Triton `_ConvLinearFn`（非纯 torch 参考） |

## 3. 结果

### 3.1 主对照（30M，`meta_ash_30m.py`）

| 模型 | 参数 | loss 曲线 | final loss |
|---|---|---|---|
| ORIG ConvMaxStateSuper | 29.6M | 39.50 → 4.97(1k) → 3.67(2k) | 3.657 |
| **META MetaMaxState30** | **29.6M** | 39.42 → 3.88(1k) → 2.92(2k) | **2.870** |

META 从 step 1000 起持续领先，收敛段（2k→3k）仍保持 −0.75 nats 差距，非早期噪声。
ORIG 的 3.657 与 ConvASH30 公开报告的预训练量级一致（3.86），复刻有效。

### 3.2 快速验证与消融（143M 全宽版）

| 模型 | 参数 | final loss | 增益 |
|---|---|---|---|
| 消融（仅 br+cummax+gen_model） | 141.0M | 6.56 | — |
| Meta-ASH 完整 | 143.3M | 6.29 | −0.27 nats（4%） |

### 3.3 跨规模规律

| 参数规模 | 门控增益 | 相对提升 |
|---|---|---|
| 143.3M | −0.27 nats | ~4% |
| **29.6M** | **−0.79 nats** | **~21.5%** |

**参数预算越紧，Meta 门控收益越大。**

## 4. 分析：为什么门控在小参数下更值钱

原版 ConvMaxStateSuper 的输出是一条**硬编码的乘性交互公式**——它的计算结构固定，
可学习自由度只有 a1/a2/3 三个标量和 head_linear。这在参数充裕时是效率优势
（704 参数/分支），在预算受限时则成为表达瓶颈。

MetaGRU 门控本质是给这条固定公式加了一层**零额外大矩阵的自适应旁路**：

```
out = zg * gen + (1 - zg) * rg * x
```

- `zg→1`：走固定交互公式（原版行为）；
- `zg→0`：走重置后的原始输入（类似 GRU 的候选重置）；
- 网络因此可以**逐 token、逐通道地决定"这次用状态交互，还是直通输入"**——
  等效于在不增加 dense 参数的前提下扩充了行为空间。

这与 MetaRU v2"门控救活"一脉相承，但新发现是：在 cummax 状态体系上，
门控不只是救活，而是**把固定公式的表达瓶颈变成了可控的混合增益**，
且该增益随参数预算收紧而放大——对 30M 级端侧/边缘模型有直接意义。

## 5. 工程教训（复现者必读）

1. **内稳态 R 必须用 `.data` 更新**：R 作为 registered buffer 参与前向
   （repro 项）后，任何 in-place 修改都会触发 autograd 版本冲突
   （`tensor at version N; expected 0`）。`with no_grad(): self.R.data += ...` 解决。
2. **纯 torch 参考 ConvLinear 会 OOM**：unfold+einsum 的中间张量
   [b,s,640,64] fp32 × 7 个 CL/层 × 16 层，24GB 卡撑不住。必须用 Triton kernel。
3. **ConvLinearT 的语义是时序卷积不是通道投影**：输入输出同为 [b,s,d]，
   `view(b,s,H,dh)` 是逐头重排。早期实现按 dense Linear 语义写（全宽 d→d）
   导致维度错乱、loss 卡 8.3——那是 bug，不是机制失败。

## 6. 诚实边界

- **单种子**（42）、**单数据集**（800 docs minimind）、**短程训练**（3000 步）。
  增益方向可信（两组独立规模一致），但 21.5% 的精确数字待多种子/更长训练确认。
- **三项机制捆绑验证**：门控+繁衍+内稳态整体 vs 全无。30M 下未做单项消融；
  143M 消融同样是三项全去。繁衍项与内稳态的独立贡献未知（MetaRU v2 经验提示
  内稳态需关闭或弱化，此处 R 被限制在 [0.1,4.0] 且仅以标量均值影响 o1，风险较小）。
- **仅 loss 曲线证据**，无下游任务（困惑度之外的 QA/生成质量）评测。
- ORIG 基线为逐行复刻（与本仓库 ConvASH30 预训练数字吻合），但非同一权重热启动，
  两模型均为从头训练。

## 7. 复现

```bash
python meta_ash_30m.py    # 30M 主对照（双模型串行, ~20 分钟, RTX 4090D）
python meta_ash_v2.py     # 143M 全宽版（~5 分钟）
python meta_ash_ablation.py  # 143M 消融版
```

依赖：torch ≥2.1 CUDA, triton-windows（`conv_linear_triton_train._ConvLinearFn`）,
OpenASHVoc（本仓库 `open_ash_voc_agent.json`, 23005）。

## 8. 时间戳

首次公开推送：GitHub `dfytensor/OpenAshAnyWhere` master，commit 记录见
`git log --pretty="format:%h %ad %s" --date=iso -- .`（服务端时间，30M 对照
结果随 commit `00aaec6` 于 2026-09-09 公开）。
