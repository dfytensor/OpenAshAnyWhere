# HRC-LLM v0.17 机制验证结果

日期: 2026-09-06 | 后端: ConvASH30 (30M OpenASH) | 语料: minimind pretrain 文本
目录: F:\夸克\hrc_validate

## 阶段1: 纯算法恒等式 (零训练, hrc_m1_4.py)

| # | 机制主张 | 结果 |
|---|---|---|
| M1 | 已见段最小堆 == 全局 top-B 零误差 | ✅ 全通过 (趋势0/0.5/1.0 × B=64/204 全部 100% 重合) |
| M2 | 块内 K_seg 预筛, 聚集场景 K_seg=4 拐点 | ✅ 均匀 k=1 就基本无损; 聚集场景 k>=3 无损 (与文档 k=4 一致) |
| M3 | 禁止块内归一化 (归一化使召回崩溃) | ⚠️ 方向证实 (1.0->0.47), 幅度场景依赖 (未达 6.1x 极端值) |
| M4 | top-p 不能当预算锚点, 必须 top-k | ✅ 证实 (top-p=0.9 的 k 偏离固定预算 >30%) |

## 阶段2: 记忆桥前提 (真实文本, 30M oracle, hrc_stage2.py)

测什么: 长文档(2048 tok)埋 4 个"秘密段", 选择器按分数保留 top-B%,
测秘密段召回。oracle = 知道秘密段位置。

| 策略 | B=5% | 10% | 15% | 25% | 50% |
|---|---|---|---|---|---|
| oracle (上界) | 0.553 | 0.947 | **1.000** | 1.000 | 1.000 |
| surprise (NLL打分) | 0.090 | 0.209 | 0.287 | 0.418 | 0.766 |
| uniform (位置) | 0.029 | 0.086 | 0.119 | 0.221 | 0.512 |
| random | 0.049 | 0.107 | 0.139 | 0.225 | 0.516 |

### 判读 (n=61 文档, 4 needle/文档)

**P1 证据集中度: 部分成立**
- oracle 曲线显示 15% 预算理论上可完整覆盖证据 (1.000) —— 支持 HRC 的 15% 写入预算
- 但 5% 预算 oracle 仅 55% —— 文档默认 5% 时无法保证完整证据 (与 v0.13 提高默认到 15% 一致)

**P2 选择器有效性: 弱成立 (最大瓶颈)**
- surprise(NLL) 在 B=15% 召回 28.7% vs 随机 13.9% —— 2x 优于随机, 但远低于 oracle 100%
- 距离文档硬指标 "stage-1 ≥98% / final ≥95%" 差距一个数量级
- 含义: HRC 的加速前提 (小预算保留) 数据侧可达, 但**成败取决于选择器** ——
  而 30M 上最简单的 surprise 打分远不够。这与文档红线 "从零端到端训练打分器
  无正面证据; 必须强启发式兜底 + oracle 监督" 相互印证

## 结论

1. HRC 的**算法机制层** (堆精确性/预筛拐点/top-k 预算) 全部验证成立
2. HRC 的**数据前提** (证据集中度) 成立: 15% 预算 oracle 可完整覆盖
3. HRC 的**实际瓶颈**在选择器: 廉价信号(surprise)只有 oracle 的 ~30% 能力,
   需训练或更强信号, 否则 15% 预算下丢 71% 证据段
4. 与 CTS 的对比: CTS 的"层级解耦"前提在真实文本上不存在 (被否);
   HRC 的"证据集中"前提存在 (通过), 但它的实现依赖一个尚未被证明
   可学习的选择器 —— 这与文档自身标注的最大风险 (打分器退化) 一致

## 阶段3: oracle 监督训练选择器 (train_selector.py)

问题: 阶段2 显示廉价 surprise 只有 oracle 的 ~30%。文档红线称"从零端到端
训练打分器无正面证据"。本阶段用 oracle 标签监督训练小 MLP 打分器
(段特征 = token embedding 均值 640 + 段 NLL + 位置, 共 642 维)。

验证集召回 (16 文档, 未参与训练):

| 策略 | B=5% | B=10% | B=15% | B=25% | B=50% |
|---|---|---|---|---|---|
| oracle (上界) | 0.641 | 0.969 | 1.000 | 1.000 | 1.000 |
| **supervised** | **0.641** | **0.969** | **1.000** | **1.000** | **1.000** |
| surprise | 0.000 | 0.078 | 0.125 | 0.203 | 0.656 |
| random | 0.031 | 0.109 | 0.156 | 0.219 | 0.531 |

**supervised 打分器完全复刻 oracle**: B=15% 召回 100% >= 文档 95% 门槛。
仅用 62 训练文档 (约 4000 段, 244 正样本) 即收敛 (ep10 后 loss->0)。

诚实边界:
- needle 是合成注入 ("数字XXXX"), 段特征存在可学习的模式捷径;
  真实任务的"关键段"无此模式, 迁移性未证
- 任务=精确召回 (离散可分), 非弥散语义重要性
- 但作为"选择器可学习性"的存在性证明成立: oracle 监督路线
  (文档 v0.16 推荐的课程学习) 原则可行, 打分器不是 HRC 的死结

## 总结论

| 层 | 验证 | 结果 |
|---|---|---|
| 算法机制 | 堆/预筛/预算/归一化 | 全部成立 (M3 幅度场景依赖) |
| 数据前提 | 证据集中度 (15% oracle 覆盖) | 成立 |
| 廉价选择器 | surprise NLL | 不够 (28.7% @ 15%) |
| **supervised 选择器** | **oracle 监督 MLP** | **达标 (100% @ 15%)** |

HRC-LLM 的核心链条在小规模上**全部验证通过**: 剩余风险集中在
"oracle 标签获取成本"(文档 §3.2) 与 "合成->真实的迁移", 不在架构本身。

## 复现

`
python hrc_m1_4.py        # 阶段1 纯算法 (~秒)
python hrc_stage2.py      # 阶段2 真实文本 needle (~1分钟)
python train_selector.py  # 阶段3 supervised 选择器 (~15分钟, CUDA)
`

产物: hrc_stage2.pt / selector_v1.pt

## 阶段4: 端到端 ConvASH30 原型 - 架构不兼容

尝试在 ConvASH30 上实现 HRC-Lite (前8层编码/记忆桥/后8层解码)。
mem_soft 注入后 8 层 -> backward 触发 device-side assert (NaN)。
所有配置 (lr/tanh/缩放/去adapter) 均失败。

根因: ConvASH 层内 cummax + multiplicative gate 对输入分布极敏感,
预训练只见过 token embedding, 不接受连续 soft prompt。backward 梯度
通过这些操作产生不稳定值。

## HRC 验证最终总结

| 层 | 结论 |
|---|---|
| 算法机制 | 全部成立 |
| 数据前提 | 成立 (15% oracle 覆盖) |
| supervised 选择器 | 成立 (B=15% 达 100%) |
| 端到端因果 RNN | 架构不兼容 (soft token 注入 -> NaN) |

与 CTS 共同教训: CTS 和 HRC 都依赖非因果信息流, 需要双向或扩散式
模型作为底座。因果自回归模型在结构上无法支持这类操作。HRC 的正确定位
是需要专门双向/扩散底座的独立架构 (如 Seed Diffusion / Mercury 2)。

## Spark-X2.5-1.7B HRC-Split 原型 (hrc_spark_split.py)

架构: Spark 1.7B 28层劈成 encode(layers 0-13 + LoRA-A) / decode(layers 14-27 + LoRA-B)
em/head 共享, 原参数冻结。可训练 20.3M (LoRA + mem_proj + seg_pool)。

| 模型 | 上下文 | NLL |
|---|---|---|
| Spark 原模型 (无微调) | 全 2048 token | 2.283 |
| HRC-Split (LoRA 微调) | 10 mem + 64 window | 0.075 |

注意: HRC 经过数据微调而基线是零样本, 对比不公平。正确做法需要给基线做同数据 SFT。

## 推理速度 (bench_decode.py)

Spark-X2.5-1.7B decode 延迟 vs 上下文长度:

| ctx | ms/tok | tok/s | vs 2048 |
|---|---|---|---|
| 64 | 39.9 | 25.1 | 1.13x |
| 2048 | 45.2 | 22.1 | 1.00x |

仅 13% 差异. 原因: Spark-X2.5 已内置 GQA(2KV heads) + 21/28层滑窗(512)
+ 1.7B权重主导带宽. KV 不是瓶颈.

HRC 加速需要: 大模型 + 长上下文 + 标准MHA. Spark 这类已优化架构
不需要 HRC. HRC 的正确定位 = 大模型(7B+) + 长文档(128K+) 的专用加速器.

## Meta-ASH (MetaGRU + ConvASH 融合) 快速验证 2026-09-09
- 架构: MetaASHLM 16层×640d, br 4分支全宽(640→640) + cummax + gen_model + MetaGRU 门控(r/z) + 繁衍项 + 内稳态R
- 词表: OpenASHVoc 23005, minimind 800 docs × 256 tok, 3000 steps, lr 3e-4 cosine, BS=8
- | 模型 | 参数 | final loss |
|---|---|---|
| Meta-ASH 完整 (门控+繁衍+内稳态) | 143.3M | **6.29** |
| 消融 (仅 br+cummax+gen_model) | 141.0M | 6.56 |
- 结论: MetaGRU 门控带来 **-0.27 nats** (相对提升 ~4%), 与 MetaRU v2 结论一致(门控救活但不超 GRU 基线)
- 注意: 全宽 br 是为修维度 bug 的简化, 参数量 143M 非 30M; 与 ConvASH30 原版(~2-3 SFT loss)不可直接比
- 文件: meta_ash_v2.py (完整版, ckpt meta_ash_final.pt 617MB), meta_ash_ablation.py (消融)

## Meta-ASH 30M 公平对比 2026-09-09
- 参数化对齐: br/gate 均为 ConvLinearT(k=9,w=64), 与 ConvASH30 相同; 两模型均 29.6M
- 同数据 (minimind 800 docs x 256) 同种子 (42) 同 3000 步 lr 3e-4 cosine BS=8
- | 模型 | 参数 | final loss |
|---|---|---|
| ORIG ConvMaxStateSuper (原版逐行复刻) | 29.6M | 3.657 |
| META MetaMaxState30 (门控+繁衍+内稳态) | 29.6M | **2.870** |
- **结论: Meta 机制 -0.79 nats (21.5% 相对提升)**, 远大于 143M 全宽版的 -0.27
- MetaGRU 门控用 ConvLinearT 参数化 (gate_r/gate_z), 参数等级与原版一致
- 繁衍项 R*h(1-h) 注入 o1 分支 (x0.1), 内稳态 R buffer 用 .data 更新避免 autograd 版本冲突
- 文件: meta_ash_30m.py
- 速查: 143M 全宽版 meta_ash_v2.py (6.29) vs 消融 meta_ash_ablation.py (6.56)

## Meta-ASH 30M 全量训练: 快验增益不可迁移 (负结果, 重要) 2026-09-16
- 配置: 全量 minimind (PT 1.27M 样本 39,697 步 + SFT 28,304 步), 同 ORIG 管线 (lr 1e-3/2e-4 常数), 17h
- 终评 (同 120 批 SFT 评测集): **ORIG ConvASH30 NLL 3.031 vs META MetaASH30 3.627** — META 落后 0.60 nats
- **与快验 (META 2.870 vs ORIG 3.657, -0.79) 完全反转**
- 失效诊断: R 内稳态 buffer 极化 — 16 层 R 全部跑到钳位值 (4.0 或 0.1, 无中间态);
  R=4.0 层繁衍项 R*h(1-h)*0.1 成为恒定噪声注入, R=0.1 层直接死亡;
  门控本身未饱和 (zg 0.21-0.73), gate_r/gate_z 不是问题
- 教训链: 与 MetaRU v2 '内稳态必须关' 结论一致; 快验 3000 步时 R 尚未漂移完成,
  扮演正则角色; 68k 步后极化成毒 — **短程 A/B 增益不能外推到长程训练**
- 混杂因素 (诚实记录): 全量管线 lr 1e-3 常数 (ORIG 配方) vs 快验 lr 3e-4 cosine,
  META 的最优超参可能不同; 未做超参重扫即判负, 但 R 极化是结构性问题, 非超参可完全解释
- 生成质量: META 样本 <|unk|> 串与复读更多, 与 NLL 一致
- 文件: train_meta_ash30.py, eval_meta_vs_orig.py, bench_meta_full.py
- 修复候选 (未跑): 关内稳态 / eta 降 3 个量级 / R 用慢 EMA; 建议先 8k 步中程验证再上全量

## Meta-ASH 30M 损伤定位: SFT 阶段不收敛, PT 阶段不输 (2026-09-16 中程三臂)
- 8k 步 x 3 臂 (全量管线条件 lr 1e-3, midcheck.py): orig 3.999 / **gates-only 4.310** / **asis 3.924**
- **gates-only 反而更差** -> 快验 A/B 的增益来自 R*繁衍项的正则效应, 不是门控本身
- asis 的 R 轨迹: 500 步内即极化到 4.0 并全程钉死 — 但 8k 步仍领先 orig
- 全量日志复核: META PT 终点 ~3.62-3.78 vs ORIG ~3.86 (不输); **SFT 轨迹 3.6-4.2 游走不收敛** vs ORIG 降到 3.03
- **修正诊断**: R 极化在 PT 是良性正则; 低 lr (2e-4) SFT 下 R=4.0 层的恒定噪声注入阻止精细收敛
- 决定性测试 (待跑): pt_full 出发 6k 步 mini-SFT, repro 关闭 vs 已有 repro-on 轨迹 (6k~3.90)
- 战略账: 即使修复, 全量 PT 优势已蒸发 (打平非 -21%), 最好结局是 SFT 追平
