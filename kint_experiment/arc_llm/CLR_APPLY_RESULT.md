# 闭环思路(CLR-wd) 应用到 FRSMASH v3.6 与 GLA: 不提升 (诚实)

把第 11 节"救活 SGR"的 CLR 阀门原则,作为 **wd 调度**直接应用到 FRSMASH v3.6 和 GLA 骨干,
看能否提升 val ppl。

## 结果 (minimind pretrain, open_ash_voc, seq=512)

| 骨干 | 参数 | fixed wd | CLR wd | 判定 |
|------|------|----------|--------|------|
| **FRSMASH v3.6** | 60M | **ppl 49.33** | ppl 50.27 | CLR **略差 1.9%** |
| **GLA + dense** | 17M | **ppl 81.21** | ppl 81.42 | 持平(噪声内) |

CLR 的 λ 行为: FRSMASH 上 λ 从 0.01 降到 0.0086, GLA 上降到 0.0034 —— **CLR 在"退场"**,
因为 train/val gap 太弱(+0.03~0.2, 噪声级)触发不了闭环。等价于 CLR < fixed wd 的有效正则,
所以 FRSMASH 上略差。

## 诚实裁定
**CLR 作为 wd 调度,对 FRSMASH 和 GLA 都不提升。** 与第 6 节(vanilla transformer)结论一致:
闭环控制**需要一个强泛化 gap 信号**才有用;LM 预训练 train≈val(gap 噪声级), CLR 无信号可反应,
退化为"比 fixed 还弱的 wd"。**这跟骨干架构(SSM vs GLA vs transformer)无关**——根因是任务(标准 LM
预训练)没有强 gap, 不是模型的问题。

## 与第 11 节"CLR 救活 SGR"的关系(关键区别)
- 第 11 节 CLR 阀门原则能救 SGR, 是因为它控制的是 **SGR 的路由 frac**(一个有明确"该收紧/放松"语义的结构旋钮),
  且 SGR 训练有可观测的 val ppl 变化供闭环反应。
- 这里 CLR 控制 **wd**, wd 本身在 LM 上没有"强 gap 触发"的对象, 闭环无的放矢。
- **结论: CLR 阀门原则的价值取决于"有没有一个能被泛化信号有意义驱动的旋钮"。SGR 路由有, wd 没有。**

## 未测的变体(诚实声明)
FRSMASH 有结构性"遗忘/保留"旋钮(SlowMemory 的 A/B 门、GLA recall 的 g_proj 偏置=8 强保留)。
理论上可对这些做闭环(过拟合就增遗忘、欠拟合就强保留)。但**鉴于 wd 这种全局旋钮都因 LM 弱 gap 而退化**,
预计结构旋钮的闭环**同样会因信号不足而退化**——故未单独实现验证, 不外推。

## 一句话
> 闭环思路作为 **wd 调度**用在 FRSMASH/GLA 上**不提升**(和 vanilla transformer 一样退化);
> 它**只在有"强 gap 信号 + 有语义的旋钮"时**(如 grokking 的 wd、SGR 的路由 frac)才有用。
> LM 预训练两者都不满足, 所以 CLR 在此场景是空转。

## 产出
- clr_apply.py (FRSMASH / GLA 骨干 × fixed / clr, 解耦 wd 闭环)
- log_clrapply_{frsmash,gla}_{fixed,clr}.csv
