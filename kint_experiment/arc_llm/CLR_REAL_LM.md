# CLR 上真实 LM 预训练: 负面结果 (诚实)

把论文里唯一在 grokking 上验证有效的机制 **CLR(闭环 weight_decay)** 搬到真实 LM 预训练,
测它能否打败固定 wd / cosine schedule。

## 设置
- 模型: 干净 transformer (ARC_LLM, 15M 参数, d=256/L=4), 因 SGR 已证无效故关闭。
- 数据: minimind pretrain (open_ash_voc, VOCAB=23005), seq=256, 1200 步。
- 三条件(全部用 AdamW **解耦** wd, 公平对比):
  - fixed : wd=0.1 恒定
  - cosine: wd 0.1→0 余弦衰减(标准 schedule)
  - clr   : wd=λ(t)=λ_max·compress(V_train)·gap_signal(val−train), 过拟合才强压、gap 消失放松

## 结果

| 条件 | val ppl | λ 均值 | λ 范围 |
|------|---------|--------|--------|
| fixed wd=0.1 | 104.20 | 0.100 | 0.100~0.100 |
| **cosine** | **103.49** | 0.048 | 0.000~0.100 |
| clr | 104.97 | 0.097 | 0.034~0.100 |

## 诚实裁定
- **CLR 没有打败 fixed/cosine**, 且略差(ppl 104.97 vs fixed 104.20 vs cosine 103.49)。
- **CLR 退化成≈恒定 wd**: λ 全程在 0.034~0.100(均值 0.097), 闭环几乎没起作用。
  原因: 真实 LM 预训练的 train/val gap 很小且噪声大(gap 均值仅 +0.07~0.09),
  CLR 的 `gap_signal` 没有足够强的信号去动态调 wd; 而 train loss 一旦低于阈值, `compress` 立即饱和到 1。
- **标准 cosine schedule 反而(边际)最好**——这是社区常识, 此处再次印证。

## 结论 (对论文 CLR 贡献的边界)
> CLR 的优势**局限于 grokking 类任务**(存在戏剧性的 train/val gap 供闭环反应)。
> 在**标准 LM 预训练**(train≈val, 无明显过拟合 gap)上, CLR 退化为≈恒定 wd,
> **不提供收益**, 标准 cosine schedule 仍是(边际)最优。
>
> 这给 K-Net 论文的 CLR 结论划了清晰边界: "闭环调控取代固定 wd 诱发 grokking" 成立,
> 但**不能外推到 "闭环 wd 是更好的通用 LM 训练策略"**。

## 教训
- 闭环控制器需要一个**强信号**才能work。grokking 的 train/val gap 是强信号(从 ~1 到 ~0);
  LM 预训练的 gap 是弱噪声(±0.1), 不足以驱动有意义的闭环。
- 把 toy task 上有效的机制推广到真实任务前, 必须验证"控制信号在真实任务上是否足够强"。

## 产出
- clr_real_lm.py (3 条件, 解耦 wd, 动态 param_group)
- log_clrrl_{fixed,cosine,clr}.csv
