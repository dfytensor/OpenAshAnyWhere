# ARC-LLM 的 LLM 指标评测 (用户要求: 测 LLM 指标, 不是内部/玩具指标)

把新架构 ARC-LLM(transformer + SGR)在**真实 LM 指标**上与 vanilla 对齐对比:
- **val perplexity**(核心 LM 指标)
- **next-token top-1 accuracy**(下游式指标)

## 设置
- 数据: minimind pretrain (open_ash_voc, VOCAB=23005), seq=512, 2500 步, wd=0.1
- 三条件:
  - vanilla: 标准 transformer, d_ffn=1024
  - ARC(matched): + SGR, d_ffn=1024 (参数对齐)
  - ARC(eff): + SGR, d_ffn=768 (**少 3.4% 参数**, 测效率)

## 结果 (LLM 指标)

| 架构 | 参数量 | **val ppl** | **top-1 acc** | 门控 ḡ |
|------|--------|-----------|--------------|--------|
| vanilla (d_ffn=1024) | 15.19M | 76.15 | 0.2966 | — |
| ARC matched (d_ffn=1024+SGR) | 15.19M | **75.65** | 0.2974 | 0.93 |
| ARC efficient (d_ffn=768+SGR) | 14.67M (−3.4%) | 77.26 | 0.2954 | 0.94 |

## 诚实裁定 (基于 LLM 指标)
1. **参数对齐时**: ARC(SGR) ppl=75.65 vs vanilla 76.15 —— **0.7% ppl 差, 在 run-to-run 噪声内**;
   top-1 准确度几乎相同(0.2974 vs 0.2966)。**SGR 无质量收益。**
2. **少参数时(−3.4%)**: ARC ppl=77.26, **比 vanilla 还差 1.1%**。低秩路径(rank 64)**无法补偿**
   FFN 缩小损失的质量。**SGR 无效率收益。**
3. 门控全程 ~0.93-0.94(模型几乎总走全秩), 再次印证低秩路径不具竞争力。

## 最终结论 (针对"新 LLM 架构")
> 论文"有效秩是复杂度货币"这一**描述性**发现(grokking 时秩塌缩)是**对的**;
> 但它**不能转成更好的 LLM 架构**。把"秩感知路由(SGR)"塞进前向传播,
> 在 perplexity 和 next-token accuracy 上**既打不败 vanilla, 也不省参数**。
>
> **描述性洞察 ≠ 处方性设计**: "rank matters" 不蕴含 "build rank-routing 会让 LM 更好"。
> 经 LLM 指标公平检验, ARC-LLM 的 SGR **架构上中性偏负**。

## 与之前内部指标测试的一致性
- grokking 上: SGR 不加速/不消 slingshot
- 真实 LM 门控-surprise 相关性: r=+0.10(弱), 无 ppl 收益
- **本测试(LLM 指标)**: 参数对齐 ppl 持平、少参数更差 —— 三者一致指向 SGR 无用。

## 产出
- lm_eval.py (vanilla/arc/arc_eff, val ppl + top1)
- log_lmeval_{vanilla,arc,arc_eff}.csv
