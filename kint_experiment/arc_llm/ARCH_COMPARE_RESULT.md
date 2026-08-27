# 架构对比 (同一 LM 数据/指标, 参数规模相近)

5 种架构在 minimind pretrain (open_ash_voc, VOCAB=23005) 上, seq=256, 1000 步, untied head, lr warmup,
公平比较 val perplexity + next-token top-1。

## 结果

| 排名 | 架构 | 参数量 | **val ppl** | **top1** | 注意力 / FFN |
|------|------|--------|-----------|---------|-------------|
| 🥇 | **gla** | 17.03M | **104.26** | **0.2809** | 门控线性注意力 / dense |
| 2 | gla_sgr | 17.23M | 105.50 | 0.2798 | GLA / SGR |
| 3 | sgr | 16.84M | 112.08 | 0.2715 | softmax / SGR |
| 4 | vanilla | 16.64M | 113.71 | 0.2684 | softmax / dense |
| 5 | moe | 19.79M | 115.88 | 0.2632 | softmax / soft-MoE |

## 诚实裁定

### 1. 注意力类型 > FFN 类型 (主结论)
- **GLA(门控线性注意力)全面胜过 softmax**: ppl 104.26 vs vanilla 113.71 (**好 8.3%**),
  top1 0.281 vs 0.268。在这个规模/序列长度上, 线性注意力的记忆式归纳偏置更有效。
- 这与 FRSMASH(用 GLA-style recall)是个像样的模型一致。

### 2. SGR(ARC-LLM 的新颖部分)仍是中性
- softmax 下: sgr 112.08 vs vanilla 113.71(略好 1.4%, 噪声内)
- GLA 下: gla_sgr 105.50 vs gla 104.26(**略差**, SGR 反而无益)
- ⇒ **SGR 在两种注意力下都≈中性**, 再次印证"秩感知路由无 LLM 收益"。

### 3. soft-MoE 在小规模最差
- moe ppl 115.88(参数最多 19.79M 却最差)。原因: soft-mixture(非稀疏 top-k)+ 1000 步训练不足,
  专家路由没学起来。MoE 通常需要更大规模/更多步才见效——此处不构成 MoE 本身的定论。

### 4. 最佳组合 = GLA + dense FFN
- 不需要 SGR、不需要 MoE, **单纯把 softmax 换成 GLA 就拿到全部增益**。
- 加 SGR 反而(边际)变差 → ARC-LLM 的架构创新在该设置下是负贡献。

## 结论 (架构层面)
> 在这个受控对比里, **注意力类型决定性能排序(GLA > softmax), FFN 变体(SGR/MoE/dense)几乎不影响**。
> ARC-LLM 的 SGR 路由**不是优胜架构**——纯 GLA+dense 更好。这把"基于论文秩洞察造新架构"这条路
> 再次判负: 描述性秩洞察既不能变成更好的 FFN(SGR), 也打不过现成的线性注意力。

## 边界
- 单种子、seq=256、1000 步、~17M 规模。更大规模/更长序列下排序可能变(MoE 通常 scale-favorable)。
- soft-MoE ≠ 稀疏 top-k MoE, 不能代表 MoE 族。
- GLA 在 seq=256 短序列上的优势, 在超长序列/大模型上是否保持, 未测。

## 产出
- arch_compare.py (5 架构: softmax/gla × dense/sgr/moe, untied head, warmup)
- log_arch_{vanilla,sgr,moe,gla,gla_sgr}.csv
