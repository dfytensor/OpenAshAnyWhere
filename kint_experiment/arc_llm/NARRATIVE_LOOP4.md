# Loop 4: 混合注意力(局部 softmax 窗口 ⊕ 全局 GLA) — 首个真实架构改进(部分正例)

C·V 理论三轮榨干后, 换新命题开 loop 4(与 C·V 无关)。种子来自正面发现:
GLA 长上下文碾压 softmax(O(T)), 但短程精细依赖弱于 softmax。
**命题: 局部窗口 softmax(短程精度) + 全局 GLA(长程廉价) 的混合注意力, 跨长度占优。**

## 结果 (GLA 骨干 d256/L6, 同数据, 各长度同步数)

| seq | pure softmax (O(T²)) | pure GLA (O(T)) | **hybrid**(本地softmax W=128 + 全局GLA) | 短/长最优 |
|-----|---------------------|-----------------|-----------------------------------------|----------|
| 256  | 111.39 | 104.83 | **102.49** | hybrid 胜 |
| 1024 | 216.81 | 218.41 | 217.72 | softmax 微胜(hybrid≈) |
| 2048 | **371.32** | 393.86 | 382.15 | softmax 胜(长程) |

## 诚实裁定

### ✅ hybrid 在所有长度上严格改进 GLA
hybrid ppl **每个长度都低于 pure GLA**: 102.49<104.83、217.72<218.41、382.15<393.86。
**混合注意力 = GLA 的严格升级**——加一路局部 softmax 补上了 GLA 短程精度的短板。
这是本研究(自 GLA 之后)**第一个真正的架构正收益**, 且方向全新(与 C·V 无关)。

### ✅ hybrid 在短上下文打败 softmax
seq256: hybrid 102.49 < softmax 111.39。短程上, 局部 softmax 给了精度, GLA 补了全局, 二者合力超过纯 softmax。

### ❌ 但 hybrid 没在长上下文打败 softmax(命题的"Pareto 占优两者"不成立)
seq2048: softmax 371 < hybrid 382。长上下文里, **全 softmax 的全局注意力容量更大**(每个 token 看全程),
而 hybrid 的局部 softmax 窗口 W=128 在 seq2048 只覆盖 6% 上下文, 错失长程精确依赖。
GLA 的递归压缩本身在长程有损(hybrid 也继承了这损失)。所以长程 ppl 上 softmax 仍胜, 代价是 O(T²)。

### ⚠ 效率优势未兑现(实现限制)
hybrid 理论成本 O(T·W+T) 线性, 但我的 LocalSoftmax 是朴素 band-mask(实际仍 O(T²)), 没拿到加速。
要兑现效率需写**融合窗口 softmax kernel**(loop 5 候选)。

## 命题修正(loop 4 → loop 5 种子)
v4 命题"hybrid Pareto 占优两者"**部分成立**: 它**严格占优 GLA、短程胜 softmax**, 但**长程 ppl 输 softmax**。
真正的 Pareto 前沿点是: **hybrid = 比 softmax 便宜(O(T) vs O(T²))但长程 ppl 略差; 比 GLA 好(所有长度)**。
要让它长程也胜 softmax, loop 5 方向:
- **多尺度窗口**(W=64/256/1024 多路局部 softmax + GLA), 让局部覆盖不同尺度, 弥补长程精度;
- 或 **融合 kernel** 兑现 O(T·W) 效率, 在超长(seq 4096+)softmax 不可行时 hybrid 凭效率胜出。

## 关于"继续变革"(本轮的诚实进展)
> loop 4 拿到了**第一个真实架构正收益**: 混合注意力严格改进 GLA(所有长度)、短程胜 softmax。
> 不是"惊天变革", 是**一步实打实的、方向全新的前进**——而且它自然指出了 loop 5(多尺度窗口 + 融合 kernel)。
> 这正是 theory↔experiment 迭代的正确节奏: 每轮要么证伪、要么前进一步, 而不是空喊变革。
> C·V 死了, 但"更好的注意力"这条路活着——loop 4 证明了。

## 产出
- hybrid_attn.py (LocalSoftmax 窗口 band-mask + GlaAttn, 三条件 softmax/gla/hybrid)
- 数据见上表(三长度 × 三注意力)
