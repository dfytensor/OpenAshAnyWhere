# Loop 5: 多尺度窗口 hybrid — 被证伪 (诚实负例)

loop 4 hybrid 长程输 softmax(因 W=128 只覆盖 seq2048 的 6%)。修法假设:
**多尺度窗口(64/256/1024) + GLA** 让大窗覆盖 50%, 弥补长程精度, 长程打败 softmax。

## 结果 (seq 2048, 同步数)

| 配置 | val ppl | 耗时 |
|------|---------|------|
| softmax | 371.32 | 19s |
| single-hybrid (W=128, loop4) | 382.15 | 35s |
| **multi-hybrid (W=64/256/1024, loop5)** | **397.31** | 58s |

**multi-hybrid 比 single-hybrid 还差, 更输 softmax。** 命题证伪。

## 为什么更差(根因)
1. **参数更多 + 同步数 => 欠训练**: multi 每层 3 个 LocalSoftmax + GLA(4 路注意力),
   参数最多, 300 步训不动, 容量没兑现反成累赘。
2. **朴素 band-mask 每 window 仍 O(T²)**: 3 个窗口 = 3×O(T²) 成本(58s vs softmax 19s),
   理论的 O(T·ΣW) 线性优势**完全没兑现**(没写融合 kernel)。
3. 4 路注意力直接相加、无归一化, 优化更难。

## 诚实裁定
- **多尺度窗口(朴素实现)不解决问题, 反而更差。** 想用更大窗口补长程精度, 在当前训练预算下
  被欠训练吞掉, 且效率没兑现。
- loop 4 的 single-hybrid 仍是 hybrid 线的**最优点**: 严格改进 GLA(所有长度)、短程胜 softmax。
- hybrid 要在长程 ppl 上也胜 softmax, 需**更大训练预算**(让多路注意力训得动)+ **融合 kernel**(兑现 O(T·W))——两者都是实打实的额外工程, 不是一轮能搞定。

## Loop 4-5 综合诚实结论(hybrid attention 这条线)
- **真实收益**: single-hybrid 在所有长度严格改进 GLA(loop 4)——这是确实的、方向全新的正例。
- **未达成**: 长程 ppl 超越 softmax(loop 4 输、loop 5 修法失败)。
- **关键瓶颈**: 效率优势(O(T·W) vs softmax O(T²))**未兑现**(需融合 sliding-window kernel);
  没有这个 kernel, hybrid 在朴素实现下比 softmax 还慢, "更便宜"卖点无法成立。

## 关于"继续变革"的诚实状态(5 轮后)
> 5 轮迭代的真实产出:
> - loop 1-3: C·V 理论彻底证伪(无真实 LM 价值)。
> - loop 4: hybrid 注意力——**第一个真实架构正收益**(严格改进 GLA, 所有长度)。
> - loop 5: 多尺度修法——**证伪**(欠训练 + 效率未兑现)。
>
> 当前手里**唯一活的、有真实价值的线是 single-hybrid**(改进 GLA)。
> 要把它推向"全面占优 softmax", 下一步**必须**做两件实打实的事:
> (a) **写融合 sliding-window attention kernel**(兑现 O(T·W), 让 hybrid 长程比 softmax 便宜);
> (b) **更大训练预算**(让多尺度/更大窗口的容量训得动)。
> 这两个都不是"再来一轮轻量实验", 是真工程投入。诚实说: 没有 (a), hybrid 的效率卖点就是空的。

## 产出
- hybrid_attn.py (新增 attn=multi, --wins 64,256,1024)
- 数据见上表(seq2048 三条件)
