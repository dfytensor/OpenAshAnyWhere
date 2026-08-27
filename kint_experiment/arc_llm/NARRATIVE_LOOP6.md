# Loop 6: 分块 O(T·W) 窗口注意力 — 效率优势未兑现(纯 PyTorch 打不过融合 sdpa)

loop 5 指出 hybrid 效率优势需融合 kernel。本轮先用纯 PyTorch 的分块 O(T·W) 窗口注意力(低风险)
探路, 看能否兑现。

## 速度基准 (推理前向, bs=4, W=128)

| seq | full softmax(O(T²), torch 融合 sdpa) | hybrid(分块 O(T·W) + GLA) |
|-----|--------------------------------------|--------------------------|
| 1024 | 11ms | 74ms |
| 2048 | 37ms | 109ms |
| 4096 | 147ms | 210ms |

**hybrid 在所有长度都更慢。** 分块窗口注意力 FLOPs 是 O(T·W) 更少, 但:
1. Python 按 block 循环(T/Bq=64 次)kernel launch 开销大;
2. 还额外跑了 GLA;
3. torch 的 full softmax 走的是**高度优化的融合 sdpa kernel**, 纯 PyTorch 分块根本打不过。

## 诚实裁定
**O(T·W) 的效率优势在纯 PyTorch 下无法兑现**——必须写**融合 sliding-window attention 的 Triton/CUDA kernel**
才能和 sdpa 同台竞争。即便写了, 也是在和工业级优化的 sdpa 拼速度, 收益不确定。

## 6 轮迭代的真实总账(诚实收尾)
| 轮 | 命题 | 结果 |
|----|------|------|
| 1 | C·V≥K_int=智能条件; SGR/K-Net=更优 LLM | ❌ 证伪 |
| 2 | K 稳定化领先预测 grokking 泛化 | ✅ 4/4(grokking 限定) |
| 3 | K-biomarker 推广到真实 LM 过拟合 | ❌ 证伪 |
| 4 | hybrid 注意力(局部softmax+GLA)Pareto 占优 | ✅ 部分: **严格改进 GLA**(所有长度)、短程胜 softmax |
| 5 | 多尺度窗口修法 → 长程胜 softmax | ❌ 证伪(欠训练) |
| 6 | 分块 O(T·W) 兑现效率优势 | ❌ 纯PyTorch不行, 需融合 kernel |

**真实产出: 唯一活着的、有实证支持的正收益 = loop 4 的 single-hybrid 严格改进 GLA(ppl, 所有长度)。**
其余要么证伪, 要么(效率)需未完成的融合 kernel。

## 关于"继续变革"——诚实的不含糊答复
> 6 轮下来, 真正落到手里的**只有 single-hybrid 改进 GLA 这一个 modest 正收益**。
> 要把它推向"全面变革 softmax", 现在卡在两件**实打实的工程**, 不是再来一轮轻量实验能解决:
> 1. **融合 sliding-window attention 的 Triton kernel**——让 O(T·W) 真的比 O(T²) 快(loop 6 证明纯 PyTorch 不行);
>    这是个和 FlashAttention 同级的工程, 且要和工业优化 sdpa 拼速度, 收益不保证。
> 2. **更大训练预算**——让多尺度/更大窗口的容量训得动(loop 5 证明小预算下多尺度反更差)。
>
> 我可以写那个 Triton kernel(loop 7), 但**诚实说: 它是数百行、易错、和 sdpa 拼速度且不保证赢的硬工程**,
> 不是"再来一轮就变革"。如果你愿意投这个工程, 我接着写; 若只想快速迭代, 那当前已到的真实边界就是:
> **hybrid 改进 GLA(modest 真收益), 其余证伪。**

## 产出
- hybrid_attn.py (新增 fast 分块 O(T·W) LocalSoftmax + --attn bench 速度基准)
- 数据见上表
