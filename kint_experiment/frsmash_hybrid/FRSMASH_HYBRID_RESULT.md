# FRSMASH-Hybrid (~80M): hybrid 注意力嫁接到 FRSMASH — 不提升 (诚实负例)

把 loop4-8 验证的 hybrid 成果(局部窗口 softmax + GLA)嫁接到 FRSMASH v3.6 上:
FRSMASH 已有 GLA recall(全局线性) + SSM 骨干 + SlowMemory, 缺的正是 hybrid 里"局部窗口 softmax 精度"
那半。新增 x_local = LocalWindowAttn(Triton O(T·W) 滑动窗口), 加进 Gated Fusion。
预期: 补上短程精度 → ppl 提升。

## 结果 (~80M, d512/L8, seq512, minimind pretrain)

| 配置 | 参数 | val ppl | 速度 |
|------|------|---------|------|
| vanilla FRSMASH | 80.3M | **37.07** (1500步) / 52.92 (800步) | 80k tok/s |
| FRSMASH-Hybrid | 81.3M | 36.94 (1500步) / 52.32 (800步) | 83k tok/s |
| **差距** | +1M | **-0.4% / -1.1%(噪声级)** | +4% |

**hybrid 分支在 FRSMASH 上 ppl 只好 0.4-1%(噪声内), 速度略快 4%。没有真实质量提升。**

## 诚实裁定: hybrid 洞察不迁移到 FRSMASH
### ❌ hybrid 的"局部窗口 softmax 补短程精度"在 FRSMASH 上是冗余的
- loop4-8 里 hybrid 之所以胜 GLA, 是因为**纯 GLA 短程精度弱**, 局部 softmax 补上了这个短板。
- **FRSMASH 不是纯 GLA**——它有 SSM 骨干 + SlowMemory(线性递归记忆)+ GLA recall 三路,
  **短程精度早被 SSM+SlowMemory 覆盖了**。再加局部 softmax = 重复造轮子, 无新增益。
- 这印证了 loop4-8 的边界: **hybrid 注意力的价值是"修纯 GLA 的短程短板", 对本来短程就强的架构无效。**

### ✅ 唯一正点: Triton kernel 没拖慢(工程成立)
hybrid 的局部分支走 Triton O(T·W) kernel, **没让训练变慢**(83k vs 80k tok/s, 甚至略快)——
loop7 的融合 kernel 工程在真模型上也成立。但这是工程, 不是质量。

## 与 loop4-8 的一致性(边界清晰化)
| 模型 | 短程机制 | hybrid 局部softmax 加上去 |
|------|----------|--------------------------|
| 纯 GLA(loop4) | 弱 | ✅ ppl 提升(补短板) |
| FRSMASH(本实验) | 强(SSM+SlowMemory) | ❌ 冗余, 无提升 |

**hybrid 注意力 = "纯线性注意力短程精度补丁", 只对短程弱的架构有用。FRSMASH 不需要它。**

## 对"100M 实验"的诚实结论
> 把 hybrid 嫁接到 80M FRSMASH, **没拿到质量提升**(噪声级 0.4-1%)。这是诚实的负迁移:
> hybrid 的价值是修纯 GLA 的短板, FRSMASH 的多机制设计早已覆盖短程, hybrid 是冗余。
> 唯一成立的是 loop7 的 Triton kernel 工程(没拖慢)。**FRSMASH 本身(d512/L8/80M, ppl 37)就是这套研究里
> 最强的真模型; hybrid 没能再加分。**

## 产出
- frsmash_hybrid.py (FRSMASHHybrid = FRSMASHv36 + LocalWindowAttn(Triton) 分支)
- train_frsmash_hybrid.py
- 数据见上表(seq512, 800/1500 步)
