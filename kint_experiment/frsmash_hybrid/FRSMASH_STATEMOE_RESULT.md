# FRSMASH-StateMoE (10× 状态): 不提升 (诚实负例, 短/长上下文都验证)

用户提议: FRSMASH v3.6 的 state 用 MoE 模式放大 10×。实现: SlowMemory(单一 d 维递归状态)
→ N=10 独立 LinearSlowMemory 专家 + top-1 路由, 每专家独立递归状态、只被路由到它的 token 更新。
总状态容量 10×, 每 token 只激活 1 专家(稀疏, ~1× 激活算力, 仅慢 5%)。

## 结果 (~82M, d512/L8, minimind pretrain)

| 配置 | seq512 ppl(1500步) | seq1024 ppl(800步) |
|------|--------------------|--------------------|
| vanilla FRSMASH | **37.10** | **52.87** |
| StateMoE(10× 状态) | 37.38 | 52.86 |
| 差距 | +0.8%(噪声) | -0.02%(完全持平) |

**短(seq512)、长(seq1024)上下文均无提升。** State-MoE 在 FRSMASH 上判负。

## 为什么不提升(根因, 诚实)
1. **SlowMemory 是次要分支**: 主状态是 SSM 骨干(8 层) + GLA recall, SlowMemory 不是容量瓶颈。
   放大一个非瓶颈部件 10×, 无效。
2. **top-1 路由碎片化状态**: 每专家状态只整合 ~1/10 的 token, 失去"全序列一致性"。
   10× 容量被 10× 碎片化抵消——长程连贯性反而可能受损。
3. **此规模/上下文 state 非容量受限**: 现有状态已够用, 不是瓶颈。

## 诚实裁定
- **10× 状态容量(MoE) 在 FRSMASH SlowMemory 上无效**, 短长上下文都验证。
- 要让 state-MoE 有用, 需: (a) 套到**主状态**(SSM 骨干)而非次要的 SlowMemory;
  (b) 避免碎片化(所有专家看全部 token 但独立状态 = dense, 10× 算力);
  (c) 在**状态真是瓶颈**的规模/上下文(超长序列、大模型)。
  本设置三条都不满足。

## 与 loop9(hybrid) 的一致性: FRSMASH 的"加法失效"
| 实验 | 加什么 | 结果 |
|------|--------|------|
| loop9 hybrid | 局部窗口 softmax 分支 | ❌ 冗余(SSM+SlowMemory 已覆盖短程) |
| **loop10 StateMoE** | 10× 状态(SlowMemory MoE) | ❌ 非瓶颈 + 碎片化 |

**FRSMASH v3.6 是个均衡设计: SSM 骨干 + SlowMemory + GLA recall 三路已覆盖短/中/长程。
往上加微创新(hybrid 局部softmax、state-MoE)都不提升——因为加的都不是瓶颈。**
瓶颈在 数据/规模/训练步数, 不在架构微调。

## 对"state 放大 10×"的最终诚实答复
> 直接把 SlowMemory 状态 MoE 10× 倍, **不提升**(短长上下文都验证)。根因: SlowMemory 非瓶颈 +
> top-1 路由碎片化。要让"10× 状态"真正有用, 得换主状态(SSM 骨干) + 反碎片化设计 + 状态受限的超长上下文。
> FRSMASH 本身(ppl 37@80M)仍是这套研究里最强的真模型; 这两个加法(hybrid/state-MoE)都加不上去。

## 产出
- frsmash_statemoe.py (StateMoESlow: N 专家 top-1 路由 + masked 递归 + 负载均衡 aux; FRSMASHStateMoE)
- train_statemoe.py
- 数据见上表(seq512 1500步 / seq1024 800步)
