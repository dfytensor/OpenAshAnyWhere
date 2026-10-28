# V37 异步架构设计: 权重轮换 + 长上下文 (16k–64k)

日期: 2026-10-28 · 前置: V36_E2E_FINDINGS.md (v36-MS 4096 后缀 2.3228 < relay 2.3377 < 零 2.3477)
对标: DeepSeek V4.1-Flash (CED 相位分离 / 6+1·384 MoE 稀疏激活 / Engram 大表下沉 / State Cache 分块落盘)

## 一、目标与不等式

目标: 推理中异步换权重不影响速度 + 省显存; 上下文可携带可挂起。

核心不等式: **可交换字节数 ≤ 传输带宽 × chunk 计算时间**
- Prefill (主战场): 数千 token/chunk 计算时间, PCIe4 ~28GB/s 轻松覆盖
- Decode: 单 token 预算毫秒级 → 热专家 LRU 常驻 + batch decode, 只流式 FFN 专家

两条独立的交换轴:
| 轴 | attention 系 | delta-state 系 |
|---|---|---|
| 权重换入换出 | ✓ (KV 需常驻) | ✓ 同样可行 |
| **上下文携带/挂起** | KV O(L) 不可压 | **O(1) checkpoint (20KB), 任意边界逐位等价** ← 结构垄断 |

## 二、已验证地基 (V36 阶段完成)

1. s0 接力 = 全扫描, 逐位一致 (kernel diff 0.0) → chunk 边界 = 精确交换屏障
2. relay 权重轮换 demo: 23MB 常驻 + 20KB state 交接 (任意长度续跑)
3. v36-MS bank: 4096 后缀 2.3228 < relay-s0 2.3377 (2000 步单调) → bank 路线在长距离兑现
4. 异步专家轮换方案定型: 常驻 = scan 主干 + router + shared expert + 双缓冲; 流式 = routed experts

## 三、V37 架构

```
常驻显存: scan/attention 主干 + router + shared FFN + state (20KB) + 专家双缓冲 ×2
流式区:   routed experts (内容路由, 位置无关 → 交换零质量损失, 与 relay 位置分工失败的本质区别)
调度:     chunk_i 计算 ‖ DMA 预取 chunk_{i+1} 专家 (router 在上一层隐状态就绪即知路由)
屏障:     chunk 边界 = state 检查点 = 权重交换点 (三位一体)
```

组件:
- A. MoE 化: FFN → n expert + top-k 路由 (首版 n=8, k=2); 专家按内容路由 → 无接口失配
- B. 双缓冲: CUDA stream 异步预取; 交换隐藏在 chunk 计算内
- C. 热专家 LRU: decode 期命中率保障 (PowerInfer/ktransformers 同款)
- D. state 检查点: 每 chunk 边界 20KB 落盘 → 断电续跑/多会话挂起/无限上下文
- E. (可选) n-gram Engram 下沉 RAM: 统计表作便宜的长程记忆, 与 20KB state 互补

## 四、实验路线

| 阶段 | 内容 | 通过标准 |
|---|---|---|
| E1 (本夜) | 16k/64k 后缀 NLL: relay 链 vs v36-MS bank (零训练, 结构泛化验证) | v36 ≤ relay 且差距随长度扩大 |
| E2 | v36-MS + 写入门 (已并入 v3) 长跑 2000 步 @4096 → 权重存档 | 门学会稀疏化 (平均 g 下降) 且 NLL 不劣化 |
| E3 | FFN MoE 化 (8选2), 双缓冲开/关对照: tokens/s 与 NLL | NLL Δ=0, prefill 提速 >30% |
| E4 | 16k 训练 (B=1, chunk=128, K=128) + 64k 推理外推 | 后缀 NLL 随长度平坦 |
| E5 | 整模权重轮换 + state 交接 (23MB 级) → 端侧连续推理 demo | 交接零 NLL 损失 |

## 五、风险与对策

- decode 期专家流式延迟 → 热缓存 + 批解码 + 只流 FFN
- 专家负载不均 → 容量因子 + 热专家永久常驻
- 交接处 state 语义漂移 → 同模型延续 (Test A 逐位等价), 不存在 relay 式接口失配
- compile@长L 数值异常 → 长上下文运行禁用 compile (v2 实装教训)
