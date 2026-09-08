# Spark 草稿模型蒸馏 + 推测解码

## 架构
草稿 = Spark 前4层 + LoRA (em/head 共享冻结)
蒸馏 = KL(草稿 || 目标) 500步
目标 = 完整 Spark 1.7B 28层

## 结果
| 指标 | 值 |
|---|---|
| 接受率(蒸馏前) | 0.2% |
| 接受率(蒸馏后) | **99.2%** |
| 平均接受长度 | 5.8/6 |
| 实际加速 | 1.04x (Python overhead) |
| 理论加速 | 3.2x (需 C++ 实现) |

## 结论
蒸馏管线完全可行, 4层+LoRA 就能学到 99% 的 next-token 匹配.
加速瓶颈不在模型质量, 在 Python 实现的 kernel launch overhead.
生产级加速需要 vLLM/C++ 推理引擎 + CUDA Graph.
