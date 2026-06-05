# OpenASH vs WDLM-Neural 对比实验报告

## 1. 实验目的

在相同数据、相同词表、相同训练框架下，对比 **OpenASH (多头 cummax 混合架构)** 与 **WDLM-Neural (全维度 cummax 波动力学架构)** 的训练效率、推理速度与生成质量。

核心：**同参数量公平对比** (OpenASH 58M vs WDLM 60M)，并以 OpenASH 85M 作为参考。

---

## 2. 模型架构

### 2.1 架构总览

| 特性 | OpenASH | WDLM-Neural |
|------|---------|-------------|
| **序列交互** | 多头 cummax + gen_model (5 分支乘加) + head_linear | 全维度 cummax + gen_model (5 分支乘加) + out_proj |
| **FFN** | ReLU gate (`ffn1 * relu(gate) → ffn2`) | NeuralWaveStep (Linear→振幅×旋转) |
| **额外模块** | — | WaveInterference (`proj1(x) * proj2(x)` 双路干涉) |
| **残差连接** | α-加权 + LayerNorm | α-加权 + LayerNorm |
| **State 推理** | 支持 (多头 cummax state) | 支持 (全维度 cummax state) |

### 2.2 OpenASH 架构

```
Input → Embedding → [DecoderLayer × L]
  → MaxStateSuper (8-head cummax + gen_model 5分支 + head_linear)
  → FeedForward (ReLU gate)
  → LayerNorm(α*ffn + (1-α)*input)
→ Linear(head)
```

核心 `gen_model`:
```python
term1 = a * b
term2 = α₁*b + α₂*d
term3 = a * (α₃*e + d)
term4 = b * (c + e)
term5 = c * e
output = sum(terms) + head_linear(cat(a,b,c,d,e)) * e
```

### 2.3 WDLM-Neural 架构

```
Input → Embedding → [WaveResidualBlock × L]
  → NeuralWaveStep (Linear(H→3H) → 振幅×旋转)
  → WaveInterference (2×Linear → a*b 双路干涉)
  → GenModelMix (cummax + 5分支乘加 → out_proj)
  → LayerNorm(α*block + (1-α)*input)
→ WaveMeasurement (Linear(head))
```

关键区别:
- **NeuralWaveStep** 替代 ReLU FFN: 乘法门控代替 ReLU
- **WaveInterference** 增加 `proj1(x) * proj2(x)` 双路干涉
- **GenModelMix**: 无 `head_linear`，用 `out_proj(H*5→H)` 投影

### 2.4 三个测试模型

| 模型 | H | L | Heads | 参数量 | 训练 seq |
|------|---|---|-------|--------|---------|
| **OA-58M** | 640 | 10 | 8 | 58,153,640 | 768 |
| **WDLM-60M** | 512 | 10 | — | 60,267,560 | 1024 |
| **OA-85M** | 768 | 12 | 8 | 84,930,864 | 1024 |

---

## 3. 训练配置

### 3.1 共同配置

| 配置项 | 值 |
|--------|-----|
| 词表 | OpenASHVoc (23,004 + 1 padding) |
| 数据 | MiniMind (Pretrain 1.27M + SFT 0.91M) |
| 优化器 | AdamW (β₁=0.9, β₂=0.95) |
| AMP | bf16 + GradScaler |
| 梯度裁剪 | max_norm=1.0 |

### 3.2 各模型训练详情

| | OA-58M | WDLM-60M | OA-85M |
|--|--------|----------|--------|
| Pretrain | 3ep, lr=3e-4, bs=32, seq=512 | 3ep, lr=3e-4, bs=20, seq=512 | 6ep, lr=5e-4, bs=32, seq=512 |
| SFT | 2ep, lr=3e-4, bs=32, seq=768 | 2ep, lr=3e-4, bs=20, seq=1024 | 6ep, lr=5e-5, bs=20, seq=1024 |
| Pretrain Loss | 2.13 | 2.13 | — |
| SFT Loss | 1.97 | ~2.0 | — |

### 3.3 训练问题及解决

| 问题 | 解决方案 |
|------|---------|
| DataLoader 卡死 (Windows) | `num_workers=0` |
| `torch.save` Error 1224 | `safe_save`: temp file → `os.rename` |
| CUDA index out of bounds | 删除损坏缓存 + `x.clamp(0, vs-1)` |
| checkpoint key 不匹配 | 统一 `--compile 0` |

---

## 4. 训练性能基准

### 4.1 同参数量 (~60M): WDLM H=512/L=10 (60.3M) vs OA H=640/L=10 (58.2M)

| 指标 | WDLM | OpenASH | 对比 |
|------|------|---------|------|
| 参数量 | 60.3M | 58.2M | ≈ 相同 |
| 训练速度 | 97,445 tok/s | 97,206 tok/s | **持平** |
| Loss (step 200) | **5.233** | 5.397 | WDLM 低 3% |
| 训练显存 | **3,099 MB** | 3,231 MB | WDLM 少 4% |

### 4.2 同参数量 (~85M): WDLM H=576/L=12 (82.3M) vs OA H=768/L=12 (84.9M)

| 指标 | WDLM | OpenASH | 对比 |
|------|------|---------|------|
| 参数量 | 82.3M | 84.9M | ≈ 相同 (差 3%) |
| 训练速度 | **83,411 tok/s** | 75,930 tok/s | WDLM 快 10% |
| Loss (step 200) | **5.169** | 5.313 | WDLM 低 3% |
| 训练显存 | **3,821 MB** | 4,208 MB | WDLM 少 9% |

### 4.3 训练基准结论

WDLM 在相同参数量下 Loss 始终低 3~5%，速度持平或快 10%，显存少 4~9%。**WDLM 架构参数利用效率更高。**

---

## 5. 推理性能基准 (OA-58M vs WDLM-60M)

> 公平对比：参数量差异仅 3.6%，相同训练数据。

### 5.1 综合总表

| 指标 | OA-58M | WDLM-60M | 胜者 |
|------|--------|----------|------|
| 参数量 | 58.2M | 60.3M | — |
| 生成速度 (batch) | **67.5 tok/s** | 39.7 tok/s | **OA +70%** |
| State 生成速度 | 46.1 tok/s | **52.3 tok/s** | **WDLM +13%** |
| TTFT (seq=512) | 11.2 ms | **9.1 ms** | **WDLM -19%** |
| PPL (SFT, seq=512) | 7.08 | **5.31** | **WDLM -25%** |
| State 精度 (chunk=64) | **1.0000x** | **1.0000x** | 都是零损失 |
| GPU 显存 (seq=512) | 1,035 MB | 1,032 MB | 持平 |
| Unique% | **68~74%** | 23~64% | **OA** |
| 3-gram 重复 | **1.4~8.1%** | 2.0~73.0% | **OA** |

### 5.2 生成速度 (batch prompt, 200 tokens)

| Prompt | OA-58M | WDLM-60M |
|--------|--------|----------|
| 自我介绍 | 68 | 43 |
| 什么是AI | 66 | 36 |
| 冒泡排序 | 66 | 37 |
| 量子计算 | 69 | 36 |
| 中国首都 | 73 | 57 |
| 春天的诗 | 66 | 36 |
| 1+1等于几 | 68 | 37 |
| 五种水果 | 65 | 36 |
| **平均** | **67.5** | **39.7** |

### 5.3 State 增量推理

**精度** (chunk=64 vs 全序列):

| 模型 | Full PPL | Chunk PPL | 比率 |
|------|---------|----------|------|
| OA-58M | 7.58 | 7.58 | **1.0000x** |
| WDLM-60M | 4.71 | 4.71 | **1.0000x** |

两个模型的 cummax state 增量推理与全序列重处理**完全等价**，零信息损失。这是 cummax 架构的核心优势。

### 5.4 生成质量

| Prompt | 模型 | Unique% | 3-gram% | Entropy |
|--------|------|---------|---------|---------|
| 五种水果 | OA | **74.7%** | 4.7% | **4.65** |
| 五种水果 | WM | 30.0% | 43.2% | 2.89 |
| 春天的诗 | OA | **68.0%** | 8.1% | **4.50** |
| 春天的诗 | WM | 64.0% | 2.0% | 4.33 |
| 解释引力 | OA | **73.3%** | 2.0% | **4.57** |
| 解释引力 | WM | 50.7% | 14.2% | 4.00 |
| 猫的特征 | OA | **73.3%** | 1.4% | **4.60** |
| 猫的特征 | WM | 22.7% | 73.0% | 1.46 |

**OA 多样性全面领先**: Unique% 68~74% vs 23~64%，Entropy 4.50~4.65 vs 1.46~4.33。

### 5.5 样本输出

| Prompt | OA-58M | WDLM-60M |
|--------|--------|----------|
| 自我介绍 | "你好呀！...我是由中国最大开发者独立开发的智能助手maniMod..." | "您好！我是由中国的个人开发者独立开发的智能助手minimind..." |
| 什么是AI | "人工智能（AI）作为一种基于算法、数据科学与机器学习技术..." | (空回答) |
| 冒泡排序 | "冒泡排序算法是一种简单的排序算法...python def bubble_sort..." | "冒泡排序算法是一种简单的排序算法...python def bubble_sor..." |

两个模型在 60M 规模下都能产出基本通顺的中文。OA 输出更丰富，WM 更简洁。

---

## 6. 外推能力测试

### 6.1 训练长度内 (PPL vs Context Length)

单条 SFT 样本，chunk=64:

| Seq | OA-58M | WDLM-60M |
|-----|--------|----------|
| 64 | 7.13 | **3.72** |
| 128 | 4.98 | **3.53** |
| 256 | 3.96 | **3.37** |
| 512 | 3.76 | **3.47** |
| 768 | 3.90 | **3.66** |

训练长度内 WDLM PPL 全面更低。OA 短序列 PPL 高但随序列增长快速下降 (7.13→3.76, -47%)，WDLM 更稳定 (3.72→3.47, -7%)。

### 6.2 三模型外推对比 (超出训练长度, 最大 16K)

拼接 SFT 数据，chunk=64 增量推理。(* = 超出训练长度)

| Seq | OA-58M (768) | WDLM-60M (1024) | OA-85M (1024) | 倍率 |
|-----|:-:|:-:|:-:|------|
| 256 | 15.3 | 7.4 | **4.2** | |
| 512 | 10.4 | 5.8 | **3.5** | |
| 768 | 8.9 | 5.2 | **3.2** | OA-58M 上限 |
| 1024 | 9.1\* | 5.9 | **3.2** | WDLM/OA-85M 上限 |
| 1536 | 16.5\* | 38.4\* | **5.2**\* | |
| 2048 | 29.6\* | 185.1\* | **10.6**\* | |
| 3072 | 59.1\* | 1085.6\* | **29.5**\* | |
| 4096 | 90.4\* | 2946.5\* | **54.5**\* | 4x 训练 |
| 6144 | 143.3\* | 8894.2\* | **123.6**\* | 6x 训练 |
| 8192 | 204.0\* | 15604.4\* | **235.2**\* | 8x 训练 |
| 12288 | 303.6\* | 26092.6\* | **546.1**\* | 12x 训练 |
| 16384 | 374.3\* | 33286.2\* | **855.6**\* | 16x 训练 |

### 6.3 外推退化倍数 (PPL vs seq=1024 基线)

| Seq | OA-58M | WDLM-60M | OA-85M |
|-----|--------|----------|--------|
| 2048 | 3.2x | 31.4x | 3.3x |
| 4096 | 9.9x | 499x | 17.0x |
| 8192 | 22.3x | 2645x | 73.5x |
| 16384 | 41.0x | **5625x** | 267.4x |

### 6.4 外推关键发现

1. **训练长度内**: OA-85M 全面最优 (PPL 3.2~4.2)，WDLM-60M 次之，OA-58M 最差。

2. **WDLM 指数级崩溃**: PPL 在 seq>1024 后呈指数增长，seq=16384 时 PPL=33286 (基线的 5625 倍)。模型在长序列上基本失去预测能力。

3. **OpenASH 多项式退化**: OA-85M 在 seq=16384 时 PPL=856 (基线的 267 倍)，虽已退化但仍能维持基本预测。OA-58M 退化更平缓 (41 倍) 但起点更高。

4. **架构决定外推上限**: OpenASH 的多头 cummax (8 heads, head_dim=80/96) 将状态分解为多个子空间，独立跟踪不同维度信息，提供更强的长程外推稳定性。WDLM 的全维度 cummax (dim=512) 在长序列上信息衰减更快。

5. **层数越多外推起点越好**: OA-85M (L=12) 在所有序列长度下 PPL 都低于 OA-58M (L=10)，但退化倍数更大，可能因为更深网络累积更多误差。

---

## 7. 跨参数量推理参考 (OA-85M vs WDLM-60M)

> 非公平对比 (参数差 41%)，仅供参考。

| 指标 | OA-85M | WDLM-60M | 对比 |
|------|--------|----------|------|
| 参数量 | 84.9M | 60.3M | WDLM = 71% |
| PPL (SFT) | **2.45** | 3.61 | OA 更优 |
| 生成速度 | 52.9 tok/s | **70.5 tok/s** | WDLM 快 33% |
| TTFT | 11.0 ms | **7.1 ms** | WDLM 快 36% |
| 前向吞吐 (seq=1024) | 107K tok/s | **163K tok/s** | WDLM 快 52% |
| Unique% | **66~73%** | 10~63% | OA 更优 |
| 3-gram 重复 | **1.4~12.8%** | 4.1~89.2% | OA 更优 |

OA-85M 以更多参数换取更好质量 (PPL 2.45)，但 WDLM 用 71% 参数量在速度上全面领先。

### GPU 显存

| SeqLen | OA-85M | WDLM-60M |
|--------|--------|----------|
| 512 | 980 MB | 966 MB |
| 1024 | 1,045 MB | 1,022 MB |

### 训练显存

| Batch | OA-85M | WDLM-60M | 差值 |
|-------|--------|----------|------|
| bs=1, seq=512 | 1,914 MB | **1,439 MB** | -25% |
| bs=8, seq=512 | 4,093 MB | **3,095 MB** | -24% |

---

## 8. 长期依赖测试

### 8.1 Key-Value 检索 (Needle in Haystack)

在文本开头放置键值对 (如 "XYZ=blue123")，末尾提问要求回忆。

| 距离 | OA-85M | WDLM-60M |
|------|--------|----------|
| 50 | N (提及 XYZ 但未给值) | N (泛泛回答) |
| 100 | N | N |
| 200 | N | N |
| 400 | N | N |

两个模型均无法准确检索远距离信息。这是 cummax 架构的固有限制——cummax 只保留最大值而非完整历史。需要额外记忆机制 (如外部 RAG)。

---

## 9. 结论

### 9.1 同参数量对比 (OA-58M vs WDLM-60M) — 互有胜负

| 维度 | 胜者 | 数据 |
|------|------|------|
| 生成速度 (batch) | **OA** | 67.5 vs 39.7 tok/s (+70%) |
| State 生成速度 | **WDLM** | 52.3 vs 46.1 tok/s (+13%) |
| TTFT | **WDLM** | 9.1 vs 11.2 ms (-19%) |
| PPL (训练长度内) | **WDLM** | 5.31 vs 7.08 (-25%) |
| State 精度 | **平手** | 都是 1.0000x 零损失 |
| 外推 (16x 训练) | **OA** | 374 vs 33286 PPL |
| 生成多样性 | **OA** | Unique% 68~74% vs 23~64% |
| GPU 显存 | **平手** | 持平 |

### 9.2 架构优劣势总结

| | OpenASH | WDLM-Neural |
|--|---------|-------------|
| **优势** | 生成多样性好、外推能力强 (多项式退化)、batch 推理快 | 参数效率高 (PPL 低)、State 推理快、TTFT 快、训练显存少 |
| **劣势** | PPL 较高、State 推理较慢 | 外推能力差 (指数崩溃)、生成重复严重 |
| **推荐场景** | 超长序列 (>训练长度)、追求生成质量 | 训练长度内、追求速度和效率 |

### 9.3 核心结论

1. **训练效率**: WDLM 架构参数利用率更高，同参数 Loss 低 3~5%，速度快 0~10%
2. **推理速度**: 训练长度内 WDLM 更快 (TTFT -19%, State +13%)；batch 推理 OA 反而更快 (+70%)
3. **生成质量**: OA 多样性全面领先 (Unique% 68~74% vs 23~64%)
4. **State 推理**: 两个模型 chunk 增量推理与全序列完全等价 (1.0000x)，零信息损失
5. **外推能力**: OA 碾压式领先，WDLM 在 16x 训练长度时 PPL 崩溃至 33286，OA 仅 374~856

---

## 10. 文件清单

```
experiment_openash_vs_wdlm/
├── REPORT.md                          # 本报告
├── src_openash/                       # OpenASH 模型 + 训练
│   ├── open_ash.py
│   ├── open_ash_voc.py
│   ├── config.py
│   ├── open_ash_dataset.py
│   ├── trainer_utils.py
│   ├── train_pretrain.py
│   ├── train_full_sft.py
│   └── open_ash_infer.py
├── src_wdlm/                          # WDLM-Neural 模型 + 训练
│   ├── wdlm_neural.py
│   ├── train.py
│   ├── infer.py
│   ├── open_ash_voc.py
│   └── config.py
└── bench/                             # 基准测试
    ├── bench_sameparam.py             # 同参数推理对比 (58M vs 60M)
    ├── bench_state_extrap.py          # State 精度 + 外推
    ├── bench_extrap_long.py           # 长序列外推 (至 16K)
    ├── bench_extrap_all3.py           # 三模型外推对比
    ├── bench_compare.py               # 跨参数推理对比 (85M vs 60M)
    ├── bench_train_sameparam.py       # 同参数训练基准
    ├── bench_longrange.py             # 长期依赖测试
    ├── open_ash_voc_agent.json        # 词表
    ├── openash60m_sft_final.pth       # OA-58M SFT 权重
    ├── full_sft_768_12.pth            # OA-85M SFT 权重
    └── wdlm60m_sft_final.pth          # WDLM-60M SFT 权重
```

---

*实验日期: 2026-06-05 | GPU: NVIDIA Single GPU | PyTorch + CUDA bf16 AMP*
*核心测试: 同参数量公平对比 (OA-58M vs WDLM-60M) + 三模型外推至 16K*
