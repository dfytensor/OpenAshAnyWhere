<div align="center">

# OpenASH-85M

**基于累积最大值注意力的无 Softmax 语言模型，支持有状态推理**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

## 命题

Softmax 不是注意力的必要组成。OpenASH 用**累积最大值（`torch.cummax`）**——一种纯比较运算——替代了整个 softmax 注意力机制，并与可学习的多项式混合层结合。该架构：

- 从注意力路径中完全消除指数运算
- 原生携带**状态张量**跨 chunk 传递，实现逐 token O(1) 推理，无需 KV Cache
- 在 85M 参数量级、消费级硬件上，可端到端完成完整 LLM 训练流程（Pretrain → SFT → DPO → GRPO）

这是一个**概念验证**：一种基于序统计量而非概率归一化的替代注意力范式，能够训练出可用的语言模型。

## 架构

```
输入 → 词嵌入 → [解码器层 × 12] → 线性输出头 → Logits
                    │
                    ├─ MaxStateSuper（cummax 注意力，有状态）
                    ├─ 门控 FFN（Swish-Gate 线性层）
                    └─ 加权残差连接 + LayerNorm
```

### MaxStateSuper：基于 Cummax 的注意力

核心机制用以下操作替代 softmax(QK^T)V：

```python
combined = self.combined(x).view(b, s, 4, self.heads, -1)
out, out1, out2, out3 = combined.unbind(2)

out4, _ = torch.cummax(out2, dim=2)          # 累积最大值，不是 softmax

output = term1 + term2 + term3 + term4 + combined  # 可学习的多项式混合
```

输出为受可学习 `alpha` 参数控制的逐元素交互项之和：

- `term1 = a * b`（查询-键乘积）
- `term2 = alpha1 * b + alpha2 * d`（值 + cummax 状态线性混合）
- `term3 = a * (alpha3 * e + d)`（查询-状态门控路径）
- `term4 = b * (c + e)`（键-状态交叉）
- `c * e`（原始值-cummax 残差）
- `head_linear(cat([a,b,c,d,e])) * e`（可学习头投影，以状态加权）

**无指数运算。无注意力矩阵。"注意力权重"隐式地由投影值的运行最大值构成。**

### 有状态推理

生成过程中，前一个 chunk 的 cummax 状态向前传递：

```python
out4, _ = torch.cummax(torch.cat([state, out2], dim=2), dim=2)
state = out4[:, :, -1:]  # 传递到下一个 chunk
```

这意味着每个解码步相对于序列历史是 **O(1)** 的——模型不对历史 token 重新做注意力，而是携带一个压缩状态。

### 模型配置

| 参数 | 值 |
|---|---|
| 词表大小 | 23,004 基础 + 2,095 代理词表 |
| 隐藏维度 | 768 |
| 层数 | 12 |
| 注意力头数 | 8 |
| 总参数量 | **84,930,864（85M）** |
| 最大序列长度（chunk） | 8,192 |
| 权重文件大小 | ~162 MB |

### 代理词表编码

为在紧凑的词表内覆盖大量字符空间，OpenASH 采用二维/三维索引方案：

- **2D 词元**：字符 → `(ts_i, te_j)`——用 `2N` 个基础 token 编码最多 `N²` 个额外字符
- **3D 词元**：字符 → `(rs_i, rc_j, re_k)`——用 `3N` 个基础 token 编码最多 `N³` 个额外字符

分词基于 jieba，遇到未知词时回退到字符级编码。

## 训练流水线

OpenASH 经历了完整的四阶段训练，全部基于 PyTorch 从零实现：

```
阶段 1：预训练 Pretrain    （原始文本上的下一词预测）
阶段 2：SFT                （指令数据的监督微调）
阶段 3：DPO                （直接偏好优化）
阶段 4：GRPO               （组相对策略优化 / 强化学习）
```

每个阶段加载上一阶段的权重。支持 PyTorch DDP 分布式训练、BFloat16 混合精度、梯度累积和检查点断点续训。

### 实现细节

- **优化器**：AdamW + 余弦学习率调度
- **精度**：BFloat16 AMP + GradScaler
- **分布式**：`torchrun` + NCCL 后端，`SkipBatchSampler` 支持断点续训
- **采样策略**：Temperature / Top-K / Top-P / 重复惩罚（可配置）
- **流式输出**：逐 token 实时解码，含代理词表缓冲区管理

## 评测：C-EVAL

在 **C-EVAL**（中文综合考试基准，52 个科目，1,346 道题）上进行 5-shot 评测：

| 模型 | 阶段 | 总体准确率 | 正确 / 总题数 |
|---|---|---|---|
| full_sft_768_12 | SFT | 23.11% | 311 / 1,346 |
| dpo_768_12 | DPO | 22.88% | 308 / 1,346 |
| **grpo_768_12** | **GRPO** | **26.82%** | **361 / 1,346** |

### GRPO 亮点科目

| 科目 | 准确率 |
|---|---|
| 法律职业 | 43.48% |
| 大学物理 | 42.11% |
| 大学化学 | 41.67% |
| 中学地理 | 41.67% |
| 中国语言文学 | 39.13% |
| 税务会计 | 38.77% |
| 高等数学 | 36.84% |

完整结果：[`ceval_results_grpo.json`](ceval_results_grpo.json)、[`ceval_results_dpo.json`](ceval_results_dpo.json)、[`ceval_results_sft.json`](ceval_results_sft.json)。

## 局限性（坦诚说明）

1. **C-EVAL 绝对数值有限。** 85M 参数下 26.82% 是基线，不是有竞争力的结果。本项目的声明不是"打败了谁"，而是"这个架构可以被训练，且产生了有意义的信号"。

2. **非标准分词器。** jieba + 代理词表方案不符合 HuggingFace `tokenizers` 或 `sentencepiece` 接口。外部评测框架（`lm-evaluation-harness` 等）需要自定义适配层。

3. **概念验证级规模。** 85M 足以证明可训练性和架构合理性。扩展行为（cummax 注意力在 1B+ 规模下是改善还是退化？）是一个开放问题。

4. **无安全对齐。** 模型未经过 RLHF 安全训练。输出为原始补全。

5. **DPO 回退。** DPO 略低于 SFT，表明偏好数据或超参数需要针对该架构进一步调优。

## 项目结构

```
OpenASH-85M/
├── open_ash.py              # 模型定义：MaxStateSuper, DecoderLayer, OpenASH
├── open_ash_voc.py           # 分词器：jieba + 代理词表编码/解码
├── open_ash_infer.py         # 推理引擎：采样、流式生成、对话、工具调用
├── open_ash_webui.py         # Streamlit WebUI
├── open_ash_dataset.py       # 数据集：Pretrain / SFT / DPO / GRPO
├── train_pretrain.py         # 阶段 1：预训练
├── train_full_sft.py         # 阶段 2：SFT
├── train_dpo.py              # 阶段 3：DPO
├── train_grpo.py             # 阶段 4：GRPO
├── trainer_utils.py          # DDP、检查点、学习率调度、日志
├── config.py                 # 配置路径
├── configuration.json        # 模型元数据
├── open_ash_voc_agent.json   # 词表文件（基础词表 + 代理映射）
├── vocabulary_nnn.json       # 基础词表
├── models/
│   ├── full_sft_768_12.pth
│   ├── dpo_768_12.pth
│   └── grpo_768_12.pth
├── ceval_results_grpo.json
├── ceval_results_dpo.json
└── ceval_results_sft.json
```

## 快速开始

### 安装依赖

```bash
pip install torch jieba streamlit numpy tqdm
```

### 推理

```bash
python open_ash_infer.py
```

### 交互式对话

```bash
streamlit run open_ash_webui.py
```

### 训练

```bash
# 阶段 1：预训练
python train_pretrain.py \
    --data_path pretrain_t2t.jsonl \
    --epochs 6 --batch_size 40 --learning_rate 1.25e-4 \
    --max_seq_len 512 --save_dir ./models

# 阶段 2：SFT
python train_full_sft.py \
    --data_path sft_t2t.jsonl \
    --epochs 6 --batch_size 40 --learning_rate 1.25e-5 \
    --from_weight pretrain --max_seq_len 512 \
    --save_dir ./models --save_weight full_sft

# 阶段 3：DPO
python train_dpo.py \
    --data_path dpo_data.jsonl \
    --from_weight full_sft --save_dir ./models

# 阶段 4：GRPO
python train_grpo.py \
    --data_path rl_data.jsonl \
    --from_weight full_sft --save_dir ./models

# 多卡训练
torchrun --nproc_per_node=4 train_pretrain.py \
    --data_path pretrain_t2t.jsonl --batch_size 10
```

### 训练参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--data_path` | 训练数据（JSONL） | `pretrain_t2t.jsonl` |
| `--epochs` | 训练轮数 | 6 |
| `--batch_size` | 批次大小 | 40 |
| `--learning_rate` | 学习率 | 1.25e-4 / 1.25e-5 |
| `--max_seq_len` | chunk 最大长度 | 512 |
| `--accumulation_steps` | 梯度累积步数 | 1 |
| `--grad_clip` | 梯度裁剪 | 1.0 |
| `--from_weight` | 加载权重前缀 | `none` |
| `--hidden_size` | 隐藏维度 | 768 |
| `--num_layers` | Transformer 层数 | 12 |
| `--num_heads` | 注意力头数 | 8 |
| `--use_compile` | `torch.compile` | 0 |

### 数据格式

预训练：
```json
{"text": "..."}
```

SFT / DPO / GRPO：
```json
{"conversations": [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

支持 `system`、`user`、`assistant`、`tool` 角色，以及 `reasoning_content`（思考内容）、`tool_calls`（工具调用）、`tools`（工具定义）字段。

## 特性一览

- 基于 cummax 的注意力机制（零指数运算）
- 有状态跨 chunk 推理（逐 token O(1)，无需 KV Cache）
- 代理词表编码（二维/三维索引扩展）
- 多策略采样（Temperature / Top-K / Top-P / 重复惩罚）
- 流式输出
- 工具调用 / Function Calling 支持
- 思维链推理（`<|think|>...<|end_think|>`）
- 完整训练流水线（Pretrain → SFT → DPO → GRPO）
- 多卡 DDP + BFloat16 + 梯度累积 + 检查点断点续训
- Streamlit 可视化对话界面

## 引用

如需引用本工作：

```bibtex
@misc{openash,
  title={OpenASH: A Softmax-Free Language Model with Cumulative-Max Attention},
  author={dfytensor},
  year={2026},
  url={https://github.com/dfytensor/OpenASH}
}
```

## 许可证

MIT
