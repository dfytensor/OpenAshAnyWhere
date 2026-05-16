<div align="center">

# OpenASH

**85M 参数，不用 Softmax，也能通过中国考试。**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---



> **一句话概括**：OpenASH 是一个完全抛弃 Softmax 注意力的语言模型——核心运算只有 `torch.cummax`。在 85M 参数量级、C-EVAL 52 个科目的中文考试评测中，GRPO 强化学习版本取得 **26.82%** 的总体准确率，超越了同参数量级的 SFT 基线（23.11%）。

## 为什么关注 OpenASH？

| 对比维度 | 传统 Transformer | OpenASH |
|----------|-----------------|---------|
| 注意力核心 | Softmax（指数运算） | **Cumulative Max（仅比较运算）** |
| 训练计算复杂度 | O(n²d) | O(n²/2) |
| 推理计算复杂度 | O(n²d) | O(2)|
| 推理状态 | 无状态 / KV Cache | **有状态累积（Stateful）** |
| 参数量 | 数百 M 起步 | **85M 即可评测** |

不需要 GPU 集群，一张消费级显卡就能完整训练 + 评测。

## CEVAL 评测结果

在 C-EVAL 中文综合考试基准（52 个科目，1346 道题）上，5-shot 评测：

| 模型 | 训练方式 | 总体准确率 | 正确 / 总题数 |
|------|---------|-----------|--------------|
| SFT (full_sft_768_12) | 监督微调 | 23.11% | 311 / 1346 |
| DPO (dpo_768_12) | 偏好优化 | 22.88% | 308 / 1346 |
| **GRPO (grpo_768_12)** | **强化学习** | **26.82%** | **361 / 1346** |

### 亮点科目（GRPO）

| 科目 | 准确率 | 说明 |
|------|--------|------|
| legal_professional 法律职业 | 43.48% | 接近及格线 |
| college_physics 大学物理 | 42.11% | 理工科表现突出 |
| college_chemistry 大学化学 | 41.67% | |
| middle_school_geography 中学地理 | 41.67% | |
| chinese_language_and_literature 中文文学 | 39.13% | |
| tax_accountant 税务会计 | 38.77% | |
| advanced_mathematics 高等数学 | 36.84% | |
| metrology_engineer 计量工程师 | 37.50% | |

> 完整评测数据见 [`ceval_results_grpo.json`](ceval_results_grpo.json)、[`ceval_results_dpo.json`](ceval_results_dpo.json)、[`ceval_results_sft.json`](ceval_results_sft.json)。

## 模型架构

```
Input → Embedding → [DecoderLayer × N] → Linear Head → Output
                      │
                      ├─ MaxStateSuper (核心注意力，cummax 替代 softmax)
                      ├─ Gated FFN (门控前馈网络)
                      └─ Residual + LayerNorm
```

**MaxStateSuper** 是核心创新：

```python
out4, _ = torch.cummax(out2, dim=2)   # 累积最大值，替代 softmax
output = term1 + term2 + term3 + term4 + combined  # 多项式交互混合
```

- 没有指数运算，只有比较和线性组合
- 可学习的 `alpha` 参数控制各交互项的权重
- 支持**有状态推理**：跨 chunk 传递 `cummax` 状态，天然支持超长序列

### 模型参数

| 参数 | 值 |
|------|-----|
| 词表大小 | 23,004 + 代理词表 2,095 |
| 隐藏维度 | 768 |
| 层数 | 12 |
| 注意力头数 | 8 |
| 最大序列长度 | 8,192 |
| 总参数量 | **84,930,864 (85M)** |
| 模型文件大小 | ~162 MB |

## 项目结构

```
demo/
├── open_ash.py              # 模型定义（MaxStateSuper, DecoderLayer, OpenASH）
├── open_ash_voc.py           # 分词器（jieba + 代理词表编码）
├── open_ash_infer.py         # 推理引擎（采样、流式生成、交互式聊天）
├── open_ash_webui.py         # Streamlit WebUI（工具调用、思考模式）
├── open_ash_dataset.py       # 数据集（Pretrain / SFT / DPO / RLAIF / AgentRL）
├── train_pretrain.py         # 预训练脚本
├── train_full_sft.py         # SFT 全参数微调
├── train_dpo.py              # DPO 偏好优化
├── train_grpo.py             # GRPO 强化学习
├── trainer_utils.py          # 训练工具（分布式、检查点、学习率调度）
├── config.py                 # 配置
├── open_ash_voc_agent.json   # 代理词表映射
├── vocabulary_nnn.json       # 基础词表
├── models/                   # 模型权重
│   ├── full_sft_768_12.pth
│   ├── dpo_768_12.pth
│   └── grpo_768_12.pth
├── ceval_results_grpo.json   # GRPO 评测结果
├── ceval_results_dpo.json    # DPO 评测结果
└── ceval_results_sft.json    # SFT 评测结果
```

## 特性

- **自研注意力机制**: 累积最大值（cummax）替代 softmax，零指数运算
- **有状态推理**: 跨 chunk 状态传递，天然支持超长序列推理
- **代理词表编码**: 二维/三维索引扩展词表，23K 基础词表覆盖海量词汇
- **多种采样策略**: Temperature / Top-K / Top-P / 重复惩罚
- **流式输出**: 逐 token 实时解码
- **工具调用**: 支持 function calling / tool use
- **思考模式**: `<|think|>...<|end_think|>` 链式推理
- **全训练流水线**: 预训练 → SFT → DPO / GRPO，一站式完成
- **分布式训练**: 多卡 DDP + BFloat16 混合精度 + 梯度累积 + 断点续训
- **Streamlit WebUI**: 可视化对话界面，中英文双语

## 快速开始

### 安装依赖

```bash
pip install torch jieba streamlit numpy tqdm
```

### 推理

```bash
python open_ash_infer.py
```

### WebUI 对话

```bash
streamlit run open_ash_webui.py
```

### C-EVAL 评测

```bash
python ../eval_ceval.py
```

## 训练

### 预训练

```bash
python train_pretrain.py \
    --data_path pretrain_t2t.jsonl \
    --epochs 6 --batch_size 40 --learning_rate 1.25e-4 \
    --max_seq_len 512 --save_dir ./models
```

### SFT 微调

```bash
python train_full_sft.py \
    --data_path sft_t2t.jsonl \
    --epochs 6 --batch_size 40 --learning_rate 1.25e-5 \
    --from_weight pretrain --max_seq_len 512 \
    --save_dir ./models --save_weight full_sft
```

### GRPO 强化学习

```bash
python train_grpo.py \
    --data_path rl_data.jsonl \
    --from_weight full_sft --save_dir ./models
```

### 多卡训练

```bash
torchrun --nproc_per_node=4 train_pretrain.py \
    --data_path pretrain_t2t.jsonl --batch_size 10
```

## 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_path` | 训练数据（JSONL） | `pretrain_t2t.jsonl` |
| `--epochs` | 训练轮数 | 6 |
| `--batch_size` | 批次大小 | 40 |
| `--learning_rate` | 学习率 | 1.25e-4 / 1.25e-5 |
| `--max_seq_len` | chunk 最大序列长度 | 512 |
| `--accumulation_steps` | 梯度累积步数 | 1 |
| `--grad_clip` | 梯度裁剪 | 1.0 |
| `--from_weight` | 加载权重前缀 | `none` / `pretrain` |
| `--hidden_size` | 隐藏层维度 | 768 |
| `--num_layers` | 层数 | 12 |
| `--num_heads` | 注意力头数 | 8 |
| `--use_compile` | torch.compile | 0 |

## 数据格式

### 预训练

```json
{"text": "这是一段训练文本..."}
```

### SFT / RL

```json
{"conversations": [
  {"role": "user", "content": "你好"},
  {"role": "assistant", "content": "你好！有什么可以帮你的？"}
]}
```

支持 `system`、`user`、`assistant`、`tool` 角色，以及 `reasoning_content`（思考内容）、`tool_calls`（工具调用）、`tools`（工具定义）字段。

## License

MIT
