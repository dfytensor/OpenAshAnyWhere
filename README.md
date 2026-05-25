<div align="center">

# OpenASH-85M

**A Softmax-Free Language Model with Cumulative-Max Attention and Stateful Inference**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

## Proposition

Softmax is not a necessary component of attention. OpenASH replaces the entire softmax-based attention mechanism with **cumulative maximum (`torch.cummax`)**, a pure comparison operation, combined with learnable polynomial mixing. The resulting architecture:

- Eliminates all exponential computation from the attention path
- Carries a **state tensor** across sequence chunks natively, enabling constant-time token-by-token inference without KV cache
- Is trainable end-to-end through a complete LLM pipeline (pretrain → SFT → DPO → GRPO) at 85M parameters on consumer hardware

This is a **proof-of-concept** that an alternative attention paradigm—grounded in order statistics rather than probability normalization—can produce a functional language model.

## Architecture

```
Input → Embedding → [DecoderLayer × 12] → Linear Head → Logits
                       │
                       ├─ MaxStateSuper (cummax attention, stateful)
                       ├─ Gated FFN (swish-gate linear)
                       └─ Weighted Residual + LayerNorm
```

### MaxStateSuper: Cummax-Based Attention

The core mechanism replaces softmax(QK^T)V with:

```python
combined = self.combined(x).view(b, s, 4, self.heads, -1)
out, out1, out2, out3 = combined.unbind(2)

out4, _ = torch.cummax(out2, dim=2)          # cumulative max, not softmax

output = term1 + term2 + term3 + term4 + combined  # learnable polynomial mixing
```

Where the output is a sum of element-wise interaction terms controlled by learnable `alpha` parameters:

- `term1 = a * b` (query-key product)
- `term2 = alpha1 * b + alpha2 * d` (value + cummax-state linear mix)
- `term3 = a * (alpha3 * e + d)` (query-state gated path)
- `term4 = b * (c + e)` (key-state cross)
- `c * e` (original value-cummax residual)
- `head_linear(cat([a,b,c,d,e])) * e` (learned head projection weighted by state)

**No exponential operations. No attention matrix. The "attention weight" is implicitly the running maximum of projected values.**

### Stateful Inference

During generation, the cummax state from the previous chunk is passed forward:

```python
out4, _ = torch.cummax(torch.cat([state, out2], dim=2), dim=2)
state = out4[:, :, -1:]  # carry forward to next chunk
```

This means each decoding step is **O(1)** with respect to sequence history — the model does not re-attend to past tokens, it carries a compressed state.

### Model Configuration

| Parameter | Value |
|---|---|
| Vocabulary size | 23,004 base + 2,095 agent-proxy tokens |
| Hidden dimension | 768 |
| Layers | 12 |
| Attention heads | 8 |
| Total parameters | **84,930,864 (85M)** |
| Max sequence length (chunk) | 8,192 |
| Weight file size | ~162 MB |

### Proxy Vocabulary Encoding

To cover a large character space with a compact token table, OpenASH uses a two/three-dimensional indexing scheme:

- **2D tokens**: Character → `(ts_i, te_j)` — encodes up to `N²` additional tokens using `2N` base tokens
- **3D tokens**: Character → `(rs_i, rc_j, re_k)` — encodes up to `N³` additional tokens using `3N` base tokens

Tokenization is jieba-based with fallback to character-level encoding.

## Training Pipeline

OpenASH was trained through a complete four-stage pipeline, all implemented from scratch in PyTorch:

```
Stage 1: Pretrain    (next-token prediction on raw text)
Stage 2: SFT         (supervised fine-tuning on instruction data)
Stage 3: DPO         (direct preference optimization)
Stage 4: GRPO        (group relative policy optimization / RL)
```

Each stage loads the previous stage's weights. Distributed training is supported via PyTorch DDP with BFloat16 mixed precision, gradient accumulation, and checkpoint-based resume.

### Key Implementation Details

- **Optimizer**: AdamW with cosine learning rate schedule
- **Precision**: BFloat16 AMP with GradScaler
- **Distributed**: `torchrun` with NCCL backend, `SkipBatchSampler` for resume
- **Sampling**: Temperature / Top-K / Top-P / Repetition penalty (configurable)
- **Streaming**: Real-time token-by-token decoding with agent-vocabulary buffer management

## Evaluation: C-EVAL

Evaluated on **C-EVAL** (Chinese comprehensive exam benchmark, 52 subjects, 1,346 questions), 5-shot:

| Model | Stage | Overall Accuracy | Correct / Total |
|---|---|---|---|
| full_sft_768_12 | SFT | 23.11% | 311 / 1,346 |
| dpo_768_12 | DPO | 22.88% | 308 / 1,346 |
| **grpo_768_12** | **GRPO** | **26.82%** | **361 / 1,346** |

### Subject Highlights (GRPO)

| Subject | Accuracy |
|---|---|
| Legal Professional | 43.48% |
| College Physics | 42.11% |
| College Chemistry | 41.67% |
| Middle School Geography | 41.67% |
| Chinese Language & Literature | 39.13% |
| Tax Accountant | 38.77% |
| Advanced Mathematics | 36.84% |

Full results: [`ceval_results_grpo.json`](ceval_results_grpo.json), [`ceval_results_dpo.json`](ceval_results_dpo.json), [`ceval_results_sft.json`](ceval_results_sft.json).

## Limitations (Honestly Stated)

1. **C-EVAL absolute numbers are modest.** 26.82% on 85M is a baseline, not a competitive result. The claim is not "this beats existing models" — it is "this architecture can be trained and produces non-trivial signal."

2. **Non-standard tokenizer.** The jieba + proxy vocabulary scheme does not conform to HuggingFace `tokenizers` or `sentencepiece` interfaces. External evaluation frameworks (`lm-evaluation-harness`, etc.) require a custom adapter.

3. **Proof-of-concept scale.** 85M is sufficient to demonstrate trainability and architectural soundness. Scaling behavior (does cummax attention improve or degrade at 1B+?) is an open question.

4. **No safety alignment.** The model has no RLHF safety training. Outputs are raw completions.

5. **DPO regression.** DPO slightly underperformed SFT, suggesting the preference data or hyperparameters need tuning for this architecture.

## Project Structure

```
OpenASH-85M/
├── open_ash.py              # Model: MaxStateSuper, DecoderLayer, OpenASH
├── open_ash_voc.py           # Tokenizer: jieba + proxy vocabulary encode/decode
├── open_ash_infer.py         # Inference: sampling, streaming, chat, tool call
├── open_ash_webui.py         # Streamlit WebUI
├── open_ash_dataset.py       # Datasets: Pretrain / SFT / DPO / GRPO
├── train_pretrain.py         # Stage 1: Pretrain
├── train_full_sft.py         # Stage 2: SFT
├── train_dpo.py              # Stage 3: DPO
├── train_grpo.py             # Stage 4: GRPO
├── trainer_utils.py          # DDP, checkpoint, LR schedule, logging
├── config.py                 # Paths
├── configuration.json        # Model metadata
├── open_ash_voc_agent.json   # Tokenizer vocab (base + proxy mappings)
├── vocabulary_nnn.json       # Base vocabulary
├── models/
│   ├── full_sft_768_12.pth
│   ├── dpo_768_12.pth
│   └── grpo_768_12.pth
├── ceval_results_grpo.json
├── ceval_results_dpo.json
└── ceval_results_sft.json
```

## Quick Start

### Install

```bash
pip install torch jieba streamlit numpy tqdm
```

### Inference

```bash
python open_ash_infer.py
```

### Interactive Chat

```bash
streamlit run open_ash_webui.py
```

### Train

```bash
# Stage 1: Pretrain
python train_pretrain.py \
    --data_path pretrain_t2t.jsonl \
    --epochs 6 --batch_size 40 --learning_rate 1.25e-4 \
    --max_seq_len 512 --save_dir ./models

# Stage 2: SFT
python train_full_sft.py \
    --data_path sft_t2t.jsonl \
    --epochs 6 --batch_size 40 --learning_rate 1.25e-5 \
    --from_weight pretrain --max_seq_len 512 \
    --save_dir ./models --save_weight full_sft

# Stage 3: DPO
python train_dpo.py \
    --data_path dpo_data.jsonl \
    --from_weight full_sft --save_dir ./models

# Stage 4: GRPO
python train_grpo.py \
    --data_path rl_data.jsonl \
    --from_weight full_sft --save_dir ./models

# Multi-GPU
torchrun --nproc_per_node=4 train_pretrain.py \
    --data_path pretrain_t2t.jsonl --batch_size 10
```

### Training Arguments

| Argument | Description | Default |
|---|---|---|
| `--data_path` | Training data (JSONL) | `pretrain_t2t.jsonl` |
| `--epochs` | Training epochs | 6 |
| `--batch_size` | Batch size | 40 |
| `--learning_rate` | Learning rate | 1.25e-4 / 1.25e-5 |
| `--max_seq_len` | Max chunk length | 512 |
| `--accumulation_steps` | Gradient accumulation | 1 |
| `--grad_clip` | Gradient clipping | 1.0 |
| `--from_weight` | Weight prefix to load | `none` |
| `--hidden_size` | Hidden dimension | 768 |
| `--num_layers` | Transformer layers | 12 |
| `--num_heads` | Attention heads | 8 |
| `--use_compile` | `torch.compile` | 0 |

### Data Format

Pretrain:
```json
{"text": "..."}
```

SFT / DPO / GRPO:
```json
{"conversations": [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

Supports `system`, `user`, `assistant`, `tool` roles with `reasoning_content`, `tool_calls`, and `tools` fields.

## Features

- Cummax-based attention (zero exponential ops)
- Stateful cross-chunk inference (O(1) per token, no KV cache)
- Proxy vocabulary encoding (2D/3D index expansion)
- Multi-strategy sampling (Temperature / Top-K / Top-P / Repetition Penalty)
- Streaming output
- Tool calling / function use support
- Chain-of-thought (`<|think|>...<|end_think|>`)
- Full training pipeline (pretrain → SFT → DPO → GRPO)
- Multi-GPU DDP + BFloat16 + gradient accumulation + checkpoint resume
- Streamlit WebUI

## Citation

If you reference this work, please cite:

```bibtex
@misc{openash,
  title={OpenASH: A Softmax-Free Language Model with Cumulative-Max Attention},
  author={dfytensor},
  year={2026},
  url={https://github.com/dfytensor/OpenASH}
}
```

## License

MIT
