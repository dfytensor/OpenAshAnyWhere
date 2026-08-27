# OpenASH 项目实验指南

本文件整理了项目中所有训练、推理、基准测试脚本的用途和运行方式。

工作目录: `F:\OpenASH2605`
虚拟环境: `F:\OpenASH\.venv`
数据目录: `F:\OpenASH2605\minimind_data\`

---

## 1. 模型定义文件

| 文件 | 模型 | 说明 |
|------|------|------|
| `open_ash.py` | OpenASH | 多头 cummax + ReLU FFN，主模型 |
| `open_ash_v2.py` | OpenASH_V2 | PhaseGate FFN (sin+cos) 变体 |
| `open_ash_voc.py` | — | 词表编解码器 (23,004 tokens) |
| `open_ash_dataset.py` | — | 数据集类 (Pretrain/SFT/DPO/RL) |
| `trainer_utils.py` | — | 训练工具 (lr调度/断点续训/DDP) |
| `config.py` | — | 词表路径配置 |
| `wdlm_verification/wdlm_neural.py` | WDLM-Neural | 全维度 cummax + NeuralWaveStep，用于 60M 实验 |
| `wdlm_verification/wdlm.py` | WDLM | 完整波动力学模型 (含复数) |

---

## 2. 核心训练流程 (OpenASH 85M)

完整训练管线: **Pretrain → SFT → (可选 GRPO/DPO)**

### 2.1 Pretrain

```powershell
python train_pretrain.py --epochs 6 --batch_size 32 --learning_rate 5e-4 --max_seq_len 512 --data_path pretrain_t2t.jsonl --save_weight pretrain --use_compile 0
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--epochs` | 6 | 训练轮数 |
| `--batch_size` | 32 | 批大小 |
| `--learning_rate` | 5e-4 | 学习率 |
| `--max_seq_len` | 512 | 最大序列长度 |
| `--data_path` | pretrain_t2t.jsonl | 数据文件名 |
| `--from_weight` | — | 加载已有权重继续训练 |
| `--from_resume` | 0 | 自动检测最新 checkpoint 续训 |
| `--save_dir` | ../out | 保存目录 |
| `--save_weight` | pretrain | 权重文件名前缀 |
| `--accumulation_steps` | 1 | 梯度累积步数 |
| `--use_compile` | 0 | torch.compile 开关 |

### 2.2 SFT

```powershell
python train_full_sft.py --epochs 6 --batch_size 20 --learning_rate 5e-5 --max_seq_len 1024 --data_path sft_data.jsonl --from_weight pretrain --save_weight full_sft --use_compile 0
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--max_seq_len` | 1024 | SFT 序列长度 (通常大于 pretrain) |
| `--learning_rate` | 5e-5 | 比 pretrain 低 10 倍 |
| `--from_weight` | pretrain | 加载预训练权重 |

### 2.3 GRPO (强化学习对齐)

```powershell
python train_grpo.py --epochs 1 --batch_size 4 --from_weight full_sft --num_generations 4 --max_gen_len 256 --data_path agent_rl.jsonl
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--num_generations` | 4 | 每个 prompt 生成数 |
| `--num_ppo_epochs` | 2 | PPO 更新轮数 |
| `--clip_eps` | 0.2 | PPO clip |
| `--beta_kl` | 0.3 | KL 惩罚系数 |
| `--max_gen_len` | 256 | 最大生成长度 |

### 2.4 DPO (偏好对齐)

```powershell
python train_dpo.py --epochs 3 --batch_size 4 --from_weight full_sft --data_path dpo.jsonl --beta 0.1
```

---

## 3. 同参数量训练 (60M 对比实验)

训练脚本位于 `train_60m/`，支持断点续训、数据缓存、自动生成测试。

### 3.1 WDLM-Neural 60M

```powershell
python train_60m/train.py --pretrain_epochs 3 --sft_epochs 2 --compile 0
```

| 参数 | 说明 |
|------|------|
| `--pretrain_epochs` | Pretrain 轮数 |
| `--sft_epochs` | SFT 轮数 |
| `--skip_pretrain` | 跳过 pretrain (已有权重时) |
| `--skip_sft` | 跳过 SFT |
| `--compile` | 0/1, torch.compile 开关 |
| `--test_only` | 只跑生成测试和 PPL |
| `--max_lines_pretrain` | 限制 pretrain 数据行数 (调试用) |
| `--max_lines_sft` | 限制 SFT 数据行数 |

模型配置: H=512, L=10, 60.3M 参数
训练序列: Pretrain seq=512, SFT seq=1024
产出: `train_60m/wdlm60m_pretrain_final.pth`, `train_60m/wdlm60m_sft_final.pth`

### 3.2 OpenASH 58M

```powershell
python train_60m/train_openash.py --pretrain_epochs 3 --sft_epochs 2 --compile 0
```

参数同上。跳过已完成的阶段:

```powershell
python train_60m/train_openash.py --skip_pretrain --sft_epochs 2 --compile 0
```

模型配置: H=640, L=10, heads=8, 58.2M 参数
训练序列: Pretrain seq=512, SFT seq=768
产出: `train_60m/openash60m_pretrain_final.pth`, `train_60m/openash60m_sft_final.pth`

**注意事项**:
- 缓存文件在 `train_60m/cache/` 下，损坏时删除重建
- Windows 环境必须 `num_workers=0`
- 使用 `safe_save` 避免 Windows 文件锁

---

## 4. 推理 / 交互对话

### 4.1 OpenASH 推理

```powershell
python open_ash_infer.py --weight full_sft_768_12
```

支持: 单轮对话、多轮对话、流式输出、工具调用

### 4.2 WDLM 60M 推理

```powershell
python train_60m/infer.py
```

### 4.3 Web UI

```powershell
streamlit run open_ash_webui.py
```

---

## 5. 基准测试脚本

所有 bench 脚本位于 `experiment_openash_vs_wdlm/bench/`，自包含 (不依赖外部路径)。

### 5.1 同参数量推理对比 (OA-58M vs WDLM-60M)

```powershell
python experiment_openash_vs_wdlm/bench/bench_sameparam.py
```

测试项: 生成速度、TTFT、PPL、生成质量、样本输出、GPU 显存
产出: 终端表格输出

### 5.2 跨参数量推理对比 (OA-85M vs WDLM-60M)

```powershell
python experiment_openash_vs_wdlm/bench/bench_compare.py
```

测试项: 生成速度、TTFT、PPL、序列扩展性、批量吞吐、质量、显存、长期依赖

### 5.3 同参数量训练速度对比

```powershell
python experiment_openash_vs_wdlm/bench/bench_train_sameparam.py
```

对比 ~60M 和 ~85M 两组配置的训练速度、Loss、显存 (各 200 steps)

### 5.4 外推能力测试

三模型外推 (至 16K):
```powershell
python experiment_openash_vs_wdlm/bench/bench_extrap_all3.py
```

拼接数据外推 (至 16K):
```powershell
python experiment_openash_vs_wdlm/bench/bench_extrap_long.py
```

State 精度 + 短序列外推:
```powershell
python experiment_openash_vs_wdlm/bench/bench_state_extrap.py
```

### 5.5 长期依赖测试

```powershell
python experiment_openash_vs_wdlm/bench/bench_longrange.py
```

测试: KV 检索、PPL vs 上下文长度、state 信息保持

---

## 6. 外推崩溃分析实验

脚本位于 `experiment_extrap_analysis/`。

### 6.1 逐层 State 分析

```powershell
python experiment_extrap_analysis/analyze_layer_state.py
```

追踪每层 cummax state 的 norm、max 随序列长度的变化，定位爆炸层。

### 6.2 State 截断消融实验

```powershell
python experiment_extrap_analysis/ablation_state.py
```

测试: 单层/多层 clamp、norm cap、skip、freeze 等干预对 PPL 的影响。

### 6.3 修复后外推极限 (至 128K)

```powershell
python experiment_extrap_analysis/bench_extrap_limit.py
```

对比 WM-base / WM-fix(200) / OA58-base / OA58-fix 的 PPL 至 128K tokens。

### 6.4 修复后外推 + 三模型对比

```powershell
python experiment_extrap_analysis/bench_fixed_extrap.py
```

WM-loose / WM-tight / WM-200 vs OA-58M / OA-85M。

### 6.5 OA 修复尝试

```powershell
python experiment_extrap_analysis/bench_oa_fix.py
```

测试 OA-58M / OA-85M 加 state cap、decay、output cap 的效果。

### 6.6 Cap + Decay 组合方案

```powershell
python experiment_extrap_analysis/bench_combo_cap_decay.py
```

测试 cap-same / cap-200 + decay (0.99/0.97/0.95) 的组合效果，对比单独使用。
三模型 (OA-58M, OA-85M, WM-60M) × 8 种干预方案。

### 6.7 通用 Cap 扫描

```powershell
python experiment_extrap_analysis/bench_universal_cap.py
```

测试固定 cap 值 (50/100/150/200/300/500)、decay 值 (0.9~0.995)、cap+decay 组合、
ratio-based cap (k*sqrt(H)) 的跨模型通用性。
四组实验: 通用 cap 扫描 / decay 扫描 / cap=150+decay 组合 / ratio-based cap。

### 6.8 修复后推理性能检查

```powershell
python experiment_extrap_analysis/bench_inference_check.py
```

对比 baseline vs fixed 的生成速度、PPL、质量、样本输出。

### 6.9 训练时集成 State Cap 验证

```powershell
python experiment_extrap_analysis/train_with_cap.py
```

从头训练 WDLM 200 steps，对比有/无 state cap 的 loss 收敛、PPL、外推。

---

## 7. WDLM 验证脚本

位于 `wdlm_verification/`，用于开发调试 WDLM 模型。

### 7.1 训练速度对比

```powershell
python wdlm_verification/compare_neural.py    # Neural vs Real 速度
python wdlm_verification/compare_train.py     # OpenASH vs WDLM loss 曲线
```

### 7.2 收敛测试

```powershell
python wdlm_verification/converge_test.py     # 2000步收敛
```

### 7.3 State 一致性验证

```powershell
python wdlm_verification/test_state.py        # chunked == full-sequence
```

### 7.4 组件测试

```powershell
python wdlm_verification/test_wdlm.py         # 编码/演化/干涉/注意力/梯度
```

### 7.5 FFN 消融

```powershell
python wdlm_verification/compare_phasegate.py --steps 2000  # PhaseGate vs ReLU
```

---

## 8. 30M Cap+Decay 训练

脚本位于 `train_30m_cap_decay/`，训练集成 state norm cap + decay 的 OpenASH 30M 模型。

模型配置: H=432, L=8, heads=8, ≈30M 参数
默认参数: cap=150, decay=0.97, chunk=64
训练时每个 chunk 后对 cummax state 做范数截断 + 衰减，推理时无需额外干预。

### 8.1 完整训练

```powershell
python train_30m_cap_decay/train.py --pretrain_epochs 3 --sft_epochs 2 --compile 0
```

### 8.2 跳过 Pretrain

```powershell
python train_30m_cap_decay/train.py --skip_pretrain --sft_epochs 2 --compile 0
```

### 8.3 仅生成测试

```powershell
python train_30m_cap_decay/train.py --test_only
```

### 8.4 自定义 cap/decay 参数

```powershell
python train_30m_cap_decay/train.py --state_cap 200 --state_decay 0.99 --compile 0
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--state_cap` | 150 | state 范数截断阈值 |
| `--state_decay` | 0.97 | 每个 chunk 后的 state 衰减系数 |
| `--pretrain_epochs` | 3 | Pretrain 轮数 |
| `--sft_epochs` | 2 | SFT 轮数 |
| `--skip_pretrain` | — | 跳过 pretrain |
| `--skip_sft` | — | 跳过 SFT |
| `--test_only` | — | 只跑生成测试 |
| `--compile` | 0 | torch.compile 开关 |
| `--max_lines_pretrain` | 0 | 限制 pretrain 数据行数 |
| `--max_lines_sft` | 0 | 限制 SFT 数据行数 |

产出: `train_30m_cap_decay/openash30m_cd_pretrain_final.pth`, `openash30m_cd_sft_final.pth`

---

## 9. 20M 多模型对比

```powershell
python train_20m/train.py                     # 训练 WDLM/Transformer/OpenASH
python train_20m/benchmark.py                 # PPL/生成质量/速度基准
```

---

## 9. 权重文件

| 文件 | 模型 | 位置 |
|------|------|------|
| `full_sft_768_12.pth` | OA-85M SFT | `experiment_openash_vs_wdlm/bench/` |
| `openash60m_sft_final.pth` | OA-58M SFT | `experiment_openash_vs_wdlm/bench/` |
| `wdlm60m_sft_final.pth` | WDLM-60M SFT | `experiment_openash_vs_wdlm/bench/` |
| `openash60m_pretrain_final.pth` | OA-58M Pretrain | `train_60m/` |
| `wdlm60m_pretrain_final.pth` | WDLM-60M Pretrain | `train_60m/` |

---

## 10. 数据文件

| 文件 | 路径 | 说明 |
|------|------|------|
| `sft_t2t_mini.jsonl` | `minimind_data/` | SFT 数据 (905K 条) |
| `pretrain_t2t_mini.jsonl` | `minimind_data/` | Pretrain 数据 (1.27M 条) |
| `open_ash_voc_agent.json` | `experiment_openash_vs_wdlm/bench/` | 词表文件 |

---

## 11. 常见问题

**Q: DataLoader 卡死**
A: Windows 下 `num_workers` 必须为 0。

**Q: `torch.save` 报 Error 1224**
A: 使用 `safe_save` (先写临时文件再 `os.rename`)。

**Q: CUDA index out of bounds**
A: 删除缓存 `.pt` 文件重建: `Remove-Item train_60m/cache/*.pt`。

**Q: `load_state_dict` key 不匹配 (`_orig_mod.` 前缀)**
A: 统一 `--compile 0`，不使用 `torch.compile`。

**Q: SFT 缓存损坏 (token ID 超出范围)**
A: 删除缓存重建，已内置 `x.clamp(0, vs-1)` 安全措施。
