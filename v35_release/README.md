# OpenASH v3.5 —— 衰减最大值三通道混合架构（独立复现包）

O(S) 训练 / O(1) 推理状态的线性注意力架构。核心机制：**快/慢双通道衰减最大值扫描（decay-max）+ 归一化键 KV 检索通道**，配合分层多尺度 γ 门控。最新配方 **v3.7**（+FLA delta-rule 写入 + depthwise 因果短卷积）在同预算协议下于全部测试长度击败同规模 Transformer 基线与 v3.5 前代（40000 步全量：@384 领先 tf −21.8%、@2048 −74.8%），推理状态与每 token 计算保持严格 O(1)。

同预算（95-105M 参数、15000 步 × 184M tokens）下**全面超过**同规模现代 Transformer 基线（RoPE + RMSNorm + SwiGLU + FlashAttention）：全长度 PPL 领先（2048 区 +58.5%）、40000 步全量预训练 PPL 领先扩大至 +18.3%、推理状态与每 token 计算为严格 O(1)（相对序列长度）。

## 最终结果（big_* = 15000 步大数据缩放对决）

| 上下文 L | v3.5 PPL | tf PPL | v3.5 领先 |
|---|---|---|---|
| 64 | 18.56 | 19.47 | +4.7% |
| 128 | 17.03 | 18.05 | +5.7% |
| 256 | 16.36 | 17.69 | +7.5% |
| 384 | 16.01 | 17.93 | +10.7% |
| 512 | 15.92 | 19.01 | +16.3% |
| 1024 | 15.71 | 25.55 | +38.5% |
| 2048 | 15.64 | 37.68 | **+58.5%** |

（tf 无位置外推适配；即使加 RoPE 线性插值 PI ×4，tf@2048 = 35.20，v3.5 仍领先 22.8%）

全量预训练（40000 步 = 1 epoch × 488M tokens）：v3.5 **12.75** vs tf 15.61（领先 +18.3%，缩放优势随训练量扩大）。

## v3.7（2026-11-08 更新）：+delta 写入 +短卷积，全部七个长度击败 tf

在 v3.5 基础上两步进化（`train_ab.py --arch v37`）：

1. **v3.6 delta 写入**：KV 检索通道的 sum 累积 → FLA `chunk_gated_delta_rule` 精确覆盖（依赖 `pip install flash-linear-attention`，`baseline_fla.py` 同含 GDN/DeltaNet 基线适配器）
2. **v3.7 depthwise 因果短卷积**（kernel=4，per-channel SiLU 门控，注意力前）

15000 步等预算五模型对决（184M tokens；GDN 为 bs16×30000 步等 token 换算，101-106M 参数）：

| L | v3.5 | v3.6 | **v3.7** | GDN | tf |
|---|---|---|---|---|---|
| 64 | 20.68 | 18.39 | **17.16** | 21.60 | 19.47 |
| 128 | 19.02 | 17.01 | **15.98** | 17.88 | 18.05 |
| 384 | 17.82 | 16.02 | 15.10 | **14.40** | 17.93 |
| 2048 | 17.40 | 15.68 | 14.79 | **12.81** | 37.68 |

- **v3.7 在 64→2048 全部七个长度击败同预算 tf**（@384 −15.8%，@2048 −60.7%），保持严格 O(1) 推理状态
- 短上下文（≤128）同时反超 GDN；@384 与 GDN 残差收窄至 4.9%
- 远程复制召回：重复 span 绝对 CE 全部 gap 档位五模型最低

演进链（15000 步 PPL@384）：v3.0 84.53 → v3.5 17.82 → v3.6 16.02 → **v3.7 15.10**（累计 −82%）。详见仓库根 `REPORT_v35.md` §11。

## 终局：40000 步全 epoch（1 × 488M tokens，warmup 800 对齐协议）

| L | full_v35 | **full_v37** | full_tf | v37 vs tf |
|---|---|---|---|---|
| 64 | 15.15 | **14.38** | 17.63 | −18.4% |
| 384 | 12.68 | **12.10** | 15.48 | −21.8% |
| 1024 | 12.39 | **11.80** | 25.97 | −54.6% |
| 2048 | 12.32 | **11.73** | 46.58 | −74.8% |

**七个长度双杀 v35 与 tf**；复制召回 gap=128 gain +0.95 反超 tf（+0.84）；曲线未饱和。γ 诊断证实 delta KV 通道接管长程记忆（v37 全层快 γ 仍在 @2048 胜过保留慢 γ 的 v35）。原始数据：`results/longrange_results_40k_full.json`（五模型 15000 步版见 `longrange_results_15k_5model.json`）。

## 环境要求

- Python 3.11+，CUDA GPU ≥ 16GB 显存（24GB 更佳）
- PyTorch ≥ 2.1（CUDA 版）、Triton ≥ 2.2（随 PyTorch 附带）
- `pip install jieba pandas tqdm numpy`

词表编码器依赖 jieba 分词；正确性回归与长程评测依赖 pandas（C-Eval parquet）。

## 文件结构

```
open_ash_v3.py        v3.5/v3.6/v3.7 架构（三通道注意力层 + 短卷积 + delta 写入 + 完整模型）
triton_ops.py         全部 Triton/torch kernel（max-scan 前向/反向、gen_model 融合、KV scan、CE 融合）
baseline_tf.py        同预算 Transformer 基线（RoPE+RMSNorm+SwiGLU+SDPA，95.5M）
baseline_fla.py       FLA 库基线适配器（Gated DeltaNet 106.3M / DeltaNet 99.2M；v3.6+ delta 写入依赖）
open_ash_voc.py       jieba 词级编码器（含对话特殊 token）
open_ash_voc_agent.json  词表文件（311MB，本仓库 LFS）
config.py             词表路径配置
train_ab.py           预训练 harness（8 种架构、seed/数据对齐的 A/B 协议）
train_sft.py          SFT 微调
prepare_sft.py        SFT 数据预处理（jsonl → token 窗口缓存）
bench_v3_triton.py    正确性回归（58 项：前向/反向/融合 kernel 逐位对照）
bench_longrange.py    长程能力四探针（长度扩展 PPL / 远程复制 / 利用曲线 / γ 诊断）
eval_ceval.py         C-Eval 48 科目 5-shot 评测
demo_o1_decode.py     O(1) 增量解码演示（常数状态 + 恒定延迟验证）
demo_sft_compare.py   SFT 后生成对比
summarize_ab.py       训练结果汇总
test_tf_pi.py         tf 基线 RoPE-PI 公平性评测
results/              全部训练曲线与评测 json（本仓库实验的原始数据）
```

## 数据准备

三个数据文件需置于 `minimind_data/`（脚本默认相对路径；可用参数覆盖）：

| 文件 | 说明 | 获取 |
|---|---|---|
| `minimind_data/pretrain_cached_1270238_384.pt` | 预训练流：1,270,238 个 384+1 窗口（LongTensor 列表），约 4.88 亿 token | 由 minimind 项目的预训练语料分词缓存生成（`torch.save(list_of_LongTensor)` 格式） |
| `minimind_data/sft_t2t_mini.jsonl` | SFT 对话：905,718 条多轮（`conversations: [{role, content}, ...]`） | minimind 项目发布数据 |
| `minimind_data/pretrain_cached_30000_384.pt` | 小规模预训练流（可选，快速 A/B 用） | 同上 |

词表 `open_ash_voc_agent.json` 已在本仓库（LFS），无需额外下载。

C-Eval 数据（48 科目 dev+val parquet）由 `eval_ceval.py` 首次运行时自动从 hf-mirror.com 下载（也可用 `fetch_ceval2.py` 预下载）。

## 复现步骤

```bash
# 0) 正确性回归（58 项，全部 PASS 才继续）
python bench_v3_triton.py

# 1) 2500 步小数据 A/B（复现 v3.1 基线；约 15 分钟/模型）
python train_ab.py --arch v30 --steps 2500 --out ab_v30
python train_ab.py --arch v31 --steps 2500 --out ab_v31

# 2) 3000 步架构消融（v3.2/v3.3/v3.5s 等按 train_ab.py 的 build() 分支）
python train_ab.py --arch v35 --steps 3000 --out ab_v35_3k

# 3) 15000 步大数据缩放对决（复现主表；v35 ~75 分钟，tf ~53 分钟）
python train_ab.py --arch v35 --steps 15000 --warmup 500 \
    --data minimind_data/pretrain_cached_1270238_384.pt --out big_v35
python train_ab.py --arch tf  --steps 15000 --warmup 500 \
    --data minimind_data/pretrain_cached_1270238_384.pt --out big_tf

# 4) 长程能力四探针（编辑 bench_longrange.py 的 MODEL_KEYS 选择对比模型）
python bench_longrange.py

# 5) 全量预训练 1 epoch（40000 步 ≈ 3.5 小时/模型）
python train_ab.py --arch v35 --steps 40000 --warmup 800 --eval_every 2000 \
    --data minimind_data/pretrain_cached_1270238_384.pt --out full_v35

# 6) SFT（从全量 ckpt 继续；需先运行 prepare_sft.py 生成 21.6M token 缓存）
python prepare_sft.py
python train_sft.py --base full_v35/model.pt --arch v35 --steps 5000 --out sft_v35

# 7) C-Eval 评测（1250 题 5-shot）
python eval_ceval.py --ckpt big_v35/model.pt --arch v35

# 8) O(1) 解码演示与 SFT 生成对比
python demo_o1_decode.py
python demo_sft_compare.py

# 9) 汇总
python summarize_ab.py
```

## 架构：每层三通道（详细）

```
x ──► RMSNorm₁ ──► MaxStateSuper ──►(+残差)──► RMSNorm₂ ──► SwiGLU FFN ──►(+残差)
                        │
    ┌───────────────────┼────────────────────────────┐
    │ c 流 = state_em(ids)·state_proj·em_scale（每层专属低秩查表，64 rank）    │
    │                                                                       │
    │ 通道1 快/慢 max-scan：  γ_j = exp(−softplus(c_j·v + b))                │
    │       e_j = max(γ_j·e_{j−1}, c_j)    γ 分层多尺度：                    │
    │       层0-5 快 γ∈[0.70,0.9999]（半衰期 ~4 tok）                       │
    │       层6-11 慢 γ∈[0.995,0.9995]（半衰期 ~245 tok）                   │
    │                                                                       │
    │ 通道2 KV 检索（kv_dh=48）：S_j = a_j·S_{j−1} + β·k̂_j⊗c_j              │
    │       r_j = q_j·S_j     k̂ = c‖·‖₂ 归一化，a/β per-head 可学习         │
    │                                                                       │
    │ gen_model 多项式融合（Triton）：                                      │
    │   out = head(cat[a,b,c,d,e])·e + a·b + α₁·b + α₂·d                    │
    │         + a·(α₃·e+d) + b·(c+e) + c·e + kv_gate·r                      │
    └───────────────────────────────────────────────────────────────────────┘
```

推理（逐 token）：state 元组携带 `(max-scan e 向量, KV 矩阵 S)`——每步 O(768) + O(48²×8) 更新，状态总量恒定（v35s 12 层 ≈ 460k 数值）。fp32 逐位等价性已验证（vs 全序列前向 diff 3.7e-06）。

## 预期结果（results/ 中的原始数据）

| 实验 | v3.x | tf 基线 |
|---|---|---|
| 2500 步（11.5M tok）v3.0 vs v3.1 | 84.53 / 59.07 | — |
| 3000 步架构消融 | v32 46.90 / v33 46.80 / v35s 40.87 | — |
| 15000 步大数据 | v35 16.01 | 17.93 |
| 40000 步全量预训练 | 12.75 | 15.61 |
| C-Eval（1250 题 5-shot，SFT 后） | 23.12% | 22.64% |
| 长度扩展 | 64→2048 全长度领先 | >512 崩溃（无外推适配） |

注意：100M 参数 / 数亿 token 尺度下 C-Eval 位于随机水平（~23% vs 随机 25%），由数据量与参数量主导——架构差异的正确判据是 PPL 与长程探针，C-Eval 在此仅作流水线完整性验证。

## 已知边界

1. **delta-rule 写入的分块 GPU 实现**未包含（当前 KV 通道为 sum 写入）：逐位递推的 Triton 前向 + 手工梯度反向需照 Gated DeltaNet Algorithm / FLA 源码（wy_fast.py）实现，预计专门一轮 2-3 小时。数学框架（WY 表示、l-归一化变换）已在 `REPORT_v35.md` 第 9 节完整记录。它预期增强 gap=128 中距检索。
2. 本环境（Windows + Triton）存在「循环出口累加器 store」的 codegen 非定性问题——仓库内所有 kernel 均采用「循环内逐步 store」稳定模式规避（见 `triton_ops.py` 注释）。
3. tf 基线未做位置外推适配（YaRN/PI）；PI 适配版对比见 `test_tf_pi.py`（v3.5 仍领先 22.8% @2048）。
4. C-Eval 分数在该尺度接近随机，不作为架构判据。
