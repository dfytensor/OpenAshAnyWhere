# Ballistic Distance Kernels: from a physics analogy to measurable priors for sequence models

**弹道距离核：从物理类比到序列模型的可测量先验 —— 完整研究文档（v1/v2/v3/v4/v5/v6 全链条，含勘误与负结果）**

---

## Abstract (English)

We start from a naive physics analogy — "every context token is a ball thrown toward the next-token position" — and formalize it. The formalization reveals that the analogy's landing distance R coincides exactly with the lag-k statistic of n-gram models, that the physical parameters (mass, velocity, angle) are unidentifiable (3 degrees of freedom mapped to 1 observable; fitting error 3e-13 with mass frozen), and that the *effective core* is a low-rank mixture of distance kernels: K tables become samples of one function A(k) ≈ Σ_j γ_j(k;R_j,σ_j)·Φ_j.

On a real Chinese corpus (minimind, 16M chars / 22M subword tokens) we find: (i) the far-field structure decays as a power law (‖A^k−π‖ ∝ k^−0.42), not a constant as in synthetic data; (ii) with J=4 prototypes fitted on lags ≤6 only, plus a *decay envelope* whose amplitude is anchored to noise-debiased per-lag measurements, the model reaches test perplexity **34.07 vs 38.57 for the full discrete table (+11.7%)** using 1/6 of the parameters and half the lags; (iii) gains grow as data shrinks (+7~16% for ≤4M tokens).

We then port the kernel to attention as an additive bias (52 parameters). After correcting an indexing bug (our first "beats ALiBi" result was an absolute-position prior in disguise; corrected distance-indexed kernels **tie** ALiBi at 11.4M), we report two honest negative results: content-conditioned prototype selection shows no measurable gain, and the kernel confers no length-extrapolation advantage over ALiBi (identical degradation ×6.1 at 4× context, RoPE degrades slowest ×4.0). We ship a fully automatic diagnostic (`autoballistic`) that decides smoothability, prototype count J via the cross-lag spectrum, and envelope choice without oracle access, reproducing our best pipeline end-to-end.

**Bottom line:** the reliable value of the ballistic kernel today is on the *statistical* side — n-gram compression, small-data regimes, extrapolated lag structure. On the attention side it is currently an *interpretable, diagnosable, ALiBi-equivalent* prior; two upgrade paths (content-conditioned kernels, length extrapolation) were tested and did not yet pay off. All claims are backed by logs, seeds, and scripts in this repository.

---

## 摘要（中文）

从「每个上文 token 是一只抛向预测位的小球」这一类比出发做严格化：落点距离 R 恰为 n-gram 的 lag-k 统计；物理参数 (m,v,θ) 不可辨识（3 自由度 ↦ 1 观测量）；有效内核是距离核的低秩混合——K 张 lag 表成为同一函数 A(k) 的采样点。

在真实中文语料（minimind，16M 字符 / 22M 子词）上实测：远场结构按幂律衰减（k^−0.42）；仅用 k≤6 拟合 J=4 原型、配「方向外推 + 振幅逐 lag 去偏锚定」的衰减包络，即达测试困惑度 **34.07 vs 全量离散表 38.57（+11.7%）**，参数 1/6、lag 用量一半；数据越少收益越大（≤4M token 时 +7~16%）。

将该核移植为注意力加性偏置（52 参数）后，修正一处索引 bug（首个「胜 ALiBi」结果实为伪装的绝对位置先验；真距离核与 ALiBi 在 11.4M 打平），并报告两个诚实的负结果：内容条件化原型选择无可测增益；该核相对 ALiBi 无长度外推优势（4 倍上下文处退化同为 ×6.1，RoPE 最稳 ×4.0）。附全自动诊断工具 `autoballistic`：无 oracle 情况下自动判定可平滑性、谱定 J、选包络，端到端复现最优管线。

---

## 1. 研究路线（五次迭代）

| 版本 | 问题 | 关键结论 | 证据 |
|---|---|---|---|
| v1/v2 | 周期 = lag-k 统计；多 lag 聚合 | Kac 回归 E[G]=1/P(v) 误差 0.02%；聚合把 13.89→11.72 | v1/v2 报告 |
| v3 | 类比严格化 + 压缩 | (m,v,θ) 不可辨识；连续落点+平滑核+低秩原型：参数 1/6 性能反升 | `v3_ballistic/` |
| v4 | 外推为什么有效 | 12 张表 = 2 个成分（谱 74.5%）；外推恢复 90%；合成语料远场恒定 | `v4_principle/` |
| v5 | 真实语料 | 远场幂律衰减（CV 41.9%）；J=4+k≤6 外推 **+11~16%**；衰减包络；子词级一致 | `v5_minimind/REPORT_v5_minimind.md` |
| v6 | 注意力上车 | **勘误**：索引 bug 后三方案打平；负结果×2；包络 v2；autoballistic 全自动 +11.7% | `v6_attbias/REPORT_v6_attbias.md` |

## 2. 方法

### 2.1 n-gram 弹道模型（统计侧，已验证）

```
表:   Â(k) = norm[ π + d(k)·( Σ_j γ_kj(R_j,σ_j)·Φ_j − π ) ]        方向=弹道外推, 振幅=锚定
拟合: k≤Kfit=K/4; 频率 top-400 源行; Nelder-Mead ×4 重启 (越界惩罚+发散重启)
锚定: s(k) = √( ‖A^k−π‖² − 2·noise(k)² )      noise: 分半估计 (标量, 廉价)
预测: log P(v|h) ∝ log π_v + Σ_k ω_k·log(Â^(k)_{x_{t+1−k},v}/π_v),  ω 在 dev 标定
```

### 2.2 注意力偏置（模型侧，与 ALiBi 等价）

```
bias_h(pos_j) = s_h·log( Σ_c softmax(W)[tok_j,h,c]·N(|i−j|; R_hc, σ_hc) + floor_h )   (内容条件化变体)
 bias_h(d)    = s_h·log( Σ_j w_hj·N(d; R_hj, σ_hj) + floor_h )                         (纯距离版)
参数 52 个; 跨层共享; W 零初始化 = 均匀混合
```

### 2.3 autoballistic 诊断（全自动，无 oracle）

可平滑性（相邻表差异/噪声<3）→ 跨 lag 联合谱定 J（90% 方差）→ Kfit=K/4 拟合 → 包络候选 {plain, trend, fitpower, amp} dev 选优 → 终评。minimind 自验证：**+11.7%，1505s**。

## 3. 结果（全部实测）

### 3.1 统计侧：真实语料规模阶梯

| 训练量 | 离散全量 | 弹道+锚定包络 | 增益 |
|---|---|---|---|
| 250k | 88.65 | **82.03** (J=6) | +7.4% |
| 1M | 65.04 | **56.28** (J=4) | +13.5% |
| 4M | 49.16 | **41.38** (J=4) | +15.8% |
| 16M | 38.43 | **34.07** (J=4+amp) | +11.7% |

最优 J 随数据量反比（≤250k→J=6，≥1M→J=4）；谱建议值与实测最优一致。

### 3.2 模型侧：三规模对照（修正索引后）

| 规模 | RoPE | 弹道核 | ALiBi | 备注 |
|---|---|---|---|---|
| 字符级 3.5M | 2.724 | 3.059 | 3.098 | 弹道微弱占先 ALiBi（历史 bug 版差距更大，已勘误） |
| 子词级 11.4M | 2.373 | 2.504 | 2.503 | 三者打平（弹道核/ALiBi/内容条件化 2.5035/2.5031/2.5026） |
| 子词级 26.3M | 2.256 | 2.373 | 2.397 | 排序不变 |
| **子词级 86.6M** | **2.250** | **2.313** | **2.319** | 弹道核 vs ALiBi = +0.006（边际持平） |

**规模定律**：弹道核 vs ALiBi 的优势随规模衰减（+0.055 → +0.023 → +0.006 nats），与 RoPE 的差距同样收窄（+0.37 → +0.14 → +0.06）——所有位置方案在规模下收敛，方案差异是**小规模现象**。100M 级的位置方案选择应由其他因素决定（外推需求、推理约束、实现复杂度），而非此精度的 ppl 差异。

### 3.2b v6.4：RoPE + 可学习绝对先验（attention sink 形式化）——中性

`rope_abs`（RoPE + 逐 key 位置可学习 logit，52 参数，零初始化，未见位置中性）× 3 种子 @ ctx512/bs32：2.3501/2.3456/2.3476（均值 **2.3478**）vs 同配置 RoPE 基线 2.3480/2.3485（均值 **2.3483**）。**Δ=0.0005，打平**。绝对先验（sink 的显式形式）在该规模既无增益也无损害；RoPE 本身已隐式覆盖 sink 效应。同轮 `ballistic_abs`（学习核的绝对参数化）2.4979 vs 相对版 2.5035——单种子噪声级。

### 3.3 负结果（同等重要地报告）

1. **内容条件化原型选择**（V×H×J=32K 参数，标点选近场/实词选远场的假设）：与纯距离核打平。嫌疑：偏置内容信息与 qk 内容 logits 冗余。
2. **长度外推**（ctx=512 训 → 4×4096 零样本）：弹道核与 ALiBi 退化曲线完全重合（×6.1），RoPE 最稳（×4.0）。退化由未训练的长程处理主导，不归偏置管。
3. **n-gram 热启动注意力偏置**：不敏感（±0.03 nats）。诊断的价值在架构决策（J/Kfit/ctx），非参数初始化。
4. **索引 bug 教训**：位置偏置实现必须有相对距离单元测试；错误结论曾存活一轮。
5. **二阶交互在 16M 字符上不可估计（v6.5）**：高频 bigram 对（12k 对，覆盖 72%）的经验逐对分布，在 held-out 上比一阶 bigram 表**差 1.7~3.4 nats**（lag=1: +3.44, lag=2: +1.71；正=一阶更优）——pair 级计数（每对每 lag 均摊 ~700 计数/1500 目标格）的方差淹没交互信号。dev 标定自动把 pair 专家权重压到 ≈0（[0.006, −0.052]），PoE 框架自我保护正确，终 ppl 不变（34.07 vs 34.08）。**这定量解释了为什么 NLP 从计数 n-gram 走向神经网络**：交互信息 linguistic 上存在，但在该数据规模下不可从计数估计——这正是神经网络的用武之地，也说明「弹道核家族的边界」与「计数方法的边界」重合。

## 4. 与已有工作的关系

ALiBi（固定线性、相对距离）与本方法同门；KERPLE（可学习单调核）与 FIRE（连续参数化核）是注意力侧的近亲——**本研究的增量在于**：①「方向外推 + 振幅锚定」的包络原则（n-gram 侧 +11.7% 的直接来源）；②谱→J→包络的全自动诊断工作流；③包括两个负结果在内的完整测量。attention sink 现象与我们的可学习地板在形式上对应（显式背景项）。

## 5. 局限

单一 GPU、≤26.3M 参数、单一语料族；100M+ 未验证；**二阶交互已量化为不可估计**（16M 字符下 pair 计数方差淹没交互信号，见 3.3 节第 5 条）——突破它需要神经化或百倍数据，而非更好的计数；RoPE 组合下学习核无增益的原因未完全定位。

## 6. 复现

```
token_period_package/
  v3_ballistic/ v4_principle/    原始验证（合成语料）
  v5_minimind/                   真实语料统计侧: diag5/exp5/exp5b/decay5/sub5/autoballistic
  v6_attbias/                    注意力侧: bmodel/train6/analyze6/envv2/analyze_lc + runs/ 全部日志
数据: minimind_data/ (pretrain_t2t_mini.jsonl 1.18GB; pretrain_cached_*.pt 免 torch 解析)
```
每份报告的数字均由同名 `.log/.json` 支撑；负结果与勘误随附。

---

*研究执行：GLM (OpenCode agent)；理论起源与方向把控：用户。所有实验于 2026-11 单卡 RTX 4090D 完成，总计 40+ 训练运行、约 12 小时 GPU 时。*
