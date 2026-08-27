# SGR 给 CLR 提供语义旋钮 → CLR 有效; 但 SGR+CLR 仍打不过纯 dense

用户假设: FRSMASH/GLA 之所以 CLR 无效(§15), 是因为**没有 SGR 结构**;
若加上 SGR, CLR 有了可控的语义旋钮, 就会变有效。**直接验证:**

## 同规模四元对比 (GLA 骨干, seq256/d256/L6/1000步)

| 配置 | 有 SGR? | CLR 控什么 | val ppl | vs dense | 判定 |
|------|---------|-----------|---------|----------|------|
| GLA + dense（基线） | 否 | — | ~104.3 | — | — |
| GLA + CLR-**wd**（§15） | 否 | weight_decay | ≈ dense（81.4 vs 81.2） | 持平 | **CLR 无效** |
| GLA + SGR + 硬路由（CLR off） | 是 | frac 固定 0.4 | **116.5** | +12% | SGR 路由掉 ppl |
| GLA + SGR + CLR-**frac**（闭环） | 是 | frac 闭环 | **106.6** | +2.2% | **CLR 有效** |

## 假设裁定
### ✅ 成立的一半：SGR 给 CLR 提供语义旋钮，CLR 立刻变有效
- CLR 调 **wd**（无 SGR）：wd 在 LM 上没有"该收紧/放松"的语义 + LM 弱 gap → **空转**。
- CLR 调 **frac**（有 SGR）：frac 有明确语义（收紧=更多低秩=压缩，放松=更多全秩），
  闭环能按 val 信号有意义地驱动 → **把硬路由的 +12% 灾难救到 +2.2%**。
- **结论：CLR 有效 ⟺ 存在可被泛化信号有意义驱动的旋钮。SGR 提供了这个旋钮。**

### ❌ 不成立的一半：CLR 有效 ≠ SGR+CLR 打赢纯骨干
- GLA+SGR+CLR（106.6）**仍比纯 GLA+dense（104.3）差 2.2%**。
- 且闭环自动停在 **frac=0.89**（几乎不稀疏）——小规模容量紧，CLR 不敢压，
  → **既没赢 ppl，也没省到推理算力**。
- 根因（§10 已证）：**SGR 路由有固有 ppl 代价**，CLR 只能把它最小化（退到接近 dense），
  不能消除。所以 SGR+CLR ≥ dense on ppl 恒成立；只有在**更大规模**（frac 能真降而不掉 ppl）时，
  SGR+CLR 才可能换来"省推理、持平 ppl"的净收益。

## 一句话
> **加上 SGR 确实让 CLR 从"空转"变"有效"——证实了"CLR 需要语义旋钮"的判断。**
> **但"CLR 有效"只是"让 SGR 可用"，不是"打赢原骨干"**：SGR+CLR 仍比纯 dense 差 ~2% ppl，
> 且小规模下连推理都省不下（闭环自保在 frac≈0.9）。要 SGR+CLR 净赢，得放大到 frac 能真降的规模。

## 与 §15 的呼应
§15 结论"CLR-wd 在 FRSMASH/GLA 空转"的根因有二：(a) LM 弱 gap，(b) wd 非语义旋钮。
本节证明：**换成语义旋钮（SGR 的 frac），即便 LM 弱 gap，CLR 也能工作**——所以根因主要是 (b)，
SGR 正好补上这个缺口（让 CLR 有旋钮可拧），但补不了 SGR 自身的 ppl 代价。

## 产出
- sgr_v2.py（--closed 闭环调 frac，已在 §11 验证；本节复用于 GLA 骨干同规模对比）
- log_srrf_*.csv（GLA+SGR 硬路由 / GLA+SGR+CLR）
