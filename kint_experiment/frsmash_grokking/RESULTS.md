# FRSMASH v3.6 K-Net/Grokking 验证结论

用真正的 FRSMASH v3.6 (2.1M 参数, hidden=128/L=4/H=8) 在 Modular Addition (p=113)
上跑 Grokking, 验证链接 E 节的 K-Net 命题。

## 现象基线 (已建立)
- **标准 FRSMASH + wd=0.1**: 完美 Grokking (s5400 test>0.5, 最终 test=0.998)
  记忆(s1000 train=1.0) → 顿悟(s5400) → 泛化(s10000 test=1.0). C 从 54166 压到 ~4500.
- **标准 FRSMASH + wd=0**: 不 Grokking (test≈0.002). **weight_decay 是 FRSMASH grokking 的必要条件.**

## K-Net (存在性惩罚) 验证结果

| 条件 | wd | γ | 最终 test_acc | grok 步 | K_est 行为 |
|------|----|----|--------------|---------|-----------|
| 标准 | 0.1 | 0 | **0.998** | **5400** | 训练中持续下降 |
| 标准 | 0 | 0 | 0.002 | — | — |
| K-Net | 0 | 1 | 0.026 | — | 太弱, 仍过拟合 |
| K-Net | 0 | 5 | 0.000 | — | **钉死在 K_int (偏差 0.11%)** |
| K-Net | 0.1 | 1 | 1.000 | **9300** | 同标准 |

## 逐条裁定 (链接 E 节预言)

### ✅ 成立: 存在性约束 C·V=K_int 是真实且可精确强制的
γ=5 条件下, K_est=C·V 全程锁定在 K_int=137907 (实测均值 138057, **偏差仅 0.11%**).
"守恒律"在真模型上数学可实现, Governor 机制本身工作正常.

### ❌ 不成立: 维持 C·V≥K_int 诱发 Grokking/智能 (核心预言)
- 无 wd 时: γ=1 太弱(照样过拟合, 不 grok); γ=5 太强(钉死 K_int 但 **train_acc 卡在 0.36, test=0**).
  没找到任何 γ 能在 wd=0 时 grok.
- **关键反例**: γ=5 模型精确满足 C·V=K_int, 却既不能拟合也不能泛化 ⇒ **满足约束 ≠ 智能**.
  约束是"可平凡满足"的——存在一个 C·V=K_int 但模型很蠢的定点.

### ❌ 不成立: K-Net 加速 Grokking
带 wd 时, K-Net(γ=1) grok 在 s9300, 比标准(s5400)**还慢**. 存在性惩罚没带来速度优势.

### ⏸ 未测: D 节"K-Net 必胜 SOTA"(Mamba/MoD/TTT 对比)
纯论述, 无对照实验, 无法裁定.

## 一句话最终结论
> 链接的"因果守恒律 C·V≥K_int"在 FRSMASH v3.6 上**作为机制是真实的** (可把 C·V 锁定到 0.11%),
> 但**作为智能的产生条件是假的** —— 满足该约束的模型可以完全不学习 (γ=5: train=0.36/test=0),
> 且它既不能取代 weight_decay 诱发 grokking, 也不能加速 grokking.
> "存在性约束"是必要性的诗意表达, 不是充分性的物理定律.

## 产出
- frsmash_grokking.py / analyze_frsmash_grokking.py
- frsmash_grokking_analysis.png
- log_{std_wd1e-1, std_wd0, knet_g1_wd0, knet_g5_wd0, knet_g1_wd1e-1}.csv
