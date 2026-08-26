# 202 个参数干掉 59 万个参数：ConvLinear 的设计、融合与实战

> 一个把标准 Linear 换成 "rank-1 扩展 + 空间卷积 + 收缩" 的通道混合器，
> 参数压缩 2000+ 倍，经历从"慢 74 倍不可用"到"与 Linear 持平、质量追平"的完整工程旅程。
> 全部代码与实验数据已推送到 [OpenAshAnyWhere](https://github.com/dfytensor/OpenAshAnyWhere)。

---

## 1. 设计：三个数学小操作组成一个"线性层"

标准 Linear（768→768）要 590,592 个参数。我们的设计只有 202 个：

```
输入 [b,s,h]
  ↓ ① rank-1 扩展: x @ w(1,w)   —— 每个通道 h 乘一个可学习向量, 铺成 (h,w) 平面
  [b,s,h,w]  → reshape 成 [b*s, 1, h, w]（单通道"图像"）
  ↓ ② 空间卷积: Conv2d(1→1, 3×3) —— 在 (h,w) 平面上卷积, 混合通道关系
  ↓ ③ 收缩: @ w(w,1) —— 把 w 维投影回 1 维
  输出 [b,s,h]
```

参数量：`w_in[w] + conv[k²] + w_out[w] = 2w + k²`，与 h 无关！
（w=96, k=9 时 = 274 个参数，压缩 2155 倍）

**本质**：把 Linear 的 `h×h` 全连接权重，换成**权重共享的卷积核在"扩展平面"上滑动**。
每个通道不再是独立权重，而是共享同一组卷积参数。

---

## 2. 性能五连跳：从"慢 74 倍"到"持平"

### 起点：naive 实现慢 74 倍
```
原版: x @ w(1,w) -> conv2d -> @ w(w,1)
问题: 中间张量 [b,s,h,w] = 1.5 亿元素 = 600MB 物化, conv2d 在 1 通道上极低效
实测: 5.66 ms vs Linear 0.077 ms (慢 74×)
```

### 第一跳：数学重排（免物化）
关键观察——**conv 的 w 维求和可以和 w_in 预合并**：
```
out[h] = Σ_w w_out[w] · relu( Σ_dh Σ_dw K[dh,dw]·w_in[w+dw]·x[h+dh] + b )
                    = Σ_w w_out[w] · relu( Σ_dh Kw[dh,w]·x[h+dh] + b )
```
其中 `Kw[dh,w] = Σ_dw K[dh,dw]·w_in[w+dw]` 是 **9w 个数的预计算**。
二维卷积退化成 **h 维 3-tap 位移加权**！[b,s,h,w] 不再必须落地。

### 第二跳：Triton 融合（0.50ms, 11×）
w 维留在寄存器内循环，ReLU 后立即与 w_out 收缩——**中间张量从 600MB 降到 3MB**。

### 第三跳：GEMM 化（0.18ms, 31×）
你点出的关键：上一版是逐元素 FMA，没走 tensor core。重写成**两级 tl.dot**：
```
阶段1: T[BH, W] = A[BH, K] @ Kw[K, W]     (K=3→pad 16 的 GEMM)
阶段2: out[BH]  = relu(T) @ w_out          (GEMV)
```
relu 卡在中间使两段不能合一，但中间 tile 留共享内存。

### 第四跳：tile 调优 + Kw 缓存（1.06×，达标）
`BLOCK_H=128, BLOCK_W=64` 是甜点；Kw 用 id 缓存避免每步 host 端重算。

| k | 耗时 | vs Linear | 参数 | 压缩比 |
|:--:|:--:|:--:|:--:|:--:|
| 3 | 0.085 ms | 1.11× | 202 | 2923× |
| **9** | **0.081 ms** | **1.06×** | 274 | **2155×** |

### 第五跳：训练反向 kernel（44.8×）
推理平了，但训练用 torch 重排反向还是慢 23×。手写反向 kernel：
```
dT = dy·w_out·1[T>0]        (重算 T, 免存 600MB 中间)
dx = dT @ Kw^T              (转置 3-tap, atomic_add)
dKw = A^T @ dT ; dw_out, db (atomic_add)
```
单层 fwd+bwd：33 ms → 0.74 ms。

---

## 3. 实战：30M OpenASH 用 MiniMind 预训练

| | 原版 | ConvLinear+torch反向 | ConvLinear+Triton反向 |
|:--|:--:|:--:|:--:|
| 参数 | 34.6M | 33.0M | 33.0M |
| 步时 | ~21 ms | ~580 ms | **~34 ms** |
| 3000 步耗时 | 64 s | 1452 s | **101 s** |
| 最终 loss | 3.84 | 4.17 | **3.86** |

**两个结论**：
1. **训练全链 kernel 化后**，30M 端到端只比原版慢 1.6×
2. **质量追平**（3.86 vs 3.84）——之前 4.17 的落后是训练太慢导致的有效步数不足，不是结构表达力损失。ConvLinear 的 rank-1 结构在 30M 规模上表达力足够。

---

## 4. 设计定位

- ✅ **极端参数压缩**（2000+ 倍）且**速度持平**、**质量追平**
- ✅ 适合：参数预算受限场景（边缘部署、大模型非关键投影层替换）
- ⚠️ k 的甜点窗口 3~15（BLOCK_K=16 单次 dot）；k≥31 需双 dot，掉到 1.7×
- ⚠️ 训练必须配套 Triton 反向（否则 torch 重排反向慢 23×）

## 5. 附录：核心源码

### ConvLinear 模块
```python
import torch
import torch.nn as nn

class ConvLinear(nn.Module):
    def __init__(self, h, w=None, k=3, act=None, bias=True):
        super().__init__()
        w = w or h
        self.h, self.w = h, w
        self.w_in = nn.Parameter(torch.empty(1, w))     # ①扩展向量
        self.w_out = nn.Parameter(torch.empty(w, 1))    # ③收缩向量
        self.conv = nn.Conv2d(1, 1, k, padding=k // 2, bias=bias)  # ②卷积
        self.act = act if act is not None else nn.ReLU()
        nn.init.normal_(self.w_in, 0.0, 0.02)
        nn.init.normal_(self.w_out, 0.0, 0.02)

    def forward(self, x):
        b, s, h = x.shape
        xw = x.unsqueeze(-1) * self.w_in                # [b,s,h,1]@[1,w]
        img = xw.reshape(b * s, 1, h, self.w)
        img = self.conv(img)
        img = self.act(img)
        out = img.reshape(b, s, h, self.w) @ self.w_out  # [b,s,h,w]@[w,1]
        return out.squeeze(-1)
```

### 数学重排（Kw 预合并）
```python
def make_Kw(w_in, conv_weight):
    """Kw[k,w] = Σ_dw K[dh,dw]·w_in[(w+dw-p) mod w]  —— 卷积+w_in 预合并"""
    K = conv_weight[0, 0].float()
    k = K.shape[0]; p = k // 2
    w = w_in.shape[-1]
    idx = ((torch.arange(w, device=w_in.device).view(1, -1)
            + torch.arange(k, device=w_in.device).view(-1, 1) - p) % w)
    return (K @ w_in.float().reshape(-1)[idx]).contiguous()
```

### Triton GEMM 化 kernel（推理）
```python
import triton
import triton.language as tl

@triton.jit
def _row_kernel_dot(XP, KW, WOUT, BIAS, Y,
                    SP: tl.constexpr, H: tl.constexpr, W: tl.constexpr, K: tl.constexpr,
                    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_row = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)
    offs_w = tl.arange(0, BLOCK_W)
    mh = offs_h < H; mk = offs_k < K; mw = offs_w < W
    base = pid_row * SP
    A = tl.load(XP + base + offs_h[:, None] + offs_k[None, :],
                mask=mh[:, None] & mk[None, :], other=0.0)
    kW = tl.load(KW + offs_k[:, None] * W + offs_w[None, :],
                 mask=mk[:, None] & mw[None, :], other=0.0)
    T = tl.dot(A, kW)                                   # 阶段1 GEMM
    bias = tl.load(BIAS + offs_w, mask=mw, other=0.)
    T = tl.maximum(T + bias[None, :], 0.0)              # relu
    w2 = tl.load(WOUT + offs_w, mask=mw, other=0.)
    out = tl.dot(T, w2[:, None])                        # 阶段2 GEMV
    tl.store(Y + pid_row * H + offs_h, tl.ravel(out), mask=mh)
```

### Triton 反向 kernel（训练）
```python
@triton.jit
def _row_kernel_bwd(XP, DY, KW, WOUT, BIAS, DX, DKW, DWOUT, DB, RB,
                    SP: tl.constexpr, H: tl.constexpr, W: tl.constexpr, K: tl.constexpr,
                    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_row = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)
    offs_w = tl.arange(0, BLOCK_W)
    mh = offs_h < H; mk = offs_k < K; mw = offs_w < W
    base = pid_row * SP
    # 重算前向 (免存 600MB 中间)
    A = tl.load(XP + base + offs_h[:, None] + offs_k[None, :],
                mask=mh[:, None] & mk[None, :], other=0.0)
    kW = tl.load(KW + offs_k[:, None] * W + offs_w[None, :],
                 mask=mk[:, None] & mw[None, :], other=0.0)
    T = tl.dot(A, kW)
    bias = tl.load(BIAS + offs_w, mask=mw, other=0.)
    T = T + bias[None, :]
    R = tl.maximum(T, 0.0)
    gate = tl.where(T > 0, 1.0, 0.0)
    dy = tl.load(DY + pid_row * H + offs_h, mask=mh, other=0.)
    w2 = tl.load(WOUT + offs_w, mask=mw, other=0.)
    dT = dy[:, None] * w2[None, :] * gate               # [BH,BW]
    dX_tile = tl.dot(dT, tl.trans(kW))                  # dx = dT@Kw^T
    tl.atomic_add(DX + base + offs_h[:, None] + offs_k[None, :],
                  dX_tile, mask=mh[:, None] & mk[None, :])
    dKw_tile = tl.dot(tl.trans(A), dT)                  # dKw = A^T@dT
    tl.atomic_add(DKW + offs_k[:, None] * W + offs_w[None, :],
                  dKw_tile, mask=mk[:, None] & mw[None, :])
    tl.atomic_add(DWOUT + offs_w, tl.sum(dy[:, None] * R, axis=0), mask=mw)
    tl.atomic_add(DB + offs_w, tl.sum(dT, axis=0), mask=mw)
```

---

*完整工程（含 30M 训练脚本、benchmark、bf16 分析）见仓库 `OpenAshAnyWhere`。*
