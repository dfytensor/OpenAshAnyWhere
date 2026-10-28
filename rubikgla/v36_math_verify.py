#!/usr/bin/env python3
"""v36-MultiState 独立数学验证 (不训练, 纯数值, ~1分钟).

A. 分段接续等价: kernel 两段扫描 (s0 注入) == kernel 单次全扫描 == 纯PyTorch参考
   → "每 chunk 末状态存 bank / 用 s0 续扫" 的数学基础.
B. 核心主张定量: 远程 marker 信号保持系数 c (差分法精确测)
   单 state: c ≈ exp(-rate·dist) → 远程归零
   多 state: c ≈ exp(-rate·128) 与总长无关
C. 零初始化检索信号: softmax(q·summary(Sᵢ)) 能否排对 chunk (dead-gate 风险评估)
D. bank 内存核算
"""
import sys, torch, math
sys.path.insert(0, r"F:\夸克\rubikgla")
from delta_triton import delta_scan_states

DEV = "cuda"
torch.manual_seed(0)
DT = torch.bfloat16


# ---- 纯 PyTorch 参考 (与 delta_triton 语义一致): S_t = λ(S_{t-1} − η k (kᵀS_{t-1})) + k vᵀ ----
def scan_ref(k, v, rate, eta, s0=None):
    B, L, G, dh = k.shape
    S = torch.zeros(B, G, dh, dh, device=k.device, dtype=torch.float32) if s0 is None else s0.float().clone()
    for t in range(L):
        kt, vt = k[:, t].float(), v[:, t].float()
        lam = torch.exp(-rate[:, t].float())
        et = eta[:, t].float()
        rr = (kt.unsqueeze(-1) * S).sum(-2)
        S = lam.unsqueeze(-1) * (S - et.unsqueeze(-1) * kt.unsqueeze(-1) * rr.unsqueeze(-2)) \
            + kt.unsqueeze(-1) * vt.unsqueeze(-2)
    return S


def const_inputs(B, L, G, DH, rate, eta, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = (torch.randn(B, L, G, DH, generator=g) * 0.3).to(DT).to(DEV)
    v = (torch.randn(B, L, G, DH, generator=g) * 0.3).to(DT).to(DEV)
    r = torch.full((B, L, G, DH), rate, dtype=DT, device=DEV)
    e = torch.full((B, L, G, DH), eta, dtype=DT, device=DEV)
    return k, v, r, e


def zeros_q(B, L, H, DH):
    return torch.zeros(B, L, H, DH, dtype=DT, device=DEV)


print("=" * 72)
print("A. 分段接续等价性 (s0 注入 = bank 接力的数学基础)")
print("=" * 72)
B, L, KV, SL, KVG, DH = 2, 256, 4, 1, 1, 32
G, H = KV * SL, KV * KVG
CH = 128
k, v, r, e = const_inputs(B, L, G, DH, rate=2e-3, eta=0.1, seed=1)
q = (torch.randn(B, L, H, DH) * 0.5).to(DT).to(DEV)

y_full, S_full = delta_scan_states(k, v, r, e, q, KV, SL, KVG)
y_1, S_1 = delta_scan_states(k[:, :CH], v[:, :CH], r[:, :CH], e[:, :CH], q[:, :CH], KV, SL, KVG)
y_2, S_2 = delta_scan_states(k[:, CH:], v[:, CH:], r[:, CH:], e[:, CH:], q[:, CH:], KV, SL, KVG, s0=S_1)
d_seg = (S_2 - S_full).abs().max().item()
S_ref = scan_ref(k, v, r, e)
d_ref = (S_full - S_ref).abs().max().item()
d_s0ref = (S_2 - scan_ref(k[:, CH:], v[:, CH:], r[:, CH:], e[:, CH:], s0=S_1)).abs().max().item()
print(f"  两段(s0注入) vs 单次全扫描   max|ΔS| = {d_seg:.3e}   (期望 ~1e-5, fp32 舍入级)")
print(f"  kernel 全扫描 vs 纯PyTorch   max|ΔS| = {d_ref:.3e}")
print(f"  两段(s0注入) vs PyTorch带s0  max|ΔS| = {d_s0ref:.3e}")
print(f"  读出 y 前半段一致性 max|Δy| = {(y_full[:, :CH].float() - y_1.float()).abs().max().item():.3e} (bf16 输出)")
A_OK = d_seg < 1e-3 and d_ref < 5e-2
print(f"  ⇒ 测试 A {'通过' if A_OK else '失败'}: 分段扫描+s0 续传 与 全扫描 {'等价' if A_OK else '不等价'}")

print()
print("=" * 72)
print("B. 核心主张: 远程 marker 信号保持 (差分法, eta=0 纯衰减累加, 理论可解析)")
print("   c_measured = <ΔS, k0 v0ᵀ> / ‖k0v0ᵀ‖ ;  单state理论 c=exp(-rate·dist)")
print("=" * 72)
B, KV, SL, KVG, DH = 4, 4, 1, 1, 32
G, H = KV * SL, KV * KVG
RATE, CH = 2e-3, 128

def marker_retention(k, v, dist_to_end, CH, eta_val):
    """marker 在距末尾 dist_to_end 处的 chunk 起点; 返回 (c_单, c_多)."""
    k0 = torch.randn(B, G, DH, generator=torch.Generator().manual_seed(7))
    k0 = (k0 / k0.norm(dim=-1, keepdim=True)).to(DT).to(DEV)
    v0 = torch.randn(B, G, DH, generator=torch.Generator().manual_seed(8))
    v0 = (v0 / v0.norm(dim=-1, keepdim=True)).to(DT).to(DEV)
    pos = k.shape[1] - dist_to_end  # marker 位置 (chunk 起点)
    # 带 marker
    kw, vw = k.clone(), v.clone()
    kw[:, pos], vw[:, pos] = k0, v0
    _, S_w = delta_scan_states(kw, vw, r, e, zeros_q(*k.shape[:2], H, DH), KV, SL, KVG)
    # 不带 marker (对照) → 差分 = 纯 marker 贡献
    _, S_wo = delta_scan_states(k, v, r, e, zeros_q(*k.shape[:2], H, DH), KV, SL, KVG)
    dS = (S_w - S_wo).float()
    P = torch.einsum("bg i,bg j->bgij", k0.float(), v0.float())  # k0 v0ᵀ
    c_single = (dS * P).sum() / P.norm() ** 2
    # 多 state: marker 所在 chunk 独立扫描 (从零)
    cs = pos - pos % CH
    _, Sw_c = delta_scan_states(kw[:, cs:cs + CH], vw[:, cs:cs + CH], r[:, cs:cs + CH], e[:, cs:cs + CH],
                                zeros_q(B, CH, H, DH), KV, SL, KVG)
    _, Swo_c = delta_scan_states(k[:, cs:cs + CH], v[:, cs:cs + CH], r[:, cs:cs + CH], e[:, cs:cs + CH],
                                 zeros_q(B, CH, H, DH), KV, SL, KVG)
    dSc = (Sw_c - Swo_c).float()
    d_in_chunk = (cs + CH) - pos  # marker 到该 chunk 末状态的距离
    c_multi = (dSc * P).sum() / P.norm() ** 2
    th_s, th_m = math.exp(-eta_val * 0), None  # eta=0 时理论精确
    return c_single.item(), c_multi.item(), d_in_chunk

print(f"  {'总长L':>6} {'查询距离':>8} | {'单state实测':>11} {'单state理论':>11} {'比值':>6} | {'多state实测':>11} {'多state理论':>11} {'比值':>6} | 多/单增益")
for L in (256, 1024, 4096):
    k, v, r, e = const_inputs(B, L, G, DH, rate=RATE, eta=0.0, seed=L)
    dist = L  # marker 在位置 0, 状态取末尾 → 距离 L
    c_s, c_m, dch = marker_retention(k, v, dist, CH, 0.0)
    th_s = math.exp(-RATE * dist)
    th_m = math.exp(-RATE * dch)
    print(f"  {L:>6} {dist:>8} | {c_s:>11.3e} {th_s:>11.3e} {c_s/th_s:>6.2f} | {c_m:>11.3e} {th_m:>11.3e} {c_m/th_m:>6.2f} | {c_m/max(c_s,1e-30):.0f}×")

# eta=0.1 (现实设置): 理论不再精确解析, 看方向
k, v, r, e = const_inputs(B, 4096, G, DH, rate=RATE, eta=0.1, seed=4096)
c_s, c_m, dch = marker_retention(k, v, 4096, CH, 0.1)
print(f"  [eta=0.1, L=4096] 单state c={c_s:.3e}  多state c={c_m:.3e}  增益 {c_m/max(c_s,1e-30):.0f}×  (eta 数据依赖擦除只会进一步惩罚单state)")

print()
print("=" * 72)
print("C. 零初始化检索信号: softmax 排对 chunk 的能力 (无训练, dead-gate 风险)")
print("   marker 放入随机 chunk c*, 检索打分 q=k0, summary(Sᵢ)=Sᵢ.mean(-1)")
print("=" * 72)
B, K, KV, SL, KVG, DH = 32, 16, 4, 1, 1, 32
G, H = KV * SL, KV * KVG
hits_lin, hits_eng, ranks = 0, 0, []
gseeds = torch.Generator().manual_seed(99)
for trial in range(32):
    c_star = int(torch.randint(0, K, (1,), generator=gseeds))
    k, v, r, e = const_inputs(B, K * CH, G, DH, rate=2e-3, eta=0.05, seed=1000 + trial)
    k0 = torch.randn(B, G, DH, generator=torch.Generator().manual_seed(7))
    k0 = (k0 / k0.norm(dim=-1, keepdim=True)).to(DT).to(DEV)
    v0 = torch.randn(B, G, DH, generator=torch.Generator().manual_seed(8))
    v0 = (v0 / v0.norm(dim=-1, keepdim=True)).to(DT).to(DEV)
    k[:, c_star * CH] = k0
    v[:, c_star * CH] = v0
    bank = []
    for c in range(K):
        _, Sc = delta_scan_states(k[:, c * CH:(c + 1) * CH], v[:, c * CH:(c + 1) * CH],
                                  r[:, c * CH:(c + 1) * CH], e[:, c * CH:(c + 1) * CH],
                                  zeros_q(B, CH, H, DH), KV, SL, KVG)
        bank.append(Sc.float())
    bank = torch.stack(bank, dim=1)                       # (B,K,G,DH,DH)
    summ = bank.mean(dim=-1)                              # (B,K,G,DH)
    scores_lin = torch.einsum("bgd,bkgd->bgk", k0.float(), summ)   # q·summary
    scores_eng = torch.einsum("bkgij,bgj->bkgi", bank, k0.float()).norm(dim=(-2, -1))  # ‖Sᵢᵀk0‖ 能量
    pred_lin = scores_lin.argmax(dim=-1)                  # (B,G)
    pred_eng = scores_eng.argmax(dim=-1)
    hits_lin += (pred_lin == c_star).float().mean().item()
    hits_eng += (pred_eng == c_star).float().mean().item()
    ranks.append((scores_lin.argsort(dim=-1) == c_star).nonzero()[:, -1].float().mean().item())
print(f"  32 trials, K=16 chunks (随机猜 top-1 = 6.25%)")
print(f"  线性 summary (q·mean(Sᵢ)):        top-1 = {hits_lin/32*100:.1f}%   平均 rank = {sum(ranks)/len(ranks):.1f}")
print(f"  能量打分 (‖Sᵢᵀ k0‖):             top-1 = {hits_eng/32*100:.1f}%")
print(f"  ⇒ 初始化即存在可学习信号: {'是' if hits_lin/32 > 0.15 else '弱 — 需依赖端到端训练检索头 (与设计文档决策一致)'}")

print()
print("=" * 72)
print("D. bank 内存核算 (每层)")
print("=" * 72)
for L in (4096,):
    Kc = L // CH
    per = G * DH * DH
    bank_kb = Kc * per * 2 / 1024
    kv_kb = L * 64 * 2 * 2 / 1024  # softmax-KV: K+V 各 D=64, fp16
    print(f"  L={L}: K={Kc} chunks × G={G} × {DH}×{DH} = {bank_kb:.0f} KB (bf16 bank)")
    print(f"  对照 softmax-KV @L={L} (D=64, fp16): ~{kv_kb:.0f} KB → 压缩比 ~{kv_kb/bank_kb:.1f}× (本测试 toy 维度; 设计文档 15× 按真实 v31 维度)")
print("\nDONE")
