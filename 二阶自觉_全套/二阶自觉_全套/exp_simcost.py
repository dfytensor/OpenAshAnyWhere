import numpy as np, gzip, time, math

# ================= 通用 ECA (零背景, 避免边界效应) =================
def step(S, rule):
    M, W = S.shape
    z = np.zeros((M,1), np.uint8)
    L = np.concatenate([z, S[:, :-1]], axis=1)
    R = np.concatenate([S[:, 1:], z], axis=1)
    idx = (L.astype(np.int64)<<2)|(S.astype(np.int64)<<1)|R.astype(np.int64)
    return ((rule >> idx) & 1).astype(np.uint8)

def run_k(s0, rule, k):
    S = s0.copy()
    for _ in range(k): S = step(S, rule)
    return S

# ================= S2-A/B: 线性规则的多项式捷径 vs 非线性规则 =================
W = 512
def rot(a, d, W):
    d %= W
    if d == 0: return a
    return ((a << d) | (a >> (W-d))) & ((1<<W)-1)
def pmul(a, b, W):
    r = 0; bb = b; ops = 0
    while bb:
        low = bb & -bb; i = low.bit_length()-1
        r ^= rot(a, i, W); ops += 1; bb ^= low
    return r, ops
def ppow(base, k, W):
    r = 1; b = base; ops = 0
    while k:
        if k & 1:
            r, o = pmul(r, b, W); ops += o
        k >>= 1
        if k:
            b, o = pmul(b, b, W); ops += o
    return r, ops
def rot_state(s, d, W):
    d %= W
    if d==0: return s
    return ((s << d) | (s >> (W-d))) & ((1<<W)-1)

def step_ring(s, rule, W):
    l = rot_state(s, 1, W); c = s; r = rot_state(s, W-1, W)
    idx = (l<<2)|(c<<1)|r
    return ((rule>>idx)&1)   # 大整数逐位: rule>>idx 对大整数不适用!
# 大整数逐位查表不可行 -> 用 numpy 环形实现做对照
def step_ring_np(a, rule):
    l = np.roll(a,1); r = np.roll(a,-1)
    idx = (l.astype(np.int64)<<2)|(a.astype(np.int64)<<1)|r.astype(np.int64)
    return ((rule>>idx)&1).astype(np.uint8)

print("="*84)
print("S2-A  线性规则的多项式捷径正确性校验 (环 W=%d)"%W)
print("="*84)
mask=(1<<W)-1
s0_int = int.from_bytes(np.random.default_rng(1).bytes(W//8), 'big') & mask
rule90_T = (rot(1,1,W) ^ rot(1,W-1,W))     # 算子 x + x^{-1}
s0_np = np.array([(s0_int>>i)&1 for i in range(W)], np.uint8)
for k in [1,2,3,5,8,13,16,31,32,64]:
    Tk, ops = ppow(rule90_T, k, W)
    # 应用: S_k = XOR over set bits j of Tk of rot(S_0, j)
    acc = 0; bb = Tk
    while bb:
        low = bb & -bb; j = low.bit_length()-1
        acc ^= rot_state(s0_int, j, W); bb ^= low
    # 直接模拟对照
    cur = s0_np.copy()
    for _ in range(k): cur = step_ring_np(cur, 90)
    direct = int(''.join(map(str, cur.tolist()[::-1])), 2)
    print(f"  k={k:3d}  捷径==直接模拟: {acc==direct}   多项式乘法次数={ops:4d}   直接模拟步数={k}")

print()
print("="*84)
print("S2-B  计算开销 rho(k) = 预测 k 步的最小开销 / 1 步开销")
print("="*84)
print(f"{'k':>5} |{'Rule90 捷径(pmul)':>18}{'Rule90 rho':>11} |{'Rule30 rho(直接)':>17}{'Rule110 rho':>12}")
for k in [1,2,4,8,16,32,64,128,256]:
    _, o90 = ppow(rule90_T, k, W)
    # 一次 pmul 的基准开销 ~ popcount(b) 次旋转; 用 W=512 的实测时间更公平
    print(f"{k:5d} |{o90:18d}{o90/max(math.log2(k+1),1):11.2f} |{k:17d}{k:12d}")

print()
print("="*84)
print("S2-B'  实测墙钟时间 (W=%d, 重复 200 次取总时长, 单位 ms)"%W)
print("="*84)
REP=200
print(f"{'k':>5} |{'Rule90 捷径 ms':>15}{'Rule90 直接 ms':>15} |{'Rule30 直接 ms':>15}{'加速比':>9}")
for k in [1,4,16,64,256]:
    s_np = np.array([(s0_int>>i)&1 for i in range(W)], np.uint8)
    t0=time.perf_counter()
    for _ in range(REP): ppow(rule90_T, k, W)
    t_short=(time.perf_counter()-t0)*1000
    t0=time.perf_counter()
    for _ in range(REP):
        cur=s_np.copy()
        for _ in range(k): cur=step_ring_np(cur,90)
    t_dir90=(time.perf_counter()-t0)*1000
    t0=time.perf_counter()
    for _ in range(REP):
        cur=s_np.copy()
        for _ in range(k): cur=step_ring_np(cur,30)
    t_dir30=(time.perf_counter()-t0)*1000
    print(f"{k:5d} |{t_short:15.2f}{t_dir90:15.2f} |{t_dir30:15.2f}{t_dir90/max(t_short,1e-9):9.2f}x")

# ================= S2-C: 时空图可压缩性 =================
print()
print("="*84)
print("S2-C  时空图 gzip 压缩率 (不可压缩 = 计算不可约的代理指标)")
print("="*84)
print(f"{'规则':>8}{'名称':>14} |{'原始 bytes':>11}{'压缩后':>10}{'压缩率':>9} | 判读")
K=256
for rule,name in [(90,'Rule90 线性'),(150,'Rule150 线性'),(30,'Rule30 非线性'),(110,'Rule110 非线性'),(0,'Rule0 平凡')]:
    s=np.random.default_rng(7).integers(0,2,W,dtype=np.uint8)
    rows=[s.copy()]
    cur=s.copy()
    for _ in range(K):
        cur=step_ring_np(cur,rule); rows.append(cur.copy())
    M=np.packbits(np.array(rows))
    raw=M.tobytes(); comp=gzip.compress(raw,9)
    ratio=len(comp)/len(raw)
    tag='可压缩->可约' if ratio<0.5 else ('不可压缩->不可约' if ratio>0.85 else '中间')
    print(f"{rule:>8}{name:>14} |{len(raw):11d}{len(comp):10d}{ratio:9.3f} | {tag}")

# ================= S2-D: ANF 代数次数与项数 (严格不可约性度量) =================
print()
print("="*84)
print("S2-D  元级映射 f_k: {0,1}^(2k+1) -> {0,1} 的 ANF 次数/项数 (严格复杂度)")
print("="*84)
def anf_mobius(tt, n):
    a = tt.copy().astype(np.uint8)
    N = 1<<n
    for i in range(n):
        sel = np.arange(N, dtype=np.int64)
        sel = sel[(sel>>i)&1 == 1]
        a[sel] ^= a[sel ^ (1<<i)]
    return a
def anf_stats(rule, k):
    n = 2*k+1; Wd = 4*k+5; N = 1<<n
    xs = ((np.arange(N, dtype=np.int64)[:,None] >> np.arange(n)) & 1).astype(np.uint8)
    S = np.zeros((N, Wd), np.uint8)
    off = (Wd-n)//2
    S[:, off:off+n] = xs
    cur = S
    for _ in range(k): cur = step(cur, rule)
    tt = cur[:, Wd//2].astype(np.uint8)
    a = anf_mobius(tt, n)
    nz = np.nonzero(a)[0]
    deg = int(max((bin(int(j)).count('1') for j in nz), default=0))
    return deg, int(len(nz)), n
print(f"{'k':>4}{'n=2k+1':>7} |{'Rule90 次数':>12}{'Rule90 项数':>12} |{'Rule30 次数':>12}{'Rule30 项数':>12} |{'项数比 30/90':>13}")
for k in [1,2,3,4,5,6,7]:
    d90,t90,n = anf_stats(90,k)
    d30,t30,_ = anf_stats(30,k)
    print(f"{k:4d}{n:7d} |{d90:12d}{t90:12d} |{d30:12d}{t30:12d} |{t30/max(t90,1):13.1f}x")

print()
print("  Rule110 (通用计算) 对照")
print(f"{'k':>4} |{'Rule110 次数':>13}{'Rule110 项数':>13}")
for k in [1,2,3,4,5]:
    d,t,n = anf_stats(110,k)
    print(f"{k:4d} |{d:13d}{t:13d}")
