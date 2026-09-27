import math, random

def sim(x, B_M, sigma, v_M=0.0, T=40000, seed=0):
    """标量: e_{t+1} = (1-x)e_t + Delta(对抗,B_M) + xi(sigma) - drift(v_M) ; x=eta*mu"""
    rnd = random.Random(seed)
    e = 2.0
    acc = 0.0; n = 0
    for t in range(T):
        delta = B_M * (1.0 if e >= 0 else -1.0)
        e = (1.0-x)*e + delta - v_M*rnd.gauss(0,1) + sigma*rnd.gauss(0,1)
        if t >= T//2:
            acc += abs(e); n += 1
    return acc/max(n,1)

def theory(x, B_M, sigma, v=0.0):
    det = (B_M/x)**2 if B_M>0 else 0.0
    den = 2*x - x*x
    noi = (sigma**2+v**2)/den if den>0 else float('inf')
    return math.sqrt(det+noi)

def orig(x, B_M, sigma):
    return (B_M+sigma)/x

print("="*76)
print("表1  收敛半径 r*: 原文公式 vs 精确(正交叠加) vs 数值模拟")
print("="*76)
print(f"{'x=eta*mu':>9}{'B_M':>8}{'sigma':>8} |{'原文':>9}{'精确':>9}{'模拟':>9} | 原文偏差")
for x,B,s in [(0.2,0.1,0.1),(0.5,0.1,0.1),(1.0,0.1,0.1),(1.5,0.1,0.1),
              (0.2,0.5,0.05),(1.0,0.5,0.05),(1.5,0.5,0.05),
              (0.2,0.05,0.5),(1.0,0.05,0.5),(1.5,0.05,0.5)]:
    sm=sim(x,B,s); t=theory(x,B,s); o=orig(x,B,s)
    print(f"{x:9.2f}{B:8.3f}{s:8.3f} |{o:9.4f}{t:9.4f}{sm:9.4f} | {100*(o-sm)/max(sm,1e-9):+7.1f}%")

print()
print("="*76)
print("表2  最优步长 x*=eta*mu 由 B_M/sigma 比决定 (数值扫描)")
print("="*76)
xs=[0.05+0.05*i for i in range(39)]
print(f"{'B_M':>7}{'sigma':>7}{'B/s':>7} |{'x*_理论':>9}{'x*_模拟':>9}{'r*(x*)':>9}{'r*(x=1)':>9}")
for B,s in [(0.5,0.01),(0.2,0.1),(0.05,0.3),(0.01,0.5)]:
    th=[theory(x,B,s) for x in xs]; sm=[sim(x,B,s,T=20000) for x in xs]
    print(f"{B:7.3f}{s:7.3f}{B/max(s,1e-9):7.2f} |{xs[min(range(39),key=lambda i:th[i])]:9.2f}"
          f"{xs[min(range(39),key=lambda i:sm[i])]:9.2f}{min(th):9.4f}{theory(1.0,B,s):9.4f}")

print()
print("="*76)
print("表3  元级目标漂移 v_M 等效于额外噪声 (追踪误差, x=1.0)")
print("="*76)
print(f"{'v_M':>7}{'B_M':>7}{'sigma':>7} |{'预测':>9}{'模拟':>9}")
for v,B,s in [(0.0,0.1,0.1),(0.1,0.1,0.1),(0.3,0.1,0.1),(0.5,0.1,0.1),(0.3,0.0,0.0)]:
    print(f"{v:7.3f}{B:7.3f}{s:7.3f} |{theory(1.0,B,s,v):9.4f}{sim(1.0,B,s,v_M=v,T=30000):9.4f}")

print()
print("="*76)
print("表4  多维确认 (d=10 各向同性, 理论用 ||.|| 的 rms 形式)")
print("="*76)
import numpy as np
def sim_d(x,B_M,sigma,T=20000,d=10,seed=1):
    rng=np.random.default_rng(seed); e=np.full(d,2.0); acc=0.0;n=0
    for t in range(T):
        nrm=np.linalg.norm(e); dl=B_M*(e/nrm) if nrm>1e-12 else np.zeros(d)
        e=(1-x)*e+dl+sigma*rng.normal(size=d)/math.sqrt(d)
        if t>=T//2: acc+=np.linalg.norm(e); n+=1
    return acc/n
for x,B,s in [(1.0,0.1,0.1),(1.0,0.3,0.05),(1.5,0.1,0.1)]:
    print(f"  x={x:.1f} B_M={B:.2f} sigma={s:.2f} -> 理论 {theory(x,B,s):.4f}  模拟(d=10) {sim_d(x,B,s):.4f}")
