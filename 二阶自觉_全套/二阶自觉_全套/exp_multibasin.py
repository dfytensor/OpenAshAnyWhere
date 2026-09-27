import numpy as np, math

# ---------- 多 basin 景观: V(s) = -sum A_i exp(-(s-c_i)^2/(2w^2)) ; ICO = -V ----------
def make_pot(c, A, w):
    c=np.asarray(c,float); A=np.asarray(A,float)
    def V(s):
        s=np.atleast_1d(s); d=(s[...,None]-c)/w
        return -(A*np.exp(-0.5*d*d)).sum(-1)
    def gradV(s):
        s=np.atleast_1d(s); d=s[...,None]-c
        return (A*(d/w**2)*np.exp(-0.5*(d/w)**2)).sum(-1)
    return V, gradV

c=np.array([-1.5,1.5]); A=np.array([0.25,0.25]); w=0.5
V,gradV=make_pot(c,A,w)
mu=A[0]/w**2                      # 井底曲率 = 1.0
ss=np.linspace(-1.5,1.5,30001)
maxg=np.max(np.abs(gradV(ss)))     # 势垒区最大恢复力
dV=V(0.0)[0]-V(1.5)[0]            # 势垒高度
print("="*74); print("景观参数"); print("="*74)
print(f"  井位 c={c}  井深 A={A}  宽 w={w}")
print(f"  井底曲率 mu={mu:.4f}   势垒高 dV={dV:.4f}   max|gradV|={maxg:.4f}")

# ---------- 向量化 Langevin ----------
def run(x,B_M,sigma,T,M,mode='instant',meta_period=25,seed=0,start=0):
    """s_{t+1}=s_t - eta*gradV + Delta + xi ; eta=x/mu ; 返回 (逃逸率, 逃逸时间均值, 局部误差, 峰度, 双峰系数)"""
    eta=x/mu
    rng=np.random.default_rng(seed)
    s=np.full(M,c[start])+1e-3*rng.normal(size=M)
    u=rng.choice([-1.0,1.0],size=M)
    esc=np.full(M,-1); offs=[]
    for t in range(T):
        g=gradV(s)
        near=c[np.argmin(np.abs(s[:,None]-c[None,:]),axis=1)]
        e=s-near
        if   mode=='instant':  d=B_M*np.sign(e)
        elif mode=='sustained':
            if t%meta_period==0: u=rng.choice([-1.0,1.0],size=M)
            d=B_M*u
        else:                  d=np.zeros(M)
        s=s-eta*g+d+sigma*rng.normal(size=M)
        live=esc<0
        escaped=live&(np.abs(s-c[start])>1.5)          # 越过势垒进入另一井域
        esc[escaped]=t
        offs.append(e[live])
    off=np.concatenate([o for o in offs if len(o)])     # 逃逸前的局部偏移
    nesc=int((esc>=0).sum()); rate=nesc/(M*T)
    mt=esc[esc>=0].mean() if nesc>0 else float('nan')
    k =np.mean(off**4)/np.mean(off**2)**2 if len(off)>10 else float('nan')
    sk=np.mean(off**3)/np.mean(off**2)**1.5 if len(off)>10 else float('nan')
    bc=(sk**2+1)/k if k==k else float('nan')
    return dict(rate=rate,mt=mt,loc=float(np.mean(np.abs(off))),kurt=float(k),bc=float(bc),pesc=nesc/M)

print(); print("="*74)
print("E1a  逃逸率 vs B_M   (x=1.0, sigma=0.15, 理论阈值 B_M^c = eta*max|gradV|)")
print("="*74)
eta_th=1.0/mu; Bc=eta_th*maxg
print(f"  理论硬阈值 B_M^c = {Bc:.4f}")
print(f"{'B_M':>8}{'B_M/Bc':>8} |{'instant率':>12}{'sustained率':>13} |{'instant tau':>12}{'sust tau':>10}")
for B in [0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.45,0.60]:
    a=run(1.0,B,0.15,8000,1200,'instant',seed=1)
    b=run(1.0,B,0.15,8000,1200,'sustained',seed=1)
    print(f"{B:8.3f}{B/Bc:8.2f} |{a['rate']:12.3e}{b['rate']:13.3e} |{a['mt']:12.1f}{b['mt']:10.1f}")

print(); print("="*74)
print("E1b  逃逸率 vs sigma  (B_M=0, 纯噪声 -> Kramers/Arrhenius: log kappa ~ -2dV/sigma^2)")
print("="*74)
print(f"{'sigma':>8}{'2dV/s^2':>9} |{'逃逸率':>12}{'ln(rate)':>10} | 相邻斜率")
prev=None
for sg in [0.30,0.35,0.40,0.50,0.60,0.75,0.90]:
    r=run(1.0,0.0,sg,8000,1500,'none',seed=2)
    z=2*dV/sg**2
    sl='' if prev is None else f"{(math.log(max(r['rate'],1e-12))-math.log(max(prev[1],1e-12)))/(z-prev[0]):+8.2f}"
    print(f"{sg:8.3f}{z:9.3f} |{r['rate']:12.3e}{math.log(max(r['rate'],1e-12)):10.2f} | {sl}")
    prev=(z,r['rate'])

print(); print("="*74)
print("E2   r*(B_M) 曲线形状 : 线性(强凹) vs 阶跃(basin hopping)")
print("="*74)
print(f"{'B_M':>8}{'B_M/Bc':>8} |{'局部误差':>10}{'理论B_M/x':>11} |{'逃逸比例':>10}")
for B in [0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50]:
    a=run(1.0,B,0.15,8000,1200,'instant',seed=3)
    print(f"{B:8.3f}{B/Bc:8.2f} |{a['loc']:10.4f}{B/1.0:11.4f} |{a['pesc']:10.3f}")

print(); print("="*74)
print("E3   basin 内分布形状 : sigma 单峰(kurt~3) vs B_M 边缘堆积(双峰 kurt<3, BC>0.555)")
print("="*74)
print(f"{'B_M':>8}{'sigma':>8}{'B/s':>7} |{'kurt':>8}{'BC':>8} | 判读")
for B,sg in [(0.0,0.30),(0.0,0.50),(0.0,0.70),(0.10,0.05),(0.20,0.05),(0.28,0.05),(0.20,0.15),(0.10,0.40)]:
    a=run(1.0,B,sg,8000,1500,'instant',seed=4)
    tag='噪声单峰' if a['kurt']>2.6 else ('边缘双峰' if a['kurt']<2.3 else '过渡')
    print(f"{B:8.3f}{sg:8.3f}{B/max(sg,1e-9):7.2f} |{a['kurt']:8.3f}{a['bc']:8.3f} | {tag}")
