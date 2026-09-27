import numpy as np, math

def make_pot(c,A,w):
    c=np.asarray(c,float); A=np.asarray(A,float)
    def gradV(s):
        s=np.atleast_1d(s); d=s[...,None]-c
        return (A*(d/w**2)*np.exp(-0.5*(d/w)**2)).sum(-1)
    return gradV

# ================= E1' 公平对比: 同一 T 下的陡峭度 =================
c=np.array([-1.5,1.5]); A=np.array([0.5,0.5]); w=0.5
gV=make_pot(c,A,w); mu=A[0]/w**2
ss=np.linspace(-1.5,1.5,30001); maxg=float(np.max(np.abs(gV(ss)))); Bc=(1.0/mu)*maxg

def pesc(x,B,sg,T,M,mode,seed=0):
    eta=x/mu; rng=np.random.default_rng(seed)
    s=np.full(M,c[0])+1e-3*rng.normal(size=M); u=rng.choice([-1.,1.],size=M)
    for t in range(T):
        gg=gV(s); near=c[np.argmin(np.abs(s[:,None]-c[None,:]),axis=1)]; e=s-near
        d=(B*np.sign(e)) if mode=='B' else (B*u if mode=='Bs' else 0.0)
        if mode=='Bs' and t%25==0: u=rng.choice([-1.,1.],size=M); d=B*u
        s=s-eta*gg+d+sg*rng.normal(size=M)
        if t==T-1: return float(np.mean(np.abs(s-c[0])>1.5))
    return 0.0

def width(par,ps):
    """p_esc 从 0.1 到 0.9 的相对宽度 (p90-p10)/p50 ; 硬阈值->窄, Arrhenius->宽"""
    par=np.array(par); ps=np.array(ps)
    ok=[(p,y) for p,y in zip(par,ps)]
    lo=next((p for p,y in ok if y>=0.1),None); hi=next((p for p,y in ok if y>=0.9),None)
    if lo is None or hi is None: return float('nan')
    i50=min(range(len(ps)),key=lambda i:abs(ps[i]-0.5)); p50=par[i50]
    return (hi-lo)/p50 if p50>0 else float('nan')

T=400; M=1500
print("="*78); print(f"E1'公平对比  T={T} M={M}  (两者在同一时间窗内可比)"); print("="*78)
bms=[0.15,0.18,0.20,0.22,0.25,0.28,0.30,0.33,0.36,0.42,0.50]
pb=[pesc(1.0,B,0.10,T,M,'B',seed=31) for B in bms]
sgs=[0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.85,1.00,1.20]
ps=[pesc(1.0,0.0,sg,T,M,'n',seed=32) for sg in sgs]
print(f"{'B_M':>7}{'B_M/Bc':>7}{'p':>7}   |{'sigma':>7}{'p':>7}")
for i in range(max(len(bms),len(sgs))):
    L=f"{bms[i]:7.3f}{bms[i]/Bc:7.2f}{pb[i]:7.3f}" if i<len(bms) else " "*21
    R=f"{sgs[i]:7.3f}{ps[i]:7.3f}" if i<len(sgs) else ""
    print(f"{L}   |{R}")
wb,ws=width(bms,pb),width(sgs,ps)
print(f"\n  相对过渡宽度:  B_M 驱动 = {wb:.3f}   |   sigma 驱动 = {ws:.3f}   |  比值 sigma/B_M = {ws/wb:.2f}x")
print("  判读: 比值 >> 1 => B_M 是确定性硬阈值, sigma 是平滑 Arrhenius")

# ================= E4 重做: 固定井间距 =================
print("\n"+"="*78); print("E4修正  固定井间距 gap, 隔离 basin 数 N 的效应"); print("="*78)
gap=1.2; wN=0.42
def runN(N,mode,B=0.32,sg=0.08,T=3000,M=500,x=1.0,seed=41):
    cc=np.array([(i-(N-1)/2)*gap for i in range(N)])
    AA=np.linspace(0.5,1.1,N)                 # ICO 单调递增, 最优井在最右
    gN=make_pot(cc,AA,wN); muN=AA[0]/wN**2; etaN=x/muN
    rng=np.random.default_rng(seed)
    s=np.full(M,cc[0])+1e-3*rng.normal(size=M); u=rng.choice([-1.,1.],size=M)
    for t in range(T):
        gg=gN(s); near=cc[np.argmin(np.abs(s[:,None]-cc[None,:]),axis=1)]; e=s-near
        if mode=='blind': d=B*np.sign(e)                 # 局部推离, 方向由波动随机决定
        elif mode=='sust':
            if t%25==0: u=rng.choice([-1.,1.],size=M)
            d=B*u                                        # 持续但方向随机的提案
        else:           d=B*np.ones(M)                   # 定向: 朝 ICO 更高的方向
        s=s-etaN*gg+d+sg*rng.normal(size=M)
    fin=cc[np.argmin(np.abs(s[:,None]-cc[None,:]),axis=1)]
    return float(np.mean(fin==cc[-1])), float(np.mean(np.abs(s-cc[-1])<gap/2))
print(f"{'N':>4}{'总跨度':>8} |{'盲目 p_top':>11}{'随机提案':>10}{'定向 p_top':>11} |{'定向/盲目':>10}")
for N in [2,3,4,6,8,12]:
    pb_,_=runN(N,'blind'); ps_,_=runN(N,'sust'); pd_,_=runN(N,'direct')
    span=(N-1)*gap
    print(f"{N:4d}{span:8.2f} |{pb_:11.3f}{ps_:10.3f}{pd_:11.3f} |{pd_/max(pb_,1e-9):10.2f}")

# ================= E5 重做: 只统计全程未逃逸的轨迹 =================
print("\n"+"="*78); print("E5修正  K=3 多井, 仅统计全程未逃逸轨迹 (消除幸存者偏差)"); print("="*78)
c3=np.array([-2.0,0.0,2.0]); A3=np.array([0.5,0.5,0.5]); g3=make_pot(c3,A3,0.5); mu3=A3[0]/0.25
def run3(x,B,sg,T=2500,M=700,seed=51):
    eta=x/mu3; rng=np.random.default_rng(seed)
    s=np.full(M,0.0)+1e-3*rng.normal(size=M)
    esc=np.zeros(M,bool); acc=np.zeros(M); cnt=np.zeros(M)
    for t in range(T):
        gg=g3(s); near=c3[np.argmin(np.abs(s[:,None]-c3[None,:]),axis=1)]; e=s-near
        s=s-eta*gg+B*np.sign(e)+sg*rng.normal(size=M)
        esc|=np.abs(s)>1.0
        alive=~esc; acc[alive]+=np.abs(e[alive]); cnt[alive]+=1
    ok=cnt>100
    return float(np.mean(acc[ok]/cnt[ok])) if ok.any() else float('nan'), float((~esc).mean())
def theory(x,B,sg):
    det=(B/x)**2; den=2*x-x*x
    return math.sqrt(det+sg**2/den) if den>0 else float('inf')
xs=[0.10+0.10*i for i in range(19)]
print(f"{'B_M':>6}{'sigma':>7}{'B/s':>6} |{'x*_理论':>9}{'x*_模拟':>9}{'留存率':>8} |{'r*_理论':>9}{'r*_模拟':>9}{'偏差':>8}")
for B,sg in [(0.10,0.03),(0.06,0.10),(0.03,0.20),(0.01,0.35)]:
    th=[theory(x,B,sg) for x in xs]; sm=[run3(x,B,sg)[0] for x in xs]; keep=[run3(x,B,sg)[1] for x in xs]
    it=int(np.nanargmin(sm))
    print(f"{B:6.3f}{sg:7.3f}{B/max(sg,1e-9):6.2f} |{xs[int(np.argmin(th))]:9.2f}{xs[it]:9.2f}"
          f"{keep[it]:8.2f} |{th[it]:9.4f}{sm[it]:9.4f}{100*(th[it]-sm[it])/max(sm[it],1e-9):7.1f}%")
print("\n  细节: x 扫描下的局部误差 (B_M=0.10, sigma=0.03)")
print(f"{'x':>6} |{'理论':>9}{'模拟':>9}{'偏差':>8}{'留存率':>8}")
for x in [0.2,0.5,0.8,1.0,1.2,1.5,1.8]:
    t=theory(x,0.10,0.03); sm,kp=run3(x,0.10,0.03)
    print(f"{x:6.2f} |{t:9.4f}{sm:9.4f}{100*(t-sm)/max(sm,1e-9):7.1f}%{kp:8.2f}")
