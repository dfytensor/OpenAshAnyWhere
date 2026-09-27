import numpy as np, math

def make_pot(c,A,w):
    c=np.asarray(c,float); A=np.asarray(A,float)
    def V(s):
        s=np.atleast_1d(s); d=(s[...,None]-c)/w
        return -(A*np.exp(-0.5*d*d)).sum(-1)
    def gradV(s):
        s=np.atleast_1d(s); d=s[...,None]-c
        return (A*(d/w**2)*np.exp(-0.5*(d/w)**2)).sum(-1)
    return V,gradV

# ---------- 深井景观 (dV ~ A = 0.5) ----------
c=np.array([-1.5,1.5]); A=np.array([0.5,0.5]); w=0.5
V,gradV=make_pot(c,A,w)
mu=A[0]/w**2; ss=np.linspace(-1.5,1.5,30001)
maxg=float(np.max(np.abs(gradV(ss)))); dV=float(V(0.0)[0]-V(1.5)[0])
print("="*76); print("深井景观:  mu=%.3f  dV=%.4f  max|gradV|=%.4f"%(mu,dV,maxg)); print("="*76)

def run(x,B_M,sigma,T,M,mode='instant',meta_period=25,seed=0,start=0,K=None,target=None):
    cc=c if K is None else K
    eta=x/mu; rng=np.random.default_rng(seed)
    s=np.full(M,cc[start])+1e-3*rng.normal(size=M); u=rng.choice([-1.,1.],size=M)
    esc=np.full(M,-1); offs=[]; fin=None
    for t in range(T):
        g=gradV(s); near=cc[np.argmin(np.abs(s[:,None]-cc[None,:]),axis=1)]; e=s-near
        if   mode=='instant':   d=B_M*np.sign(e)
        elif mode=='sustained':
            if t%meta_period==0: u=rng.choice([-1.,1.],size=M)
            d=B_M*u
        elif mode=='directed':  d=B_M*np.sign((cc[target]-s)) if target is not None else B_M*np.sign(e)
        else:                   d=np.zeros(M)
        s=s-eta*g+d+sigma*rng.normal(size=M)
        live=esc<0; esc[live&(np.abs(s-cc[start])>1.5)]=t
        offs.append(e[live])
    off=np.concatenate([o for o in offs if len(o)])
    fin=cc[np.argmin(np.abs(s[:,None]-cc[None,:]),axis=1)]
    return dict(pesc=float((esc>=0).mean()),mt=float(esc[esc>=0].mean()) if (esc>=0).any() else float('nan'),
                loc=float(np.mean(np.abs(off))),
                kurt=float(np.mean(off**4)/np.mean(off**2)**2),
                ptop=float(np.mean(fin==cc[target])) if target is not None else float('nan'))

# ---------- E1' 修复: 逃逸概率 vs 控制参数, 比较过渡陡峭度 ----------
print("\n"+"="*76)
print("E1'  逃逸概率 p_esc 的过渡陡峭度 : B_M(确定性) vs sigma(随机)")
print("="*76)
eta=1.0/mu; Bc=eta*maxg
print(f"  硬阈值预测 B_M^c = eta*max|gradV| = {Bc:.4f}")
def steepness(params,ps):
    ps=np.array(ps); params=np.array(params)
    d=np.gradient(ps,np.log(params[1]-params[0]+params))  # 粗斜率
    sl=np.diff(ps)/np.diff(params)
    return float(np.max(sl))*float(np.mean(params))       # 归一化最大斜率
print(f"\n[B_M 扫描] sigma=0.10 固定, x=1.0, T=4000, M=800")
bms=[0.10,0.15,0.20,0.25,0.28,0.30,0.32,0.35,0.40,0.50,0.65]
pb=[run(1.0,B,0.10,4000,800,'instant',seed=5)['pesc'] for B in bms]
for B,p in zip(bms,pb): print(f"   B_M={B:5.3f}  B_M/Bc={B/Bc:5.2f}  p_esc={p:6.3f}")
print(f"   -> 归一化最大斜率 = {steepness(bms,pb):.3f}")
print(f"\n[sigma 扫描] B_M=0 固定, x=1.0, T=4000, M=800")
sgs=[0.30,0.40,0.50,0.60,0.70,0.85,1.00,1.20,1.50]
ps=[run(1.0,0.0,sg,4000,800,'none',seed=6)['pesc'] for sg in sgs]
for sg,p in zip(sgs,ps): print(f"   sigma={sg:5.3f}  2dV/s^2={2*dV/sg**2:6.3f}  p_esc={p:6.3f}")
print(f"   -> 归一化最大斜率 = {steepness(sgs,ps):.3f}")
print("\n  判读: B_M 扫描的过渡显著更陡 => 确定性硬阈值; sigma 扫描平滑 => Arrhenius")

# ---------- E1'' instant vs sustained vs directed ----------
print("\n"+"="*76)
print("E1'' 三种元级修正的性质 (x=1.0, T=4000, M=800)")
print("="*76)
print(f"{'B_M':>7}{'sigma':>7} |{'instant p':>11}{'sust p':>9}{'dir p':>8} |{'inst loc':>10}{'sust loc':>9}{'dir loc':>8}")
for B,sg in [(0.20,0.10),(0.30,0.10),(0.20,0.40),(0.30,0.40),(0.25,0.60)]:
    a=run(1.0,B,sg,4000,800,'instant',seed=7); b=run(1.0,B,sg,4000,800,'sustained',seed=7)
    d=run(1.0,B,sg,4000,800,'directed',seed=7,target=1)
    print(f"{B:7.3f}{sg:7.3f} |{a['pesc']:11.3f}{b['pesc']:9.3f}{d['pesc']:8.3f} |{a['loc']:10.4f}{b['loc']:9.4f}{d['loc']:8.4f}")

# ---------- E4 跃迁有效性 vs basin 数 (盲目 vs 定向) ----------
print("\n"+"="*76)
print("E4  跃迁有效性: 最终停在最优 basin 的概率 p_top, N = basin 数")
print("="*76)
print(f"{'N':>4}{'井间距':>7} |{'盲目 p_top':>11}{'定向 p_top':>11} |{'盲目提升':>9}")
for N in [2,3,5,8]:
    cc=np.linspace(-3,3,N); AA=np.linspace(0.5,1.2,N)      # ICO 递增, 最深井在右端
    V2,g2=make_pot(cc,AA,0.45)
    def runN(mode,seed=11,T=4000,M=600,x=1.0,B=0.30,sg=0.10):
        muN=AA[0]/0.45**2; etaN=x/muN; rng=np.random.default_rng(seed)
        s=np.full(M,cc[0])+1e-3*rng.normal(size=M); u=rng.choice([-1.,1.],size=M)
        for t in range(T):
            g=g2(s); near=cc[np.argmin(np.abs(s[:,None]-cc[None,:]),axis=1)]; e=s-near
            if mode=='blind': d=B*(np.sign(e) if True else 0)
            elif mode=='sust':
                if t%25==0: u=rng.choice([-1.,1.],size=M)
                d=B*u
            else: d=B*np.ones(M)          # 定向: 朝 ICO 更高的一端
            s=s-etaN*g+d+sg*rng.normal(size=M)
        fin=cc[np.argmin(np.abs(s[:,None]-cc[None,:]),axis=1)]
        return float(np.mean(fin==cc[-1]))
    pb_=runN('blind'); ps_=runN('sust'); pd_=runN('direct')
    gap=float(np.mean(cc[1]-cc[0]))
    print(f"{N:4d}{gap:7.2f} |{pb_:11.3f}{pd_:11.3f} |{pd_-pb_:9.3f}")

# ---------- E5 多 basin 下 P3/P4 重扫 ----------
print("\n"+"="*76)
print("E5  多 basin(K=3) 下重扫 P3(正交叠加) 与 P4(最优 x*)")
print("="*76)
c3=np.array([-2.0,0.0,2.0]); A3=np.array([0.5,0.5,0.5])
V3,g3=make_pot(c3,A3,0.5); mu3=A3[0]/0.5**2
def run3(x,B,sg,T=5000,M=800,seed=21):
    global c,A,V,gradV,mu
    c_old,A_old=c,A; c,A=c3,A3; V,gradV=V3,g3; mu=mu3
    r=run(x,B,sg,T,M,'instant',seed=seed,start=1)
    c,A=c_old,A_old; V,gradV=make_pot(c,A,0.5); mu=A[0]/0.5**2
    return r
def theory(x,B,sg):
    det=(B/x)**2; den=2*x-x*x
    return math.sqrt(det+(sg**2)/den if den>0 else float('inf'))
xs=[0.10+0.10*i for i in range(19)]
print(f"{'B_M':>6}{'sigma':>7}{'B/s':>6} |{'x*_理论':>9}{'x*_模拟':>9}{'r*(x*)':>9}{'r*(0.5)':>9}{'r*(1.5)':>9}")
for B,sg in [(0.30,0.05),(0.15,0.15),(0.05,0.30),(0.02,0.50)]:
    th=[theory(x,B,sg) for x in xs]; sm=[run3(x,B,sg)['loc'] for x in xs]
    print(f"{B:6.3f}{sg:7.3f}{B/max(sg,1e-9):6.2f} |{xs[int(np.argmin(th))]:9.2f}{xs[int(np.argmin(sm))]:9.2f}"
          f"{min(th):9.4f}{sm[4]:9.4f}{sm[14]:9.4f}")
print("\n  (r*(0.5)=x=0.5处, r*(1.5)=x=1.5处)")
print("\n对比: 局部误差理论 vs 模拟 (K=3, 中央井出发)")
print(f"{'x':>6}{'B_M':>7}{'sigma':>7} |{'理论':>9}{'模拟':>9}{'偏差':>9}")
for x,B,sg in [(0.5,0.15,0.15),(1.0,0.15,0.15),(1.5,0.15,0.15),(1.0,0.30,0.05),(1.0,0.05,0.30)]:
    r=run3(x,B,sg); t=theory(x,B,sg)
    print(f"{x:6.2f}{B:7.3f}{sg:7.3f} |{t:9.4f}{r['loc']:9.4f}{100*(t-r['loc'])/max(r['loc'],1e-9):8.1f}%")
