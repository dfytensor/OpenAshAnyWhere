import itertools, numpy as np

def sim(N, f, allow_self, use_B, trials=4000, seed=0):
    """界面一致性监控: 观察者 v 对界面(i,j)投票; Byzantine节点撒谎; 多数投票; 归因=susp集合"""
    rng=np.random.default_rng(seed)
    nodes=list(range(N)); ifaces=list(itertools.combinations(nodes,2))
    c_if=0; t_if=0; u_if=0; c_at=0; t_at=0; suspsz=[]
    for _ in range(trials):
        faulty=set(rng.choice(N,size=f,replace=False).tolist())
        truth={ij:(ij[0] in faulty or ij[1] in faulty) for ij in ifaces}
        verdict={}
        for ij in ifaces:
            votes=[]
            for v in nodes:
                if (v in ij) and not allow_self: continue      # 禁止自评(自访问盲区)
                votes.append((not truth[ij]) if v in faulty else truth[ij])
            if use_B: votes.append(truth[ij])                  # 外部基准: 永真
            if not votes: verdict[ij]=None; continue
            s=sum(votes)
            verdict[ij]= True if 2*s>len(votes) else (False if 2*s<len(votes) else None)
        for ij in ifaces:
            t_if+=1
            if verdict[ij] is None: u_if+=1
            elif verdict[ij]==truth[ij]: c_if+=1
        susp=[i for i in nodes if all(verdict.get(ij) is True for ij in ifaces if i in ij)]
        suspsz.append(len(susp)); t_at+=1
        if set(susp)==faulty: c_at+=1
    return dict(iface=c_if/t_if, unk=u_if/t_if, attr=c_at/t_at,
                obs=(N-2)+ (1 if use_B else 0), suspsz=float(np.mean(suspsz)))

print("="*88)
print("E7-A  【禁止自评】制度 (原文公理: 无任何子系统能完整建模自身)  f=1")
print("="*88)
print(f"{'N':>3}{'每界面观察者':>11} |{'界面判准率':>10}{'未知率':>9}{'归因正确率':>11}{'|susp|':>7} | 判读")
for N in [2,3,4,5,6,7]:
    r=sim(N,1,False,False)
    tag={2:'无观察者->全未知',3:'1观察者=单点故障',4:'2观察者->平局',5:'3观察者->可仲裁',6:'4观察者',7:'5观察者'}[N]
    print(f"{N:3d}{r['obs']:11d} |{r['iface']:10.3f}{r['unk']:9.3f}{r['attr']:11.3f}{r['suspsz']:7.2f} | {tag}")

print()
print("="*88)
print("E7-B  【允许自评】制度 (对照: 自访问盲区不存在)  f=1")
print("="*88)
print(f"{'N':>3}{'每界面观察者':>11} |{'界面判准率':>10}{'未知率':>9}{'归因正确率':>11}{'|susp|':>7}")
for N in [2,3,4,5,6,7]:
    r=sim(N,1,True,False)
    print(f"{N:3d}{r['obs']:11d} |{r['iface']:10.3f}{r['unk']:9.3f}{r['attr']:11.3f}{r['suspsz']:7.2f}")

print()
print("="*88)
print("E7-C  监控者自身故障 (κ3 型单点故障) : 固定故障在'最后一个节点', 禁止自评")
print("="*88)
def sim_fixed(N, allow_self, use_B, victim=None, trials=4000, seed=3):
    rng=np.random.default_rng(seed)
    nodes=list(range(N)); ifaces=list(itertools.combinations(nodes,2))
    c=0;t=0;u=0;ca=0;ta=0
    for _ in range(trials):
        faulty={victim} if victim is not None else set()
        truth={ij:(ij[0] in faulty or ij[1] in faulty) for ij in ifaces}
        verdict={}
        for ij in ifaces:
            votes=[]
            for v in nodes:
                if (v in ij) and not allow_self: continue
                votes.append((not truth[ij]) if v in faulty else truth[ij])
            if use_B: votes.append(truth[ij])
            if not votes: verdict[ij]=None; continue
            s=sum(votes); verdict[ij]= True if 2*s>len(votes) else (False if 2*s<len(votes) else None)
        for ij in ifaces:
            t+=1
            if verdict[ij] is None: u+=1
            elif verdict[ij]==truth[ij]: c+=1
        susp=[i for i in nodes if all(verdict.get(ij) is True for ij in ifaces if i in ij)]
        ta+=1; ca+= (set(susp)==faulty)
    return dict(iface=c/t,unk=u/t,attr=ca/ta)
print(f"{'N':>3}{'观察者':>7} |{'界面判准率':>10}{'未知率':>9}{'归因正确率':>11} | 关键: 故障节点是界面(0,1)的唯一中立监控者")
for N in [3,4,5,6,7]:
    r=sim_fixed(N,False,False,victim=N-1)
    print(f"{N:3d}{N-2:7d} |{r['iface']:10.3f}{r['unk']:9.3f}{r['attr']:11.3f}")

print()
print("="*88)
print("E7-D  外部基准 B 能否替代第三个节点?  (禁止自评)")
print("="*88)
print(f"{'配置':>16}{'观察者':>8} |{'界面判准率':>10}{'未知率':>9}{'归因正确率':>11} | 判读")
for lbl,N,uB in [('N=2 无B',2,False),('N=2 +B',2,True),('N=3 无B',3,False),('N=3 +B',3,True),('N=4 +B',4,True),('N=5 无B',5,False)]:
    r=sim(N,1,False,uB)
    tag='界面可判但归因失败(仅1界面)' if (N==2 and uB) else ''
    print(f"{lbl:>16}{r['obs']:8d} |{r['iface']:10.3f}{r['unk']:9.3f}{r['attr']:11.3f} | {tag}")

print()
print("="*88)
print("E7-E  f=2 双故障 (检验鲁棒性门槛是否上移)")
print("="*88)
print(f"{'N':>3}{'观察者':>7} |{'界面判准率':>10}{'未知率':>9}{'归因正确率':>11}")
for N in [4,5,6,7,8]:
    r=sim(N,2,False,False)
    print(f"{N:3d}{N-2:7d} |{r['iface']:10.3f}{r['unk']:9.3f}{r['attr']:11.3f}")

print()
print("="*88)
print("E7-F  监控图拓扑: 三耦合是'环'还是'层级'?")
print("="*88)
def topo(N):
    """边 v->(i,j) 表示 v 监控界面(i,j); 禁止自评 => v not in (i,j)"""
    edges=[(v,tuple(ij)) for ij in itertools.combinations(range(N),2) for v in range(N) if v not in ij]
    # 折叠为节点级监控图: v 监控 i (若 v 参与监控任何含 i 的界面)
    adj={v:set() for v in range(N)}
    for v,ij in edges:
        for i in ij: adj[v].add(i)
    indeg={i:sum(1 for v in range(N) if i in adj[v]) for i in range(N)}
    outdeg={v:len(adj[v]) for v in range(N)}
    # 检测环 (节点级有向图)
    cyc=any((i in adj[v] and v in adj[i]) for v in range(N) for i in range(N))
    return dict(edges=len(edges),indeg=indeg,outdeg=outdeg,has_2cycle=cyc,
                sinks=[i for i in range(N) if indeg[i]==0], sources=[v for v in range(N) if outdeg[v]==0])
for N in [2,3,4,5]:
    t=topo(N)
    print(f"  N={N}: 界面监控边数={t['edges']:3d}  节点入度={list(t['indeg'].values())}  出度={list(t['outdeg'].values())}"
          f"  含2-环={t['has_2cycle']}  不可被监控者={t['sinks']}  不监控他人者={t['sources']}")
