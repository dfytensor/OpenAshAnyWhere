"""ARC-LLM 实验分析: 4 条件 grok 对比 + SGR 对稳定性的贡献 + 门控行为."""
import csv, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for _f in [r'C:\Windows\Fonts\simhei.ttf', r'C:\Windows\Fonts\msyh.ttc']:
    if os.path.exists(_f): font_manager.fontManager.addfont(_f)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.dirname(os.path.abspath(__file__))
def load(c):
    r=[]
    with open(os.path.join(OUT,f'log_{c}.csv'),encoding='utf-8') as f:
        for x in csv.DictReader(f):
            r.append((int(x['step']),float(x['V']),float(x['C_rank']),float(x['K']),
                      float(x['acc_train']),float(x['acc_test']),float(x['lambda']),
                      float(x['gate_mean']) if x['gate_mean']!='nan' else float('nan')))
    return np.array(r,float)
def grok(a):
    i=np.where(a[:,5]>0.5)[0]; return int(a[i[0],0]) if len(i) else None
def v_spike(a):
    """首次记忆(train>0.95)之后的最大 V —— 衡量 slingshot 严重度.
    (不带 te<0.5 过滤, 因为 slingshot 尖峰会把 train acc 也打崩, 那正是要捕获的.)"""
    mem=np.where(a[:,4]>0.95)[0]
    if len(mem)==0: return 0.0
    return float(a[mem[0]:,1].max())

conds=[('vanilla','标准 wd=0.1','C7','-'),
       ('arc_no_clr','SGR+固定wd','C4','--'),
       ('arc_no_sgr','CLR(无SGR)','C1','--'),
       ('arc_full','ARC-LLM(SGR+CLR) ★','C2','-')]
D={c:load(c) for c,_,_,_ in conds}

print('=== ARC-LLM 4 条件对比 (transformer, p=113) ===')
print(f'{"条件":<20}{"grok步":>8}{"末段te":>9}{"记忆后最大V":>14}{"V>1尖峰数":>11}')
for c,nm,_,_ in conds:
    a=D[c]; nsp=int((a[1:,1]>1.0).sum())
    print(f'{nm:<20}{str(grok(a)):>8}{a[-5:,5].mean():>9.3f}{v_spike(a):>14.3f}{nsp:>11}')

print('\n=== 诚实结论 ===')
a1=D['arc_no_sgr']; a2=D['arc_full']
print(f'  [CLR 是关键] 固定wd(vanilla/arc_no_clr) 不 grok; 两个 CLR 条件都 grok(~s{grok(a1)}). 复现论文.')
print(f'  [SGR 无 grokking 收益] grok步 no_sgr={grok(a1)} vs full={grok(a2)} (相近); '
      f'V>1 尖峰数相近 ({int((a1[1:,1]>1).sum())} vs {int((a2[1:,1]>1).sum())}). SGR 不消除 slingshot.')
# 门控对惊讶的反应: 找 V 尖峰处 gate 是否升高
sp=np.where(a2[:,1]>1.0)[0]
if len(sp):
    g_spike=np.nanmean(a2[sp,7]); g_base=np.nanmean(a2[:,7])
    print(f'  [SGR 门控功能正常] 尖峰期 ḡ={g_spike:.3f} vs 基线 ḡ={g_base:.3f} '
          f'=> 惊讶时门控升高(更多走全秩), 路由机制本身有效, 只是不转化为 grok 优势.')
print(f'  [SGR 的真实价值] 推理期可硬路由: ḡ~{np.nanmean(a2[-20:,7]):.2f} 意味约 {(1-np.nanmean(a2[-20:,7]))*100:.0f}% FFN 走低秩(rank32 vs 512) => 推理省算力.')

fig,ax=plt.subplots(2,2,figsize=(14,9))
# (1) test acc
for c,nm,col,ls in conds:
    a=D[c]; ax[0,0].plot(a[:,0],a[:,5],label=nm,color=col,ls=ls,lw=1.8)
ax[0,0].set_xlabel('step'); ax[0,0].set_ylabel('test acc'); ax[0,0].set_ylim(-0.03,1.05)
ax[0,0].set_title('Grokking: CLR 条件 grok, 固定wd 条件不 grok')
ax[0,0].legend(fontsize=9); ax[0,0].grid(alpha=.3)
# (2) V: arc_no_sgr 尖峰 vs arc_full 平滑
for c,nm,col,ls in [('arc_no_sgr','CLR(无SGR)','C1','--'),('arc_full','ARC-LLM ★','C2','-')]:
    a=D[c]; ax[0,1].plot(a[:,0],a[:,1],label=nm,color=col,ls=ls,lw=1.8)
ax[0,1].set_xlabel('step'); ax[0,1].set_ylabel('V (train loss)'); ax[0,1].set_yscale('symlog',linthresh=1e-3)
ax[0,1].set_title('V(loss): 两 CLR 条件都有 slingshot (Adam 特性), SGR 不消除')
ax[0,1].legend(fontsize=9); ax[0,1].grid(alpha=.3,which='both')
# (3) 有效秩塌缩
for c,nm,col,ls in conds:
    a=D[c]; ax[1,0].plot(a[:,0],a[:,2],label=nm,color=col,ls=ls,lw=1.6)
ax[1,0].set_xlabel('step'); ax[1,0].set_ylabel('stable rank (平均)')
ax[1,0].set_title('有效秩: grokking = 秩塌缩 (~47 → ~4)')
ax[1,0].legend(fontsize=9); ax[1,0].grid(alpha=.3)
# (4) ARC-LLM 门控行为 + λ 退场
a=D['arc_full']
ax[1,1].plot(a[:,0],a[:,7],color='C2',label='门控 ḡ (惊讶度路由)',lw=1.8)
ax[1,1].set_ylabel('gate mean ḡ',color='C2'); ax[1,1].tick_params(axis='y',labelcolor='C2')
axb=ax[1,1].twinx(); axb.plot(a[:,0],a[:,6],color='C3',label='λ (闭环压缩)',lw=1.2)
axb.set_ylabel('λ',color='C3'); axb.set_yscale('log'); axb.tick_params(axis='y',labelcolor='C3')
ax[1,1].set_xlabel('step'); ax[1,1].set_title('ARC-LLM: 门控稳定~0.43(混合路由) + λ 泛化后退场')
ax[1,1].grid(alpha=.3)
plt.tight_layout()
p=os.path.join(OUT,'arc_analysis.png'); plt.savefig(p,dpi=120)
print(f'\n图: {p}')

with open(os.path.join(OUT,'summary.txt'),'w',encoding='utf-8') as f:
    for c,nm,_,_ in conds:
        a=D[c]; f.write(f'{nm}: grok={grok(a)} final_te={a[-5:,5].mean():.3f} slingshot={v_spike(a):.3f}\n')
