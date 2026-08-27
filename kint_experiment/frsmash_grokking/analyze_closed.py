"""K-Net 闭环改进 最终分析: 5 种 governor 在难种子 seed=1 上的对比 + closed 在 3 种子的鲁棒性."""
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
            sr = float(x['srank']) if 'srank' in x and x['srank'] else float('nan')
            r.append((int(x['step']),float(x['V']),float(x['C']),float(x['K_est']),
                      float(x['acc_train']),float(x['acc_test']),sr))
    return np.array(r,float)
def grok(a):
    i=np.where(a[:,5]>0.5)[0]; return int(a[i[0],0]) if len(i) else None
def stable(a):   # grok 后是否保持 (末段 te>0.9)
    i=np.where(a[:,5]>0.5)[0]
    if not len(i): return False
    return a[-5:,5].mean() > 0.9

print('=== seed=1 (难种子) governor 对比 ===')
govs1 = [('std_wd0','标准 wd=0','C7','-'),
         ('knet_g5_wd0','原版K-Net(floor)','C3','--'),
         ('knet2_s1','v1 compress','C1','--'),
         ('knet2_ramp_s1','v2 ramp','C4','--'),
         ('knet2_sine_s1','v3 sine','C5','--'),
         ('knet2_closed_s1','v4 closed ★','C2','-')]
for c,nm,_,_ in govs1:
    if os.path.exists(os.path.join(OUT,f'log_{c}.csv')):
        a=load(c); g=grok(a); st=stable(a)
        print(f'  {nm:<20} grok={str(g):>6} 末段te={a[-5:,5].mean():.3f} 稳定={st}')

print('\n=== closed-loop 3 种子 ===')
for c in ['knet2_closed_s0','knet2_closed_s1','knet2_closed_s2']:
    a=load(c); print(f'  {c}: grok={grok(a)} 末段te={a[-5:,5].mean():.3f} 稳定={stable(a)}')

fig,ax=plt.subplots(1,2,figsize=(15,5.5))
# (1) seed=1 所有 governor 的 test acc
for c,nm,col,ls in govs1:
    if os.path.exists(os.path.join(OUT,f'log_{c}.csv')):
        a=load(c); ax[0].plot(a[:,0],a[:,5],label=nm,color=col,ls=ls,lw=1.8)
ax[0].set_xlabel('step'); ax[0].set_ylabel('test acc'); ax[0].set_ylim(-0.03,1.05)
ax[0].set_title('难种子 seed=1: 只有闭环 governor (★) 稳定 grok')
ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)

# (2) closed 3 种子 + λ 退场
for c,nm,col in [('knet2_closed_s0','seed0','C0'),('knet2_closed_s1','seed1','C3'),('knet2_closed_s2','seed2','C2')]:
    a=load(c)
    ax[1].plot(a[:,0],a[:,5],label=f'{nm} test acc',color=col,lw=1.8)
axb=ax[1].twinx()
for c,col in [('knet2_closed_s1','C3')]:
    a=load(c); axb.plot(a[:,0],a[:,6],color='gray',lw=1,alpha=.6,label='λ (seed1)')
axb.set_ylabel('λ (压缩强度)',color='gray'); axb.tick_params(axis='y',labelcolor='gray')
ax[1].set_xlabel('step'); ax[1].set_ylabel('test acc'); ax[1].set_ylim(-0.03,1.05)
ax[1].set_title('闭环 governor: 3/3 种子稳定 grok @ wd=0; λ 泛化后自动退场')
ax[1].legend(fontsize=9,loc='center right'); ax[1].grid(alpha=.3)
plt.tight_layout()
p=os.path.join(OUT,'knet_closed_loop.png'); plt.savefig(p,dpi=120)
print(f'\n图: {p}')
