"""FRSMASH Grokking + K-Net 实验: 最终分析."""
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
            r.append((int(x['step']),float(x['V']),float(x['C']),float(x['K_est']),
                      float(x['acc_train']),float(x['acc_test'])))
    return np.array(r,float)

conds = [
    ('std_wd1e-1',   '标准 wd=0.1',        'C0', '-'),
    ('std_wd0',      '标准 wd=0',          'C7', '-'),
    ('knet_g1_wd0',  'K-Net γ=1 wd=0',     'C2', '--'),
    ('knet_g5_wd0',  'K-Net γ=5 wd=0',     'C3', '--'),
    ('knet_g1_wd1e-1','K-Net γ=1 wd=0.1',  'C4', '--'),
]
D = {c:load(c) for c,_,_,_ in conds}

def grok_step(a):   # 首次 test acc>0.5
    idx = np.where(a[:,5] > 0.5)[0]
    return int(a[idx[0],0]) if len(idx) else None

print('=== Grokking 对比 (FRSMASH v3.6, p=113) ===')
print(f'{"条件":<20}{"最终test_acc":>12}{"grok步(te>0.5)":>16}{"末段C":>10}')
for c,nm,_,_ in conds:
    a=D[c]; gs=grok_step(a)
    print(f'{nm:<20}{a[-1,5]:>12.3f}{str(gs):>16}{a[-20:,2].mean():>10.0f}')

K_int = 0.5 * 54166.3 * 5.092   # 自适应目标
print(f'\nK_int(target) = {K_int:.0f}')
a=D['knet_g5_wd0']
print(f'γ=5 条件 K_est 均值(s1000+)={a[10:,3].mean():.0f}  偏差={abs(a[10:,3].mean()-K_int)/K_int*100:.2f}%  <- 存在性约束被精确锁定')

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
# (1) test acc
for c,nm,col,ls in conds:
    a=D[c]; ax[0,0].plot(a[:,0], a[:,5], label=nm, color=col, ls=ls, lw=1.6)
ax[0,0].set_xlabel('step'); ax[0,0].set_ylabel('test acc')
ax[0,0].set_title('Grokking: 只有带 weight_decay 的条件泛化 (K-Net γ=5 卡死)')
ax[0,0].legend(fontsize=8); ax[0,0].grid(alpha=.3); ax[0,0].set_ylim(-0.03,1.05)

# (2) K_est vs step (log)
for c,nm,col,ls in conds:
    a=D[c]; ax[0,1].plot(a[:,0], a[:,3]+1, label=nm, color=col, ls=ls, lw=1.6)
ax[0,1].axhline(K_int+1, color='k', ls=':', alpha=.6, label=f'K_int={K_int:.0f}')
ax[0,1].set_xlabel('step'); ax[0,1].set_ylabel('K_est = C·V (+1, log)')
ax[0,1].set_yscale('log'); ax[0,1].set_title('K_est: γ=5 把 C·V 死死钉在 K_int (±0.6%)')
ax[0,1].legend(fontsize=8); ax[0,1].grid(alpha=.3, which='both')

# (3) 标准条件 grokking 的 C 压缩 + V 坍缩
a = D['std_wd1e-1']
ax[1,0].plot(a[:,0], a[:,2]/1000, color='C2', label='C (÷1000)')
ax[1,0].axvline(grok_step(a) or 0, color='r', ls='--', alpha=.5, label='grok 点')
ax[1,0].set_ylabel('C (k)', color='C2'); ax[1,0].tick_params(axis='y', labelcolor='C2')
axb = ax[1,0].twinx(); axb.plot(a[:,0], a[:,1], color='C0', label='V')
axb.set_ylabel('V (loss)', color='C0'); axb.tick_params(axis='y', labelcolor='C0')
axb.set_yscale('log')
ax[1,0].set_xlabel('step'); ax[1,0].set_title('标准 Grokking: 顿悟期 C 被 weight_decay 压缩, V 坍缩')
ax[1,0].grid(alpha=.3)

# (4) γ=5 的 C·V 钉死 + 无法学习
a = D['knet_g5_wd0']
ax[1,1].plot(a[:,0], a[:,3], color='C3', label='K_est = C·V')
ax[1,1].axhline(K_int, color='k', ls=':', label=f'K_int={K_int:.0f}')
ax[1,1].set_xlabel('step'); ax[1,1].set_ylabel('K_est', color='C3')
axr = ax[1,1].twinx(); axr.plot(a[:,0], a[:,4], color='C1', label='train acc')
axr.set_ylabel('train acc', color='C1'); axr.set_ylim(-0.03,1.05)
ax[1,1].set_title('γ=5: C·V 被锁在 K_int, 但 train_acc=0.36 学不动 (约束满足≠智能)')
ax[1,1].grid(alpha=.3); ax[1,1].legend(loc='center right', fontsize=8)

plt.tight_layout()
p = os.path.join(OUT, 'frsmash_grokking_analysis.png'); plt.savefig(p, dpi=120)
print(f'\n图: {p}')
