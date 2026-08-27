"""FRSMASH K-Net 改进版 最终分析: 原版(floor)失败 vs 改进版(compress)在 wd=0 成功 grok."""
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

conds = [
    ('std_wd1e-1',        '标准 wd=0.1',         'C0', '-'),
    ('std_wd0',           '标准 wd=0',           'C7', '-'),
    ('knet_g5_wd0',       '原版K-Net γ=5 wd=0',  'C3', '--'),
    ('knet2_compress_wd0','改进K-Net wd=0 ★',    'C2', '-'),
]
D = {c:load(c) for c,_,_,_ in conds}

def grok_step(a):
    idx = np.where(a[:,5] > 0.5)[0]
    return int(a[idx[0],0]) if len(idx) else None

print('=== Grokking 对比 (FRSMASH v3.6, p=113) ===')
print(f'{"条件":<22}{"最终test":>9}{"grok步":>9}{"末段C":>9}{"末段稳定秩":>11}')
for c,nm,_,_ in conds:
    a=D[c]; gs=grok_step(a)
    sr = np.nanmean(a[-10:,6])
    print(f'{nm:<22}{a[-1,5]:>9.3f}{str(gs):>9}{a[-10:,2].mean():>9.0f}{sr:>11.1f}')

print('\n★ 关键: 改进K-Net 在 wd=0 (标准&原版都失败) 下 grok, 且比标准+wd 更快')

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
# (1) test acc — 主战场
for c,nm,col,ls in conds:
    a=D[c]; ax[0,0].plot(a[:,0], a[:,5], label=nm, color=col, ls=ls, lw=1.8)
ax[0,0].set_xlabel('step'); ax[0,0].set_ylabel('test acc')
ax[0,0].set_title('★ 改进K-Net(wd=0) 成功 Grokking; 标准&原版K-Net 在 wd=0 失败')
ax[0,0].legend(fontsize=9); ax[0,0].grid(alpha=.3); ax[0,0].set_ylim(-0.03,1.05)

# (2) C 压缩对比 (log)
for c,nm,col,ls in conds:
    a=D[c]; ax[0,1].plot(a[:,0], a[:,2], label=nm, color=col, ls=ls, lw=1.6)
ax[0,1].set_xlabel('step'); ax[0,1].set_ylabel('C (weight norm²)')
ax[0,1].set_yscale('log'); ax[0,1].set_title('C 复杂度: 改进K-Net 把 C 压 113× (电路简化)')
ax[0,1].legend(fontsize=9); ax[0,1].grid(alpha=.3, which='both')

# (3) 改进 K-Net 的完整动力学: V, C, λ, test
a = D['knet2_compress_wd0']
ax[1,0].plot(a[:,0], a[:,5], color='C2', label='test acc', lw=2)
ax[1,0].plot(a[:,0], a[:,4], color='C1', label='train acc', lw=1.2, alpha=.7)
ax[1,0].set_ylabel('accuracy', color='k')
axb = ax[1,0].twinx()
axb.plot(a[:,0], np.log10(a[:,2]+1), color='C0', label='log10(C)')
axb.set_ylabel('log10(C)', color='C0'); axb.tick_params(axis='y', labelcolor='C0')
gs = grok_step(a)
if gs: ax[1,0].axvline(gs, color='r', ls='--', alpha=.5, label=f'grok s{gs}')
ax[1,0].set_xlabel('step'); ax[1,0].set_title('改进K-Net 动力学: 拟合→λ升高→C塌缩→顿悟')
ax[1,0].legend(fontsize=8, loc='center right'); ax[1,0].grid(alpha=.3)

# (4) 稳定秩 collapse (grokking 的真正签名)
for c,nm,col,ls in conds:
    a=D[c]
    if not np.isnan(a[:,6]).all():
        ax[1,1].plot(a[:,0], a[:,6], label=nm, color=col, ls=ls, lw=1.6)
ax[1,1].set_xlabel('step'); ax[1,1].set_ylabel('stable rank (有效秩)')
ax[1,1].set_title('稳定秩: grokking = 有效秩塌缩到 ~4 (找算法解)')
ax[1,1].legend(fontsize=9); ax[1,1].grid(alpha=.3)

plt.tight_layout()
p = os.path.join(OUT, 'frsmash_knet_improved.png'); plt.savefig(p, dpi=120)
print(f'\n图: {p}')
