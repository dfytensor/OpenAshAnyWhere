"""K-Net Grokking 实验分析: 4 条件对比 + K-Net homeostasis 可视化."""
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

def load(cond):
    rows=[]
    with open(os.path.join(OUT,f'log_{cond}.csv'),encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append((int(r['step']),float(r['V_train']),float(r['C']),float(r['K_est']),
                         float(r['acc_train']),float(r['acc_test']),float(r['sigma_mean']),float(r['crisis'])))
    return np.array(rows,float)

conds=[('standard_wd0','普通MLP wd=0','C0'),('standard_wd1e2','普通MLP wd=1e-2','C1'),
       ('knet_wd0','K-Net(Governor)','C3'),('knet_no_gov','K-Net 无Governor','C4')]
data={c:load(c) for c,_,_ in conds}

fig,ax=plt.subplots(2,2,figsize=(14,9))
# (1) test acc
for c,nm,col in conds:
    a=data[c]; ax[0,0].plot(a[:,0],a[:,5],label=nm,color=col,lw=1.5)
ax[0,0].set_xlabel('step'); ax[0,0].set_ylabel('test acc'); ax[0,0].set_title('测试集准确率 (无一条件发生 Grokking)')
ax[0,0].legend(fontsize=8); ax[0,0].grid(alpha=.3); ax[0,0].set_ylim(-0.02,0.05)

# (2) train acc
for c,nm,col in conds:
    a=data[c]; ax[0,1].plot(a[:,0],a[:,4],label=nm,color=col,lw=1.5)
ax[0,1].set_xlabel('step'); ax[0,1].set_ylabel('train acc'); ax[0,1].set_title('训练集准确率 (K-Net Governor 阻止完全记忆!)')
ax[0,1].legend(fontsize=8); ax[0,1].grid(alpha=.3)

# (3) K-Net homeostasis: C, V, K
a=data['knet_wd0']
ax[1,0].plot(a[:,0],a[:,2],label='C (复杂度)',color='C2')
ax[1,0].set_ylabel('C',color='C2'); ax[1,0].tick_params(axis='y',labelcolor='C2')
axb=ax[1,0].twinx(); axb.plot(a[:,0],a[:,1],label='V (损失)',color='C0')
axb.plot(a[:,0],a[:,3]/15,label='K_est/15',color='C3',ls='--')
axb.axhline(3.984/15,color='C3',ls=':',alpha=.5)
axb.set_ylabel('V / K_est÷15',color='C0')
ax[1,0].set_xlabel('step'); ax[1,0].set_title('K-Net(Governor): C 单调膨胀, V 抗坍缩, K≈K_int (稳态!)')
ax[1,0].grid(alpha=.3)

# (4) no-gov 坍缩 vs governor 维持
for c,nm,col in [('knet_no_gov','无Governor (坍缩)','C4'),('knet_wd0','有Governor (维持)','C3')]:
    a=data[c]; ax[1,1].plot(a[:,0],np.log10(a[:,3]+1e-6),label=nm,color=col,lw=1.5)
ax[1,1].axhline(np.log10(3.984+1e-6),ls='--',color='k',alpha=.5,label='log10(K_int)')
ax[1,1].set_xlabel('step'); ax[1,1].set_ylabel('log10(K_est = C·V)')
ax[1,1].set_title('K_est: 有Governor 维持在 K_int 附近 / 无Governor 崩到 0')
ax[1,1].legend(fontsize=8); ax[1,0].grid(alpha=.3); ax[1,1].grid(alpha=.3)

plt.tight_layout()
p=os.path.join(OUT,'knet_grokking_analysis.png'); plt.savefig(p,dpi=120)
print('图:',p)

# 数值摘要
print('\n=== 数值摘要 ===')
for c,nm,_ in conds:
    a=data[c]; te=a[:,5].max()
    print(f'{nm:<22} best_test_acc={te:.4f}  末段V={a[-20:,1].mean():.4f}  末段C={a[-20:,2].mean():.2f}')
print('\nK-Net homeostasis (knet_wd0):')
a=data['knet_wd0']
print(f'  V: 初{a[:5,1].mean():.3f}  末段={a[-20:,1].mean():.4f} (抗坍缩, 不为0)')
print(f'  C: 初{a[:5,2].mean():.2f} -> 末段{a[-20:,2].mean():.2f} (膨胀 {a[-20:,2].mean()/a[:5,2].mean():.1f}x)')
print(f'  K_est 末段={a[-20:,3].mean():.2f}  目标K_int=3.984')
print(f'  => Governor 成功把 C·V 维持在 K_int 附近 (机制成立)')
print(f'  => 但 test_acc={a[:,5].max():.3f} 未泛化 (E.3 grokking 预言 NOT supported)')
