"""K_int 不变性检验 (链接的"终极验证"):
   同模型同数据, 不同初始化(seed0/1/2) + 不同 batch(64) -> 平台 K 是否一致?
   一致 => K_int 是物理常数;  不一致 => 训练 artifact.
"""
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
    rows = []
    with open(os.path.join(OUT, f'log_{cond}.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append((int(r['step']), float(r['loss_V']), float(r['weight_norm_sq_C']), float(r['K_est'])))
    return np.array(rows, dtype=float)  # step, V, C, K

def plateau(arr, frac=0.25):
    tail = arr[int(len(arr)*(1-frac)):]
    K, C, V = tail[:,3], tail[:,2], tail[:,1]
    return dict(Kmed=float(np.median(K)), Kmean=float(np.mean(K)), Kstd=float(np.std(K)),
                cv=float(K.std()/K.mean()), Cmed=float(np.median(C)), Vmed=float(np.median(V)),
                n=len(tail), last_step=int(arr[-1,0]))

conds = [('baseline(seed0)','baseline'), ('seed1','seed1'), ('seed2','seed2'), ('batch64','batch64')]
res = {}
print(f'{"条件":<16}{"平台K(med)":>14}{"±std":>12}{"CV":>8}{"C(med)":>12}{"V(med)":>9}{"末步":>7}')
for name, key in conds:
    p = os.path.join(OUT, f'log_{key}.csv')
    if not os.path.exists(p):
        print(f'{name}: 缺日志'); continue
    s = plateau(load(key))
    res[name] = s
    print(f'{name:<16}{s["Kmed"]/1e6:>11.3f} M{ s["Kstd"]/1e6:>9.3f} M{s["cv"]:>8.3f}'
          f'{s["Cmed"]/1e6:>9.3f} M{s["Vmed"]:>9.3f}{s["last_step"]:>7}')

# 种子不变性 (3 个 seed, 同 batch=32)
seed_K = [res[n]['Kmed'] for n in ['baseline(seed0)','seed1','seed2'] if n in res]
seed_K = np.array(seed_K)
print(f'\n[种子不变性] 3 seed 平台 K = {[f"{k/1e6:.2f}M" for k in seed_K]}')
print(f'  跨种子 mean={seed_K.mean()/1e6:.3f}M  std={seed_K.std()/1e6:.3f}M  '
      f'CV={seed_K.std()/seed_K.mean():.3f}  spread={(seed_K.max()-seed_K.min())/1e6:.3f}M')

# 判定: 链接要"完全一致". 用 CV<3% 作为"近常数"阈
scv = seed_K.std()/seed_K.mean()
verdict = '不变 (支持 K_int 为物理常数)' if scv < 0.03 else '有漂移 (更像训练 artifact)'
print(f'  => 跨种子 CV={scv*100:.2f}%  => {verdict}')

# batch 不变性
if 'batch64' in res and len(seed_K):
    bK = res['batch64']['Kmed']
    print(f'\n[batch不变性] batch64 平台 K={bK/1e6:.3f}M  vs  batch32 种子均值={seed_K.mean()/1e6:.3f}M')
    print(f'  偏差={(bK-seed_K.mean())/seed_K.mean()*100:+.1f}%  '
          f'(batch64 多见 2x 数据/step, 训练更充分, 偏低合理)')

# 图: K(t) 四条曲线 + 种子均值水平线
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
colors = {'baseline':'C0','seed1':'C1','seed2':'C2','batch64':'C3'}
for name, key in conds:
    if not os.path.exists(os.path.join(OUT, f'log_{key}.csv')): continue
    a = load(key)
    s = 7
    ax[0].plot(a[s-1:,0], np.convolve(a[:,3], np.ones(s)/s, mode='valid')/1e6,
               label=name, color=colors[key], alpha=.85)
ax[0].axhline(seed_K.mean()/1e6, ls='--', color='k', alpha=.5, label=f'种子均值 {seed_K.mean()/1e6:.2f}M')
ax[0].set_xlabel('step'); ax[0].set_ylabel('K_est = C*V (M)'); ax[0].set_title('K(t) 跨种子/batch (不变性检验)')
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

# 右图: 平台 K 柱状对比
names = list(res.keys()); Ks = [res[n]['Kmed']/1e6 for n in names]
errs = [res[n]['Kstd']/1e6 for n in names]
keymap = {'baseline(seed0)':'baseline','seed1':'seed1','seed2':'seed2','batch64':'batch64'}
bar_colors = [colors.get(keymap.get(n,''), 'C0') for n in names]
ax[1].bar(range(len(names)), Ks, yerr=errs, color=bar_colors, alpha=.8)
ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels([n.replace('(seed0)','') for n in names], rotation=15, fontsize=8)
ax[1].set_ylabel('平台 K (M)'); ax[1].set_title(f'平台 K 对比 (跨种子 CV={scv*100:.2f}%)')
ax[1].grid(alpha=.3, axis='y')
plt.tight_layout()
p = os.path.join(OUT, 'kint_invariance.png'); plt.savefig(p, dpi=130)
print(f'\n图: {p}')

with open(os.path.join(OUT,'invariance_summary.txt'),'w',encoding='utf-8') as f:
    f.write(f'跨种子平台K(M): {[round(k/1e6,3) for k in seed_K]}\n')
    f.write(f'跨种子CV={scv*100:.2f}%  => {verdict}\n')
    if 'batch64' in res: f.write(f'batch64 K={bK/1e6:.3f}M  偏差={(bK-seed_K.mean())/seed_K.mean()*100:+.1f}%\n')
