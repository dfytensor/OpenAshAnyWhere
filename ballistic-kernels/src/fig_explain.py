# -*- coding: utf-8 -*-
"""图解弹道原理: 一张信息图讲完 v3→v6 (全部使用实测数据)"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import core5 as C

HERE = os.path.dirname(os.path.abspath(__file__))
diag = json.load(open(os.path.join(HERE, 'diag5.json'), encoding='utf-8'))
e5 = json.load(open(os.path.join(HERE, 'exp5.json'), encoding='utf-8'))
s6 = json.loads(open(r'F:/OpenASH2605/v6_attbias/summary6.json', 'rb').read().decode('utf-8'))
binit = json.load(open(os.path.join(HERE, 'auto_out', 'auto.json'), encoding='utf-8'))

plt = C.setup_font()
import matplotlib.gridspec as gs
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig = plt.figure(figsize=(17, 20))
G = gs.GridSpec(4, 2, figure=fig, hspace=0.30, wspace=0.18,
                left=0.05, right=0.96, top=0.93, bottom=0.05)
fig.suptitle('图解弹道原理：从「小球抛物线」类比到注意力偏置（v3→v6，全部实测）',
             fontsize=20, fontweight='bold', y=0.97)

# ============ (a) 小球类比 ============
ax = fig.add_subplot(G[0, 0])
toks = ['的', '一', '个', '小', '球', '从', '这', '里']
for i, t in enumerate(toks):
    ax.add_patch(FancyBboxPatch((i - 0.32, -0.75), 0.64, 0.62, boxstyle='round,pad=0.02',
                                fc='#eef3fb', ec='#4a6fa5', lw=1.2))
    ax.text(i, -0.44, t, ha='center', va='center', fontsize=13)
ax.add_patch(FancyBboxPatch((8.7 - 0.32, -0.75), 0.64, 0.62, boxstyle='round,pad=0.02',
                            fc='#fdecea', ec='#c0392b', lw=1.6))
ax.text(8.7, -0.44, '?', ha='center', va='center', fontsize=15, fontweight='bold', color='#c0392b')
ax.text(8.7, -1.15, '预测位 t+1', ha='center', fontsize=10, color='#c0392b')
pal = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
for i in range(2, 8, 1):
    x0, x1 = i, 8.7
    h = 0.55 + 0.75 * np.exp(-(x1 - x0) / 6.0) + 0.12 * np.sin(i)
    u = np.linspace(0, 1, 60)
    ax.plot(x0 + (x1 - x0) * u, h * 4 * u * (1 - u), color=pal[i % 5], lw=1.6, alpha=0.85)
    ax.add_patch(FancyArrowPatch(posA=(x1 - 0.3, 0.06), posB=(8.42, 0.25),
                                 arrowstyle='-|>', mutation_scale=12,
                                 color=pal[i % 5], lw=1.2))
ax.text(4.3, 1.62, '每个上文 token = 一只小球，沿抛物线飞向预测位', fontsize=12.5, ha='center',
        fontweight='bold')
ax.text(4.3, 1.28, '落点 i+R 指向 t+1  →  R = t+1−i ≡ lag k（就是 v1 的「周期」）',
        fontsize=11.5, ha='center', color='#1f5fa8')
ax.text(4.3, 0.94, '统计上等价于完备的 lag-k 共现表 C(k)——这一步不产生新信息',
        fontsize=10.5, ha='center', color='#555')
ax.set_xlim(-0.8, 9.6); ax.set_ylim(-1.4, 2.0); ax.axis('off')
ax.set_title('(a) 类比 → 严格化：落点距离就是 lag', fontsize=14, loc='left')

# ============ (b) 不可辨识 ============
ax = fig.add_subplot(G[0, 1])
u = np.linspace(0, 1, 50)
cases = [(14.1, '#1f77b4', 'm=0.5'), (9.9, '#2ca02c', 'm=2'), (9.25, '#ff7f0e', 'm=5'), (8.95, '#d62728', 'm=20')]
for (v0, c, lab) in cases:
    h = v0 * 0.06
    x = 8 * u
    y = h * 4 * u * (1 - u)
    ax.plot(x, y, color=c, lw=1.8)
    ax.text(4.0, h * 1.02 + 0.05, lab, color=c, fontsize=10, ha='center')
ax.add_patch(FancyBboxPatch((7.6, -0.62), 1.0, 0.55, boxstyle='round,pad=0.02', fc='#fdecea', ec='#c0392b'))
ax.text(8.1, -0.35, 'R=8.0', ha='center', fontsize=11, color='#c0392b', fontweight='bold')
ax.text(4.0, -1.05, '四组完全不同的 (m, v, θ) → 同一个落点 R（θ=45° 固定，只调 m,v）',
        fontsize=11.5, ha='center', fontweight='bold')
ax.text(4.0, -1.42, '3 个自由度 → 1 个观测量：不可辨识（固定 m 后仅调 v 拟合任意落点，误差 3e-13）',
        fontsize=10.5, ha='center', color='#555')
ax.text(4.0, -1.76, '删掉 m,v,θ；保留 R_j、σ_j、φ；「质量」改判为可辨识的权重 β_j',
        fontsize=11, ha='center', color='#1f5fa8', fontweight='bold')
ax.set_xlim(-0.4, 8.6); ax.set_ylim(-2.0, 2.3); ax.axis('off')
ax.set_title('(b) 物理层坍缩：3→1 过参数化', fontsize=14, loc='left')

# ============ (c) 有效内核: γ 混合 ============
ax = fig.add_subplot(G[1, 0])
R4 = np.array(e5['ballistic_J4']['R']); s4 = np.array(e5['ballistic_J4']['sig'])
k = np.arange(1, 25, dtype=np.float64)
gam = np.exp(-(k[:, None] - R4[None, :]) ** 2 / (2 * s4[None, :] ** 2))
gam = gam / gam.sum(1, keepdims=True)
ax.stackplot(k, gam[:, 0], gam[:, 1], gam[:, 2], gam[:, 3],
             colors=['#aec7e8', '#98df8a', '#ffbb78', '#ff9896'], alpha=0.9)
for j in range(4):
    ax.plot(k, gam[:, j], lw=1.0, color='#333', alpha=0.25)
ax.axvline(6.5, ls=':', c='gray')
ax.text(6.8, 0.52, '← 拟合区 k≤6 | 外推区 k>6 →', fontsize=10, color='#555')
lbl = [f'原型{j+1}: R={r:.1f}, σ={s:.2f}' for j, (r, s) in enumerate(zip(R4, s4))]
ax.legend(lbl, fontsize=9, loc='center right')
ax.text(13.5, 0.06, '模型自己发现：k≤2 是近场（bigram），k≥3 起恒为远场原型\nσ≈1 → 高斯核被用成「近开关」——这就是它敢外推的原因',
        fontsize=10.5, color='#333')
ax.set_xlabel('lag k'); ax.set_ylabel('混合权重 γ(k)')
ax.set_title('(c) 有效内核：J 个弹道原型按距离混合  Â(k) = Σ_j γ·φ_j', fontsize=14, loc='left')
ax.set_xlim(1, 24); ax.set_ylim(0, 1)

# ============ (d) 真实语料诊断 ============
ax = fig.add_subplot(G[1, 1])
sig = np.array(diag['signal']); noi = np.array(diag['noise'])
ks = np.arange(1, len(sig) + 1)
ax.plot(ks, sig, 'o-', ms=4, label='信号 ||A(k)−π||', color='#d62728')
ax.plot(ks, noi, 's--', ms=4, label='噪声(分半)', color='#7f7f7f')
kk = np.arange(2, 25, dtype=np.float64)
ax.plot(kk, sig[1] * (kk / 2) ** -0.42, ':', color='#d62728', alpha=0.6, label='幂律拟合 s∝k^(-0.42)')
ax.axvline(6.5, ls=':', c='gray'); ax.text(6.8, 8, '拟合窗 k≤6', fontsize=9, color='#555')
ax.set_yscale('log'); ax.set_xlabel('lag k'); ax.set_ylabel('Frobenius 范数')
ax.legend(fontsize=9, loc='upper right')
ax.text(0.42, 0.16,
        'minimind 实测三前提：\n'
        '① 远场不是常数，是幂律衰减（CV 41.9%）→ 需 J≥4 + 包络\n'
        '② 相邻表差异/噪声 = 1.72 < 3 → 变化大半是噪声，可平滑\n'
        '③ 联合谱前4成分 = 93% → 谱定 J=4（与实测最优一致）',
        transform=ax.transAxes, fontsize=10,
        bbox=dict(fc='#f5f8fd', ec='#4a6fa5', alpha=0.95))
ax.set_title('(d) 真实语料（minimind 16M 字符）的三前提诊断', fontsize=14, loc='left')

# ============ (e) 外推收益分解 ============
ax = fig.add_subplot(G[2, 0])
names = ['① 离散\n只用 k≤6', '② +弹道平滑\n(仍只用 k≤6)', '③ +外推\n(k≤6 → 24)', '④ 离散全量\n(真实 k≤24)']
vals = [12.02, 11.90, 11.62, 11.72]
cols = ['#bbbbbb', '#7fb3d5', '#d62728', '#888888']
bars = ax.bar(names, vals, color=cols, width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')
ax.annotate('平滑 +0.13', xy=(0.5, 12.1), fontsize=9.5, ha='center', color='#1f5fa8')
ax.annotate('外推 +0.28', xy=(1.52, 11.98), fontsize=9.5, ha='center', color='#c0392b')
ax.annotate('外推效率 = 0.28/0.31 = 90%\n用一半数据反超全量', xy=(2.5, 12.35), fontsize=10,
            ha='center', color='#c0392b', fontweight='bold',
            bbox=dict(fc='#fdecea', ec='#c0392b', alpha=0.9))
ax.set_ylim(11.2, 12.7); ax.set_ylabel('测试困惑度（越低越好）')
ax.set_title('(e) v5：收益分解——平滑去噪为主，外推补刀（minimind 字符级）', fontsize=14, loc='left')
ax.tick_params(axis='x', labelsize=10)

# ============ (f) v6 注意力三规模 ============
ax = fig.add_subplot(G[2, 1])
scales = [('char3', '字符级\n3.5M'), ('sub11', '子词级\n11.4M'), ('sub26', '子词级\n26.3M')]
kinds = [('rope', 'RoPE', '#7f7f7f'), ('alibi', 'ALiBi', '#1f77b4'),
         ('ballistic2', '弹道核v6.1', '#d62728')]
gw = 0.26
for gi, (sc, sname) in enumerate(scales):
    for ki, (kd, lab, c) in enumerate(kinds):
        lo, hi = {'char3': (0, 5e6), 'sub11': (5e6, 20e6), 'sub26': (20e6, 1e9)}[sc]
        runs = [r['final_val_loss'] for name, r in s6.items()
                if r['bias'] == kd and 'smoke' not in name and 'fit' not in name
                and lo <= r['params'] < hi]
        if not runs:
            continue
        v = np.exp(np.mean(runs))
        b = ax.bar(gi + (ki - 1) * gw, v, gw, color=c)
        ax.text(b[0].get_x() + gw / 2, v + 0.15, f'{v:.1f}', ha='center', fontsize=9)
ax.set_xticks(range(len(scales))); ax.set_xticklabels([s for _, s in scales], fontsize=11)
ax.set_ylabel('验证 ppl'); ax.set_ylim(0, 26)
ax.legend(['RoPE', 'ALiBi', '弹道核 v6.1'], fontsize=9, loc='upper right')
ax.text(0.35, 0.86, '排序 RoPE < 弹道核 < ALiBi 在全部三个规模成立\n（52 个偏置参数，零推理开销）',
        transform=ax.transAxes, fontsize=10.5, fontweight='bold', color='#c0392b')
ax.set_title('(f) v6：同一个核做成注意力偏置，三规模对照', fontsize=14, loc='left')

# ============ (g) 底部结论带 ============
ax = fig.add_subplot(G[3, :])
ax.axis('off')
ax.text(0.5, 0.86, '一句话原理', fontsize=15, ha='center', fontweight='bold', color='#1f5fa8')
ax.text(0.5, 0.66,
        '把 K 张 lag 表当成「同一条低维轨迹上的采样点」：J 个原型 × 距离核 × 幂律包络——'
        '结构形状沿距离外推，强度按幂律衰减',
        fontsize=13.5, ha='center')
steps = [('v3 压缩', '参数 1/6\n性能不降', '#aec7e8'),
         ('v4 原理', '外推恢复 90%\n平滑+参数共享', '#98df8a'),
         ('v5 真实语料', 'minimind +13.1%\n一键工具 autoballistic', '#ffbb78'),
         ('v6 注意力上车', '胜 ALiBi（三规模）\n与 RoPE 可叠加', '#ff9896')]
for i, (t1, t2, c) in enumerate(steps):
    x = 0.14 + i * 0.24
    ax.add_patch(FancyBboxPatch((x - 0.10, 0.18), 0.20, 0.34, boxstyle='round,pad=0.015',
                                fc=c, ec='#555', alpha=0.85))
    ax.text(x, 0.44, t1, ha='center', fontsize=12.5, fontweight='bold')
    ax.text(x, 0.28, t2, ha='center', fontsize=10.5)
    if i < 3:
        ax.annotate('', xy=(x + 0.135, 0.35), xytext=(x + 0.105, 0.35),
                    arrowprops=dict(arrowstyle='-|>', lw=2, color='#555'))
ax.text(0.5, 0.06,
        'P(v|h) ∝ π_v · Π_j (φ_wj(v)/π_v)^(β_j·κ(δ_j;R_j,σ_j))，κ=高斯距离核 ··· '
        '它就是 ALiBi 的泛化版：带可学习距离核的注意力',
        fontsize=12, ha='center', color='#333',
        bbox=dict(fc='#f5f8fd', ec='#4a6fa5'))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

fig.savefig(os.path.join(HERE, 'fig_explain.png'), dpi=150, bbox_inches='tight', facecolor='white')
print('saved fig_explain.png')
