# -*- coding: utf-8 -*-
"""v6 分析: 读取 runs/, 输出对照表 + 曲线图 + 学到的距离核
  - 各方案 val loss 曲线 / 终值 ppl
  - ballistic 系学到的逐头混合核 vs v5 n-gram 弹道 γ 曲线 (跨模型对照!)
"""
import sys, os, json, glob
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import core5 as C

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
COLOR = dict(rope='#7f7f7f', alibi='#1f77b4', ballistic='#9467bd', ballistic_ng='#c5b0d5',
             ballistic2='#d62728', ballistic2fit='#ff9896', rope_ballistic='#e377c2')
NAME = dict(rope='RoPE', alibi='ALiBi', ballistic='弹道核v6(硬地板)', ballistic_ng='弹道核v6(ng初始化)',
            ballistic2='弹道核v6.1(地板+分化)', rope_ballistic='RoPE+弹道核(加性)')

plt = C.setup_font()
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))

summary = {}
for rdir in sorted(glob.glob(os.path.join(RUNS, '*'))):
    if 'smoke' in os.path.basename(rdir):
        continue
    res_p = os.path.join(rdir, 'result.json')
    if not os.path.exists(res_p):
        continue
    res = json.load(open(res_p, encoding='utf-8'))
    key = res['bias']
    name = os.path.basename(rdir)
    p = res.get('params', 0)
    res['fit_init'] = 'fit' in name and name.startswith('sub')
    if p < 5_000_000 and not name.startswith('sub'):
        res['scale'] = 'char3'
    elif p < 15_000_000:
        res['scale'] = 'sub11'
    else:
        res['scale'] = 'sub26'
    summary[rdir] = res
    # 曲线
    lines = [json.loads(l) for l in open(os.path.join(rdir, 'log.jsonl'), encoding='utf-8')]
    steps = [l['step'] for l in lines]
    vals = [l['val_loss'] for l in lines]
    ax[0].plot(steps, vals, '-', color=COLOR[key], alpha=0.9 if res['seed'] == 1 else 0.45,
               label=NAME[key] if res['seed'] == 1 else None)

# 终值汇总 (按 scale × kind 分组)
SHORT = dict(rope='RoPE', alibi='ALiBi', ballistic='弹道v6', ballistic_ng='弹道v6ng',
             ballistic2='弹道v6.1', rope_ballistic='RoPE+弹道')
scales = [('char3', '字符级 3.5M'), ('sub11', '子词级 11.4M'), ('sub26', '子词级 26.3M')]
finals = {}
for sc, _ in scales:
    for r in summary.values():
        if r['scale'] != sc:
            continue
        k = r['bias'] + ('fit' if (r.get('fit_init') and r['bias'] == 'ballistic2') else '')
        finals.setdefault((sc, k), []).append(r['final_val_loss'])
for (sc, k), v in sorted(finals.items()):
    print(f"{sc:>6} {k:>16}: val {np.mean(v):.4f}  ppl {np.exp(np.mean(v)):.2f}  (n={len(v)})")
print('  各规模: 弹道核(v6.1) vs ALiBi:')
for sc, _ in scales:
    a = finals.get((sc, 'ballistic2')); b = finals.get((sc, 'alibi'))
    if a and b:
        print(f'   {sc}: Δ = {np.mean(b)-np.mean(a):+.4f} nats  (ppl {np.exp(np.mean(b)):.2f} vs {np.exp(np.mean(a)):.2f})')

# 中栏: 三尺度 × 四方案 grouped bars
ax[1].set_ylabel('验证 ppl (越低越好)')
groupk = ['rope', 'alibi', 'ballistic2', 'ballistic2fit']
gshort = dict(rope='RoPE', alibi='ALiBi', ballistic2='弹道v6.1', ballistic2fit='弹道v6.1(n-gram初始化)')
gw = 0.2
for gi, (sc, sname) in enumerate(scales):
    for ki, k in enumerate(groupk):
        v = finals.get((sc, k))
        if not v:
            continue
        b = ax[1].bar(gi + (ki - 1.5) * gw, np.exp(np.mean(v)), gw, color=COLOR[k],
                      hatch='//' if sc != 'char3' else '', alpha=1.0 if sc == 'char3' else (0.8 if sc == 'sub11' else 0.62))
        ax[1].text(b[0].get_x() + gw / 2, b[0].get_height() + 0.2, f'{np.exp(np.mean(v)):.1f}', fontsize=6.5,
                   ha='center', rotation=0)
ax[1].set_xticks(range(len(scales))); ax[1].set_xticklabels([s for _, s in scales], fontsize=9)
ax[1].set_ylim(0, max(np.exp(np.mean(v)) for v in finals.values()) * 1.15)
ax[1].set_title('三规模 × 方案 (6000 步, 1~2 种子)')
from matplotlib.patches import Patch
ax[1].legend(handles=[Patch(color=COLOR[k], label=gshort[k]) for k in groupk], fontsize=7)
dgrid = np.arange(1, 257, dtype=np.float64)
for rd, style, alpha in [('ballistic2_s1', '-', 0.85), ('ropeball_s1', '--', 0.6),
                         ('sub_ballistic2_s1', ':', 0.9)]:
    rp = os.path.join(RUNS, rd, 'result.json')
    if not os.path.exists(rp):
        continue
    res = json.load(open(rp, encoding='utf-8'))
    bp = res['bias_params']
    subtag = rd.startswith('sub_')
    for h in range(len(bp['R'])):
        R = np.array(bp['R'][h]); sg = np.array(bp['sig'][h]); w = np.array(bp['w'][h])
        mix = (np.exp(-(dgrid[:, None] - R[None, :]) ** 2 / (2 * sg[None, :] ** 2)) * w[None, :]).sum(1)
        mix = mix / mix.max()
        band = ["近", "中", "远", "超远"][h]
        if subtag:
            lab = f'子词级 头{h}({band})' if h < 2 else None
        else:
            lab = f'{SHORT[res["bias"]]} 头{h} ({band})'
        ax[2].plot(dgrid, mix, style, color=plt.cm.tab10(h), alpha=alpha,
                   lw=1.2 if subtag else 1.5, label=lab)
exp5p = r'F:\夸克\token_period_package\v5_minimind\exp5.json'
if os.path.exists(exp5p):
    e5 = json.load(open(exp5p, encoding='utf-8'))
    Rj = np.array(e5['ballistic_J4']['R']); sg = np.array(e5['ballistic_J4']['sig'])
    kg = np.arange(1, 25, dtype=np.float64)
    g = np.exp(-(kg[:, None] - Rj[None, :]) ** 2 / (2 * sg[None, :] ** 2))
    g = g / np.maximum(g.sum(1, keepdims=True), 1e-12)
    for j in range(g.shape[1]):
        col = g[:, j] / max(g[:, j].max(), 1e-9)
        ax[2].plot(kg, col, ':', color='k', alpha=0.45,
                   label='v5 n-gram 弹道原型' if j == 0 else None)
ax[2].set_xscale('log'); ax[2].set_xlim(1, 256)
ax[2].set_xlabel('距离 d (token)'); ax[2].set_ylabel('核形状 (峰值归一)')
ax[2].set_title('学到的逐头距离核 vs v5 n-gram 原型')
ax[2].legend(fontsize=7)

ax[0].set_xlabel('step'); ax[0].set_ylabel('验证 loss'); ax[0].set_ylim(2.8, None)
ax[0].legend(fontsize=8); ax[0].set_title('验证损失曲线 (实线=种子1, 虚线淡=种子2)')
fig.suptitle('距离轴参数共享 as 注意力偏置: RoPE / ALiBi / 弹道核 对照', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(RUNS), 'fig6_bias.png'), dpi=130, bbox_inches='tight')

with open(os.path.join(os.path.dirname(RUNS), 'summary6.json'), 'w', encoding='utf-8') as f:
    json.dump({os.path.basename(d): r for d, r in summary.items()}, f, ensure_ascii=False, indent=1)
print('[saved fig6_bias.png, summary6.json]')
