# -*- coding: utf-8 -*-
"""长上下文外推分析: lc_* 运行的 length_eval 表 + 外推曲线图"""
import sys, os, json, glob
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
sys.path.insert(0, r'F:\OpenASH2605\v6_attbias')
sys.path.insert(0, 'F:/夸克/token_period_package/v5_minimind')
import core5 as C

RUNS = r'F:\OpenASH2605\v6_attbias\runs'
plt = C.setup_font()
COLOR = dict(rope='#7f7f7f', alibi='#1f77b4', ballistic2='#d62728')
NAME = dict(rope='RoPE', alibi='ALiBi', ballistic2='弹道核 v6.1')

data = {}
for rp in sorted(glob.glob(os.path.join(RUNS, 'lc_*', 'result.json'))):
    r = json.load(open(rp, encoding='utf-8'))
    if 'length_eval' not in r:
        continue
    key = r['bias']
    data.setdefault(key, []).append(r)

# 表
lens = sorted({int(L) for r in data.values() for rr in r for L in rr['length_eval']})
print(f"{'方案':<12}" + ''.join(f"{('ctx='+str(L)):>14}" for L in lens))
table = {}
for k, rs in data.items():
    row = []
    for L in lens:
        ps = [np.exp(rr['length_eval'][str(L)]) for rr in rs if str(L) in rr['length_eval']]
        row.append(float(np.mean(ps)))
    table[k] = row
    print(f"{NAME[k]:<12}" + ''.join(f"{v:>14.2f}" for v in row))
base = [np.exp(np.mean([rr['length_eval'][str(L)] for rr in rs if str(L) in rr['length_eval']]))
        for L in lens] if 'rope' in data else None

print('\n相对 ctx=512 的退化倍数:')
for k, row in table.items():
    print(f"  {NAME[k]:<12}: " + '  '.join(f"ctx={L}: x{row[i]/row[0]:.2f}" for i, L in enumerate(lens)))

# 图: 外推曲线 (log-y)
plt.figure(figsize=(8.5, 5.2))
for k, row in table.items():
    plt.plot(lens, row, 'o-', color=COLOR[k], label=NAME[k], lw=2)
plt.xscale('log', base=2); plt.yscale('log')
plt.xticks(lens, [str(L) for L in lens])
plt.axvline(512, ls=':', c='gray'); plt.text(530, plt.gca().get_ylim()[0] * 1.05, '← 训练域 | 零样本外推 →', fontsize=9, color='#555')
plt.xlabel('评估上下文长度 (token)'); plt.ylabel('验证 ppl (log)')
plt.title('长度外推: ctx=512 训练, 更长距离零样本评估\nminimind 子词 11.4M 模型')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(os.path.dirname(RUNS), 'fig_lc.png')
plt.savefig(out, dpi=130, bbox_inches='tight')
json.dump(dict(lens=lens, table=table), open(os.path.join(os.path.dirname(RUNS), 'lc_summary.json'), 'w', encoding='utf-8'), indent=1)
print(f'[saved {out}, lc_summary.json]')
