# -*- coding: utf-8 -*-
"""patch fig_explain.py: (f) 面板规模选择 + 缺字形替换"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
p = r'F:\夸克\token_period_package\v5_minimind\fig_explain.py'
raw = open(p, 'rb').read().decode('utf-8')

bad_f = (
    "        runs = [r['final_val_loss'] for r in s6.values()\n"
    "                if r['bias'] == kd and ((r.get('src') == sc) or\n"
    "                                        (sc == 'char3' and 'src' not in r and r['params'] < 5e6) or\n"
    "                                        (sc == 'sub11' and r['params'] < 15e6) or\n"
    "                                        (sc == 'sub26' and r['params'] >= 15e6))]"
)
good_f = (
    "        lo, hi = {'char3': (0, 5e6), 'sub11': (5e6, 20e6), 'sub26': (20e6, 1e9)}[sc]\n"
    "        runs = [r['final_val_loss'] for name, r in s6.items()\n"
    "                if r['bias'] == kd and 'smoke' not in name and 'fit' not in name\n"
    "                and lo <= r['params'] < hi]"
)
assert bad_f in raw, 'f-panel pattern not found'
raw = raw.replace(bad_f, good_f)

for a, b in [('\u21d2', '\u2192'), ('\u21a6', '\u2192'), ('\u03a3\u2c7c \u03b3\u00b7\u03c6\u2c7c', '\u03a3_j \u03b3\u00b7\u03c6_j')]:
    raw = raw.replace(a, b)

open(p, 'wb').write(raw.encode('utf-8'))
print('patched ok')
