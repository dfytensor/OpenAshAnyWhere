# -*- coding: utf-8 -*-
"""rev80_diag.py — 40k 领先 → 80k 反转：dml@80k vs dms@80k 的训练/验证轨迹对照"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import math

LOGS = {
    'dml@80k (bs16x2)': '/mnt/e/OpenASH_loom/full_v53dml_80k/train_log.json',
    'dms@80k (bs32)': '/mnt/f/OpenASH2605/full_v53dms_80k/train_log.json',
    'dmh@80k (bs16x2)': '/mnt/f/OpenASH2605/full_v53dmh_80k/train_log.json',
    'dme@80k (bs32)': '/mnt/f/OpenASH2605/full_v53dme_80k/train_log.json',
}
D = {}
for k, p in LOGS.items():
    D[k] = {e['step']: e for e in json.load(open(p, encoding='utf-8'))['log']}

names = list(D)
steps_val = [s for s in D[names[0]] if 'val_loss' in D[names[0]][s]]
print('=== 验证 ppl（每 8000 步）===')
print('step'.rjust(7) + ''.join(n.rjust(18) for n in names))
for s in steps_val:
    print(str(s).rjust(7) + ''.join(
        (('%.2f' % math.exp(D[n][s]['val_loss'])) if s in D[n] and 'val_loss' in D[n][s] else '-').rjust(18) for n in names))


def tr(n, lo, hi):
    xs = [D[n][s]['loss'] for s in D[n] if lo < s <= hi and 'loss' in D[n][s]]
    return sum(xs) / len(xs) if xs else float('nan')


print()
print('=== 训练损失（区间平均；同数据顺序，可比）===')
print('区间'.rjust(14) + ''.join(n.rjust(18) for n in names) + '   dml−dms'.rjust(10))
for lo, hi in ((0, 8000), (8000, 16000), (16000, 24000), (24000, 32000), (32000, 40000),
               (40000, 48000), (48000, 56000), (56000, 64000), (64000, 72000), (72000, 80000)):
    row = ('%d-%d' % (lo, hi)).rjust(14)
    vals = {n: tr(n, lo, hi) for n in names}
    for n in names:
        row += ('%.4f' % vals[n]).rjust(18)
    row += ('%+.4f' % (vals[names[0]] - vals[names[1]])).rjust(10)
    print(row)
