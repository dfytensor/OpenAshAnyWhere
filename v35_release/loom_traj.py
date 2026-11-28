# -*- coding: utf-8 -*-
"""loom_traj.py — dml / dmq / dms@40k / dmh@40k 的验证与训练损失轨迹对照（排查共同因素）"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import math

LOGS = {
    'dml (bs16x2)': '/mnt/e/OpenASH_loom/full_v53dml_40k/train_log.json',
    'dmq (bs16x2)': '/mnt/e/OpenASH_loom/full_v53dmq_40k/train_log.json',
    'dms@40k (bs32)': '/mnt/f/OpenASH2605/full_v53dms_40k/train_log.json',
    'dmh@40k (bs32)': '/mnt/f/OpenASH2605/full_v53dmh_40k/train_log.json',
}
D = {}
for k, p in LOGS.items():
    lg = json.load(open(p, encoding='utf-8'))['log']
    D[k] = {e['step']: e for e in lg}
steps_val = [s for s in D['dml (bs16x2)'] if 'val_loss' in D['dml (bs16x2)'][s]]
print('=== val loss @ eval steps ===')
print('step'.rjust(7) + ''.join(k.rjust(18) for k in D))
for s in steps_val:
    row = str(s).rjust(7)
    for k in D:
        e = D[k].get(s)
        row += (('%.4f' % e['val_loss']) if e and 'val_loss' in e else '-').rjust(18)
    print(row)
print()
print('=== train loss @ matched steps（单批噪声大，仅看量级）===')
for s in (1000, 5000, 10000, 20000, 30000, 40000):
    row = str(s).rjust(7)
    for k in D:
        e = D[k].get(s)
        row += (('%.3f' % e['loss']) if e else '-').rjust(18)
    print(row)
print()
print('=== dml 与 dmq 的训练损失逐点差（同种子同数据顺序下，若两条轨迹几乎相同则差值≈批间噪声 0）===')
common = sorted(set(D['dml (bs16x2)']) & set(D['dmq (bs16x2)']))
diffs = [(s, D['dml (bs16x2)'][s]['loss'] - D['dmq (bs16x2)'][s]['loss']) for s in common if 'loss' in D['dml (bs16x2)'][s] and 'loss' in D['dmq (bs16x2)'][s]]
for lo, hi in ((0, 2000), (2000, 10000), (10000, 20000), (20000, 30000), (30000, 40001)):
    dd = [abs(a - b) for s, a, b in [(s, D['dml (bs16x2)'][s]['loss'], D['dmq (bs16x2)'][s]['loss']) for s in common if lo < s <= hi]]
    if dd:
        print('  steps %5d-%5d: mean|Δloss| = %.4f  (n=%d)' % (lo, hi, sum(dd) / len(dd), len(dd)))
