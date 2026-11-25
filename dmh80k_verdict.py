# -*- coding: utf-8 -*-
"""dmh80k_verdict.py — dmh@80k 同 token 终局（预注册判据）+ 训练轨迹对照"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
root = r'F:\OpenASH2605'
r = json.load(open(root + r'\longrange_results.json', encoding='utf-8'))
a = r['A_len_ppl']
H = a['full_v53dmh_80k']
G = a['full_gdn_160k']
S = a['full_v53dms_80k']
D = a['full_v53dme_80k']
Ls = (64, 128, 256, 384, 512, 1024, 2048)

print('=== A: ppl by length（同 982M tokens）===')
print('L'.rjust(6) + 'dmh@80k'.rjust(11) + 'GDN@160k'.rjust(11) + 'dms@80k'.rjust(11) + 'dme@80k'.rjust(11)
      + 'dmh/GDN'.rjust(10) + '裁决'.rjust(6) + 'dmh/dms'.rjust(10))
wins = ties = losses = 0
for i, L in enumerate(Ls):
    rel = H[i][2] / G[i][2] - 1
    tag = '胜' if rel < -0.003 else ('平' if rel <= 0.003 else '负')
    wins += tag == '胜'
    ties += tag == '平'
    losses += tag == '负'
    print(str(L).rjust(6) + ('%.2f' % H[i][2]).rjust(11) + ('%.2f' % G[i][2]).rjust(11)
          + ('%.2f' % S[i][2]).rjust(11) + ('%.2f' % D[i][2]).rjust(11)
          + ('%+.1f%%' % (rel * 100)).rjust(10) + tag.rjust(6)
          + ('%+.1f%%' % ((H[i][2] / S[i][2] - 1) * 100)).rjust(10))
print()
print('=== 预注册判据 ===')
print('1 主判据: %d胜 %d平 %d负 → %d/7 → %s' % (
    wins, ties, losses, wins + ties, '✅ 等规模同 token LM 超越' if wins + ties >= 5 else '❌ 未达 5/7'))
print('   头条 @384: dmh %.2f vs GDN %.2f → %+.1f%%' % (H[3][2], G[3][2], (H[3][2] / G[3][2] - 1) * 100))
print('2 长程: @1024 %+.1f%%（dms +2.1%%）| @2048 %+.1f%%（dms +3.2%%）' % (
    (H[5][2] / G[5][2] - 1) * 100, (H[6][2] / G[6][2] - 1) * 100))

b = r['B_recall']
gaps = ['32', '64', '128', '256', '384']
pick = lambda s: [m for m in b[gaps[0]] if s in m][0]
mh, mg, ms, md = pick('dmh'), pick('gdn'), pick('dms'), pick('dme')
print()
print('3 召回标价: paired recall (rep / gain)')
print('gap'.rjust(6) + ''.join(x.replace('full_', '').rjust(22) for x in (mh, mg, ms, md)))
for gap in gaps:
    print(gap.rjust(6) + ''.join(('%6.2f %+0.2f' % (b[gap][x][0], b[gap][x][2])).rjust(22)
                                 for x in (mh, mg, ms, md)))
print('   dmh−GDN@160k:', ' '.join('%+.2f' % (b[g][mh][0] - b[g][mg][0]) for g in gaps))
print('   dmh−dms@80k  :', ' '.join('%+.2f' % (b[g][mh][0] - b[g][ms][0]) for g in gaps))
print('   dmh−dme@80k  :', ' '.join('%+.2f' % (b[g][mh][0] - b[g][md][0]) for g in gaps))

c = r['C_pos']
print()
print('=== C: pos-bucket @384 (b1..b8) ===')
for m, v in c.items():
    if isinstance(v, list):
        print('  %-20s' % m.replace('full_', ''), ' '.join('%.2f' % x for x in v))

# 训练轨迹对照（排查累积实现 vs 规模效应）
print()
print('=== 4 规模存续 / 轨迹对照：dmh@80k(bs16x2) vs dms@80k(bs32) ===')


def traj(key):
    lg = json.load(open(root + '\\%s\\train_log.json' % key, encoding='utf-8'))['log']
    v = {e['step']: e for e in lg if 'val_loss' in e}
    l = {e['step']: e for e in lg if e['step'] in (1000, 5000, 10000, 20000, 40000, 60000, 79000)}
    return v, l


vh, lh = traj('full_v53dmh_80k')
vs, ls = traj('full_v53dms_80k')
import math
print('step   dmh_val  dms_val   Δ     | train loss dmh / dms')
for s in sorted(set(vh) & set(vs)):
    print('%-6d %.4f   %.4f   %+.3f  | %s' % (
        s, vh[s]['val_loss'], vs[s]['val_loss'], vh[s]['val_loss'] - vs[s]['val_loss'],
        ('%.3f / %.3f' % (lh[s]['loss'], ls[s]['loss'])) if s in lh and s in ls else ''))
for s in sorted(set(lh) & set(ls)):
    if s not in vh:
        print('%-6d   -        -       -    | %.3f / %.3f' % (s, lh[s]['loss'], ls[s]['loss']))
