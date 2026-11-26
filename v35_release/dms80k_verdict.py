# -*- coding: utf-8 -*-
"""dms80k_verdict.py — dms@80k 同 token 终局（预注册判据执行）"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
r = json.load(open(r'F:\OpenASH2605\longrange_results.json', encoding='utf-8'))
a = r['A_len_ppl']
S = a['full_v53dms_80k']
G = a['full_gdn_160k']
D = a['full_v53dme_80k']
G1 = a['full_gdn_80k_fixed']
Ls = (64, 128, 256, 384, 512, 1024, 2048)

print('=== A: ppl by length（dms@80k / GDN@160k 同 982M tokens）===')
print('L'.rjust(6) + 'dms@80k'.rjust(12) + 'GDN@160k'.rjust(12) + 'dme@80k'.rjust(12)
      + 'GDN-1ep'.rjust(12) + 'dms/GDN'.rjust(12) + '裁决'.rjust(8))
wins = ties = losses = 0
for i, L in enumerate(Ls):
    rel = S[i][2] / G[i][2] - 1
    tag = '胜' if rel < -0.003 else ('平' if rel <= 0.003 else '负')
    if tag == '胜':
        wins += 1
    elif tag == '平':
        ties += 1
    else:
        losses += 1
    print(str(L).rjust(6) + ('%.2f' % S[i][2]).rjust(12) + ('%.2f' % G[i][2]).rjust(12)
          + ('%.2f' % D[i][2]).rjust(12) + ('%.2f' % G1[i][2]).rjust(12)
          + ('%+.1f%%' % (rel * 100)).rjust(12) + tag.rjust(8))

print()
print('=== 预注册判据 ===')
ok1 = (wins + ties) >= 5
print('1 主判据: %d胜 %d平 %d负 → 不劣于 GDN 的长度数 = %d/7 → %s' % (
    wins, ties, losses, wins + ties, '✅ 等规模同 token LM 超越' if ok1 else '❌ 未达 5/7'))
print('   头条 @384: dms %.2f vs GDN %.2f → %+.1f%%' % (S[3][2], G[3][2], (S[3][2] / G[3][2] - 1) * 100))
i10, i20 = 5, 6
g10 = S[i10][2] / G[i10][2] - 1
g20 = S[i20][2] / G[i20][2] - 1
d10 = D[i10][2] / G[i10][2] - 1
d20 = D[i20][2] / G[i20][2] - 1
print('2 长程专项: @1024 对 GDN %+.1f%%（dme@80k 为 %+.1f%%）| @2048 %+.1f%%（原 %+.1f%%）→ %s' % (
    g10 * 100, d10 * 100, g20 * 100, d20 * 100,
    '✅ 至少减半' if (abs(g10) <= abs(d10) / 2 and abs(g20) <= abs(d20) / 2) else '❌'))
print('4 规模存续（门控在 2-epoch 尺度的净效应 dms vs dme）:')
for i, L in enumerate(Ls):
    print('   L%-5d dms %.2f vs dme %.2f → %+.1f%%' % (L, S[i][2], D[i][2], (S[i][2] / D[i][2] - 1) * 100))

b = r['B_recall']
gaps = ['32', '64', '128', '256', '384']
models = list(b[gaps[0]].keys())
print()
print('=== 3 召回标价: paired recall (rep / gain) ===')
print('gap'.rjust(6) + ''.join(m.replace('full_', '').rjust(22) for m in models))
for gap in gaps:
    print(gap.rjust(6) + ''.join(('%6.2f %+0.2f' % (b[gap][m][0], b[gap][m][2])).rjust(22) for m in models))
ms = [m for m in models if 'dms' in m][0]
md = [m for m in models if 'dme' in m][0]
mg = [m for m in models if '160k' in m][0]
print('dms vs dme@80k:', ' '.join('%+.2f' % (b[g][ms][0] - b[g][md][0]) for g in gaps))
print('dms vs GDN@160k:', ' '.join('%+.2f' % (b[g][ms][0] - b[g][mg][0]) for g in gaps))

c = r['C_pos']
print()
print('=== C: pos-bucket @384 (b1..b8) ===')
for m, v in c.items():
    if isinstance(v, list):
        print('  %-20s' % m.replace('full_', ''), ' '.join('%.2f' % x for x in v))
