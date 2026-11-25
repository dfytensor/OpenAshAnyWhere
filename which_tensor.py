# -*- coding: utf-8 -*-
"""which_tensor.py — 定位 dme@40k 损坏成员 data/158 对应的参数名 + 检查可用副本"""
import pickletools
import subprocess
import sys
import zipfile

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
p = r'F:\OpenASH2605\full_v53dme_40k\model.pt'
z = zipfile.ZipFile(p)
pk = z.read([n for n in z.namelist() if n.endswith('data.pkl')][0])

# 扫描 pickle 操作码：定位 storage key '158' 前最近的参数名（UNICODE/SHORT_BINUNICODE）
ops = list(pickletools.genops(pk))
hit = None
for i, (op, arg, pos) in enumerate(ops):
    if op.name in ('BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE') and arg == '158':
        prev = [a for (o, a, _) in ops[max(0, i - 12):i] if isinstance(a, str)]
        print('storage key 158 at opcode', i, '| preceding strings:', prev[-4:])
        hit = i
if hit is None:
    print('storage key 158 NOT referenced in data.pkl')

# 存储类型/长度（REDUCE 前的 BinUnICODE + BININT）
for i, (op, arg, pos) in enumerate(ops):
    if op.name in ('BINUNICODE', 'SHORT_BINUNICODE') and arg == '158':
        for (o, a, _) in ops[i:i + 8]:
            print('   ', o.name, repr(a)[:40])

print()
# 是否有副本
for cand in [r'F:\OpenASH2605\v35_release', r'F:\OpenASH2605\backup', r'F:\OpenASH_bak']:
    import os
    q = os.path.join(cand, 'full_v53dme_40k', 'model.pt')
    print(cand, '->', 'EXISTS' if os.path.exists(q) else 'no')

r = subprocess.run(['git', '-C', r'F:\OpenASH2605', 'ls-files', 'full_v53dme_40k'],
                   capture_output=True, text=True)
print('git tracked under full_v53dme_40k:', repr(r.stdout[:200]))
r = subprocess.run(['git', '-C', r'F:\OpenASH2605', 'lfs', 'ls-files'], capture_output=True, text=True)
lfs = [ln for ln in r.stdout.splitlines() if 'dme_40k' in ln]
print('git lfs entries for dme_40k:', lfs[:3] or 'none', '| lfs total lines:', len(r.stdout.splitlines()))
