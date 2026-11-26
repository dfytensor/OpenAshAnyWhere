# -*- coding: utf-8 -*-
"""verify_ckpts.py — checkpoint zip 完整性校验（同盘静默损坏排查）"""
import glob
import os
import sys
import zipfile

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
keys = sys.argv[1:] if len(sys.argv) > 1 else None
root = r'F:\OpenASH2605'
pats = ['full_v53dmh_80k', 'full_gdn_160k', 'full_v53dms_80k', 'full_v53dme_80k']
if keys:
    pats = keys
bad_total = 0
for k in pats:
    p = os.path.join(root, k, 'model.pt')
    if not os.path.exists(p):
        print('%-24s MISSING' % k)
        bad_total += 1
        continue
    try:
        z = zipfile.ZipFile(p)
        r = z.testzip()
        n = len(z.infolist())
        if r is None:
            print('%-24s OK (%d entries, %.0f MB)' % (k, n, os.path.getsize(p) / 1e6))
        else:
            print('%-24s CORRUPT: %r' % (k, r))
            bad_total += 1
    except Exception as e:
        print('%-24s ERROR: %s' % (k, repr(e)[:160]))
        bad_total += 1
print('bad:', bad_total)
