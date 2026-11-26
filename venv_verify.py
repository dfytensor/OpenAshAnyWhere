# -*- coding: utf-8 -*-
"""venv_verify.py — 用 pip RECORD 的 sha256 逐文件校验 venv（运行环境完整性，不在 git 内）
用法：wsl_venv/bin/python venv_verify.py
"""
import base64
import csv
import hashlib
import os
import sys

import site

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
roots = site.getsitepackages()
print('site-packages:', roots)

ok = bad = missing = nohash = 0
bad_list = []
packages = []
for root in roots:
    for d in sorted(os.listdir(root)):
        if d.endswith('.dist-info'):
            packages.append(d)
            rec = os.path.join(root, d, 'RECORD')
            if not os.path.exists(rec):
                print('  (no RECORD)', d)
                continue
            with open(rec, newline='', encoding='utf-8') as f:
                for row in csv.reader(f):
                    if len(row) < 2 or not row[1]:
                        nohash += 1
                        continue
                    rel, h = row[0], row[1]
                    if not h.startswith('sha256='):
                        nohash += 1
                        continue
                    p = os.path.join(root, rel)
                    if not os.path.exists(p):
                        missing += 1
                        bad_list.append((d, rel, 'MISSING'))
                        continue
                    h256 = hashlib.sha256()
                    with open(p, 'rb') as fh:
                        while True:
                            b = fh.read(1 << 20)
                            if not b:
                                break
                            h256.update(b)
                    want = base64.urlsafe_b64encode(h256.digest()).rstrip(b'=').decode()
                    if want == h[7:]:
                        ok += 1
                    else:
                        bad += 1
                        bad_list.append((d, rel, 'HASH MISMATCH'))

print('packages:', len(packages))
print('files OK: %d | HASH MISMATCH: %d | MISSING: %d | (unhashed entries e.g. RECORD/pyc: %d)' % (ok, bad, missing, nohash))
for x in bad_list[:40]:
    print('  BAD:', x)
