#!/bin/bash
cd /mnt/f/OpenASH2605
echo "=== file ==="
ls -la minimind_data/pretrain_cached_1270238_384.pt
echo "=== head bytes (should be 'PK') ==="
head -c 4 minimind_data/pretrain_cached_1270238_384.pt | xxd | head -1
echo "=== zip integrity (python zipfile) ==="
wsl_venv/bin/python - <<'EOF'
import zipfile, sys
p = 'minimind_data/pretrain_cached_1270238_384.pt'
try:
    z = zipfile.ZipFile(p)
    bad = z.testzip()
    names = z.namelist()
    print('zip OK, entries:', len(names), 'first bad:', bad)
    print('first names:', names[:3])
except Exception as e:
    print('ZIP ERROR:', repr(e)[:300])
EOF
