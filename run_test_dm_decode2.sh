#!/bin/bash
cd /mnt/f/OpenASH2605
wsl_venv/bin/python -u test_dm_decode2.py > test_dm_decode2.out 2>&1
echo "exit=$?"
grep -v -i -E "warning|include|define|note:|__triton|POSIX|in file|from /|pyconfig|features.h" test_dm_decode2.out | tail -45
