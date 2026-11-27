#!/bin/bash
cd /mnt/f/OpenASH2605
wsl_venv/bin/python -u mem_probe2.py > /tmp/mem_probe2.out 2>&1
echo "exit=$?"
grep -E "allocated|OOM|ALL OK|FAILED" /tmp/mem_probe2.out
