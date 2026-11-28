#!/bin/bash
cd /mnt/f/OpenASH2605
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
wsl_venv/bin/python -u decode_leak_probe2.py > /tmp/decode_leak_probe2.out 2>&1
echo "exit=$?"
grep -E "allocated|PROBE2|Error" /tmp/decode_leak_probe2.out
