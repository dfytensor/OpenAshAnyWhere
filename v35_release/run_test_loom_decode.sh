#!/bin/bash
cd /mnt/f/OpenASH2605
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
wsl_venv/bin/python -u test_loom_decode.py > test_loom_decode.out 2>&1
echo "exit=$?"
grep -E "显存|数据流|窗口@|O\(1\)|解码 @ctx|LOOM DECODE|OutOfMemoryError" test_loom_decode.out
grep -A3 "OutOfMemoryError" test_loom_decode.out | head -4
