#!/bin/bash
# chk_chain80.sh — 80k 链健康检查
ROOT=/mnt/e/OpenASH_loom
echo "=== status ==="
cat $ROOT/chain80_status.txt 2>/dev/null
for f in $ROOT/full_v53dml_80k_wsl.log $ROOT/full_v53dmq_80k_wsl.log; do
  if [ -f "$f" ]; then
    echo "=== $(basename $f) (mtime $(stat -c %y "$f" | cut -c1-19)) ==="
    grep -v -i -E "warning|include|define|note:|__triton|POSIX|in file|from /|pyconfig|features.h" "$f" | tail -3
  fi
done
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
echo "=== procs ==="
pgrep -fa "train_ab.py|chain_loom80k" | grep -v pgrep | cut -c1-100
