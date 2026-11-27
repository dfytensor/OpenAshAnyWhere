#!/bin/bash
# chk_chain.sh — 链式训练健康检查（E: 盘上的日志与状态）
ROOT=/mnt/e/OpenASH_loom
echo "=== status ==="
cat $ROOT/chain_status.txt 2>/dev/null
for f in $ROOT/full_v53dml_40k_wsl.log $ROOT/full_v53dmq_40k_wsl.log; do
  if [ -f "$f" ]; then
    echo "=== $(basename $f) (mtime $(stat -c %y "$f" | cut -c1-19)) ==="
    grep -v -i -E "warning|include|define|note:|__triton|POSIX|in file|from /|pyconfig|features.h" "$f" | tail -3
  fi
done
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
echo "=== procs ==="
pgrep -fa "train_ab.py|chain_loom" | grep -v pgrep | cut -c1-110
