#!/bin/bash
# chk_light.sh [wait_seconds] — 等待后检查：训练瞬时步速 + 轻量评测进度 + GPU
sleep ${1:-0}
cd /mnt/f/OpenASH2605
echo "now: $(date '+%F %T')"
bash inst_speed.sh
echo "=== light eval ==="
grep -v -i -E "warning|include|define|note:|__triton|POSIX|in file|from /|pyconfig|features.h" /mnt/e/OpenASH_loom/light_eval_dml80k.out | tail -5
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
