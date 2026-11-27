#!/bin/bash
# inst_speed.sh — 量第二臂的瞬时步速（两次采样相隔 120 秒；日志每 100 步一行）
LOG=/mnt/e/OpenASH_loom/full_v53dmq_80k_wsl.log
s1=$(grep -oE "step +[0-9]+ \|" $LOG | tail -1 | grep -oE "[0-9]+")
sleep 120
s2=$(grep -oE "step +[0-9]+ \|" $LOG | tail -1 | grep -oE "[0-9]+")
awk -v a="$s1" -v b="$s2" 'BEGIN {
  d = b - a;
  if (d > 0) printf "steps %d -> %d in 120s: ~%.0f ms/step (100 步一格, 粗略)\n", a, b, 120000 / d;
  else printf "steps %d -> %d in 120s: 不足一格, 无法估计\n", a, b;
  printf "remaining steps: %d\n", 80000 - b;
}'
tail -1 $LOG
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
