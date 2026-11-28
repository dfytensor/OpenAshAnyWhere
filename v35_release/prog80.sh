#!/bin/bash
# prog80.sh — 80k 第一臂进度：当前步 / 已有的验证点 / 对照 dms@80k 同步数验证值
ROOT=/mnt/e/OpenASH_loom
echo "now: $(date '+%F %T')"
echo "=== dml@80k 已有验证点（每 8000 步：val ppl）==="
grep -F ">> val" $ROOT/full_v53dml_80k_wsl.log | awk '{printf "step %s: ppl %s; ", NR*8000, $7}'
echo
echo "=== dms@80k（对照）同步数验证 ppl ==="
grep -F ">> val" /mnt/f/OpenASH2605/full_v53dms_80k_wsl.log | awk '{printf "step %s: ppl %s; ", NR*8000, $7}'
echo
echo "=== 当前 ==="
tail -1 $ROOT/full_v53dml_80k_wsl.log
