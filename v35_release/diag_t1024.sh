#!/bin/bash
cd /mnt/f/OpenASH2605
echo "=== log mtime vs now ==="
date
stat -c '%y' full_v53dme_40k_t1024_wsl.log
echo "=== ms/step trajectory (every 2000 steps) ==="
grep -E "step +(1000|2000|4000|6000|8000|10000|11000|12000|12500|13000|13100) \|" full_v53dme_40k_t1024_wsl.log
echo "=== last 3 lines ==="
tail -3 full_v53dme_40k_t1024_wsl.log
echo "=== python procs ==="
ps aux | grep -E "python" | grep -v grep | awk '{print $2, $3, $4, $10, $11, $12, $13, $14, $15}' | head -5
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw --format=csv,noheader
echo "=== dmesg tail (xid/oom/throttle) ==="
dmesg 2>/dev/null | grep -iE "xid|oom|nvrm|throttl" | tail -5
