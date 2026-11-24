#!/bin/bash
cd /mnt/f/OpenASH2605
tail -3 full_v53dme_40k_t1024_wsl.log
P=$(pgrep -f "train_ab.py --arch v53dme" | head -1)
echo "PID=$P wchan=$(cat /proc/$P/wchan 2>/dev/null)"
[ -n "$P" ] && cat /proc/$P/io | grep read_bytes
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
