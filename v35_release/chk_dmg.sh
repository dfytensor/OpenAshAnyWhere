#!/bin/bash
cd /mnt/f/OpenASH2605
tail -2 full_v53dmg_40k_wsl.log
P=$(pgrep -f "train_ab.py" | head -1)
echo "PID=$P wchan=$(cat /proc/$P/wchan 2>/dev/null)"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
