#!/bin/bash
cd /mnt/f/OpenASH2605
grep -F ">> val" full_v53dme_40k_t1024_wsl.log
tail -3 full_v53dme_40k_t1024_wsl.log
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
