#!/bin/bash
echo "=== gpu compute apps ==="
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader
echo "=== now ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
echo "=== my python procs ==="
pgrep -fa "wsl_venv/bin/python" | grep -v pgrep | head -5
