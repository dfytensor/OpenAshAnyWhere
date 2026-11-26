#!/bin/bash
# chk_loom_env.sh — E: 盘可写性 / GPU 空闲 / 现有侧记忆相关行的精确字节
mkdir -p /mnt/e/OpenASH_loom
if touch /mnt/e/OpenASH_loom/.w 2>/dev/null; then
  rm -f /mnt/e/OpenASH_loom/.w
  echo "E_WRITABLE=yes ($(df -h /mnt/e | tail -1 | awk '{print $4}') free)"
else
  echo "E_WRITABLE=no"
fi
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
echo "TRAIN_PROCS: $(pgrep -fa train_ab | grep -v pgrep | wc -l)"
cd /mnt/f/OpenASH2605
wsl_venv/bin/python - <<'EOF'
src = open('open_ash_v3.py', encoding='utf-8').read().split('\n')
for i, ln in enumerate(src):
    if ('dm_gate' in ln) or ('d_route' in ln) or ('dm_route' in ln) or ("d_q = " in ln) or ("d_q1" in ln):
        print('%4d %r' % (i + 1, ln))
EOF
