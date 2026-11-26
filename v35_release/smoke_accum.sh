#!/bin/bash
cd /mnt/f/OpenASH2605
rm -rf /tmp/accum_smoke
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True wsl_venv/bin/python -u train_ab.py --arch v53dmh --steps 400 --warmup 100 --eval_every 400 --anchor_lambda 3e-6 --seq_len 384 --batch_size 16 --accum 2 --data minimind_data/pretrain_cached_1270238_384.pt --out /tmp/accum_smoke > /tmp/accum_smoke.log 2>&1 &
TP=$!
# 等进入训练步（最多 12 分钟）
for i in $(seq 1 48); do
  sleep 15
  if grep -q "step    1 |" /tmp/accum_smoke.log 2>/dev/null; then break; fi
done
sleep 45
echo "=== mid-run GPU (training + eval peak later) ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
wait $TP
echo "exit=$?"
grep -E "params|step|val|done" /tmp/accum_smoke.log | tail -7
grep -iE "error|traceback|out of memory" /tmp/accum_smoke.log | tail -3
