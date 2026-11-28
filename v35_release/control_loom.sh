#!/bin/bash
# control_loom.sh — 协议对照：v53dms 原架构 + bs16×累积 2（与 dml/dmq 两臂协议逐项相同，无任何新旋钮）
cd /mnt/f/OpenASH2605
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/mnt/e/OpenASH_loom
out="$ROOT/full_v53dms_40k_bs16x2"
mkdir -p "$out"
echo "control start $(date '+%F %T')" > "$ROOT/control_status.txt"
wsl_venv/bin/python -u train_ab.py --arch v53dms --steps 40000 --warmup 800 --eval_every 4000 \
    --anchor_lambda 3e-6 --seq_len 384 --batch_size 16 --accum 2 \
    --data minimind_data/pretrain_cached_1270238_384.pt --out "$out" \
    > "$ROOT/full_v53dms_40k_bs16x2_wsl.log" 2>&1
echo "control exit=$? $(date '+%F %T')" >> "$ROOT/control_status.txt"
