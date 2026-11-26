#!/bin/bash
# chain_loom.sh — Loom 读端口 A/B（v53dml → v53dmq），顺序执行；checkpoint 与日志写入 E: 盘（磁盘 0）
cd /mnt/f/OpenASH2605
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/mnt/e/OpenASH_loom
mkdir -p "$ROOT"
echo "chain start $(date '+%F %T')" > "$ROOT/chain_status.txt"
for arch in v53dml v53dmq; do
  out="$ROOT/full_${arch}_40k"
  mkdir -p "$out"
  wsl_venv/bin/python -u train_ab.py --arch "$arch" --steps 40000 --warmup 800 --eval_every 4000 \
      --anchor_lambda 3e-6 --seq_len 384 --batch_size 16 --accum 2 \
      --data minimind_data/pretrain_cached_1270238_384.pt --out "$out" \
      > "$ROOT/full_${arch}_40k_wsl.log" 2>&1
  echo "$arch exit=$? $(date '+%F %T')" >> "$ROOT/chain_status.txt"
done
echo "chain done $(date '+%F %T')" >> "$ROOT/chain_status.txt"
