#!/bin/bash
# chain_loom80k.sh — dml@80k → dmq@80k（同 token 终局：各 983M tokens，bs16×累积 2；checkpoint 与日志写 E: 盘）
cd /mnt/f/OpenASH2605
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/mnt/e/OpenASH_loom
mkdir -p "$ROOT"
echo "chain80 start $(date '+%F %T')" > "$ROOT/chain80_status.txt"
for arch in v53dml v53dmq; do
  out="$ROOT/full_${arch}_80k"
  mkdir -p "$out"
  wsl_venv/bin/python -u train_ab.py --arch "$arch" --steps 80000 --warmup 800 --eval_every 8000 \
      --anchor_lambda 3e-6 --seq_len 384 --batch_size 16 --accum 2 \
      --data minimind_data/pretrain_cached_1270238_384.pt --out "$out" \
      > "$ROOT/full_${arch}_80k_wsl.log" 2>&1
  echo "$arch exit=$? $(date '+%F %T')" >> "$ROOT/chain80_status.txt"
done
echo "chain80 done $(date '+%F %T')" >> "$ROOT/chain80_status.txt"
