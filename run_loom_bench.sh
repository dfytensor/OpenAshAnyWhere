#!/bin/bash
# run_loom_bench.sh — dml/dmq（E: 盘）对 dms@40k / GDN 的官方四探针 + 激活度检查
cd /mnt/f/OpenASH2605
export CKPT_ROOT=/mnt/e/OpenASH_loom
wsl_venv/bin/python -u bench_longrange.py > probe_loom_bench.out 2>&1
echo "bench exit=$?"
echo "config-drift lines: $(grep -c 'CONFIG DRIFT' probe_loom_bench.out)"
grep -F "[load " probe_loom_bench.out
grep -F "结果已存" probe_loom_bench.out
echo "=== activation ==="
bash run_py.sh loom_activation.py
