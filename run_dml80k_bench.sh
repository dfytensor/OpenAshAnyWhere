#!/bin/bash
# run_dml80k_bench.sh — dml@80k（E: 盘）vs GDN@160k / dms@80k / dme@80k 的官方四探针
cd /mnt/f/OpenASH2605
export CKPT_ROOT=/mnt/e/OpenASH_loom
wsl_venv/bin/python -u bench_longrange.py > probe_dml80k_bench.out 2>&1
echo "bench exit=$?"
echo "config-drift lines: $(grep -c 'CONFIG DRIFT' probe_dml80k_bench.out)"
grep -F "[load " probe_dml80k_bench.out
grep -F "结果已存" probe_dml80k_bench.out
tail -3 probe_dml80k_bench.out | grep -i -E "error|memory" | head -2
