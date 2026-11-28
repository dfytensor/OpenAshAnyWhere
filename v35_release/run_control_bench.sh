#!/bin/bash
# run_control_bench.sh — 协议对照 + 两臂 + dms@40k + GDN 的官方四探针（对照 checkpoint 在 E: 盘）
cd /mnt/f/OpenASH2605
export CKPT_ROOT=/mnt/e/OpenASH_loom
wsl_venv/bin/python -u bench_longrange.py > probe_control_bench.out 2>&1
echo "bench exit=$?"
echo "config-drift lines: $(grep -c 'CONFIG DRIFT' probe_control_bench.out)"
grep -F "[load " probe_control_bench.out
grep -F "结果已存" probe_control_bench.out
