#!/bin/bash
cd /mnt/f/OpenASH2605
wsl_venv/bin/python -u bench_longrange.py > probe_dms80k_bench.out 2>&1
echo "bench exit=$?"
echo "config-drift lines: $(grep -c 'CONFIG DRIFT' probe_dms80k_bench.out)"
grep -F "[load " probe_dms80k_bench.out
grep -F "结果已存" probe_dms80k_bench.out
