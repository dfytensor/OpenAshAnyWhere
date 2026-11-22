#!/bin/bash
cd /mnt/f/OpenASH2605
wsl_venv/bin/python -u bench_longrange.py > probe_80k_bench.out 2>&1
grep -v -i "warning\|include\|define" probe_80k_bench.out | tail -2
wsl_venv/bin/python -u -c "
import sys, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
r = json.load(open('longrange_results.json', encoding='utf-8'))
dg = r['D_gamma']
for model, layers in dg.items():
    if '80k' in model:
        for i, rec in enumerate(layers):
            print('%s L%d gamma %.4f hl %.1f tok' % (model, i, rec[0], rec[1]))
"
