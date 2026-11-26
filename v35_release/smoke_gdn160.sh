#!/bin/bash
cd /mnt/f/OpenASH2605
rm -rf /tmp/gdn_smoke
wsl_venv/bin/python -u train_gdn160k.py --steps 101 --warmup 100 --batch_size 16 --eval_every 100 --data minimind_data/pretrain_cached_1270238_384.pt --out /tmp/gdn_smoke > /tmp/gdn_smoke.log 2>&1
echo "exit=$?"
grep -E "step|val|done" /tmp/gdn_smoke.log | tail -5
grep -iE "error|traceback" /tmp/gdn_smoke.log | tail -3
