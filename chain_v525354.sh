#!/bin/bash
cd /mnt/f/OpenASH2605
wsl_venv/bin/python -u train_ab.py --arch v52 --steps 15000 --warmup 500 --eval_every 2000 --data minimind_data/pretrain_cached_1270238_384.pt --out scale_v52_15k > scale_v52_15k_wsl.log 2>&1
wsl_venv/bin/python -u train_ab.py --arch v53 --steps 15000 --warmup 500 --eval_every 2000 --data minimind_data/pretrain_cached_1270238_384.pt --out scale_v53_15k > scale_v53_15k_wsl.log 2>&1
wsl_venv/bin/python -u train_ab.py --arch v54 --steps 15000 --warmup 500 --eval_every 2000 --data minimind_data/pretrain_cached_1270238_384.pt --out scale_v54_15k > scale_v54_15k_wsl.log 2>&1
echo CHAIN_DONE > chain_525354.done
