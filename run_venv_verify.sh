#!/bin/bash
cd /mnt/f/OpenASH2605
wsl_venv/bin/python -u venv_verify.py > venv_verify.out 2>&1
echo "exit=$?"
tail -30 venv_verify.out
