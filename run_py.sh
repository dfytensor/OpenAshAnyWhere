#!/bin/bash
# run_py.sh <script.py> — 在 WSL venv 里跑一个脚本并滤掉编译器噪声
cd /mnt/f/OpenASH2605
wsl_venv/bin/python -u "$1" > "/tmp/$(basename "$1" .py).out" 2>&1
echo "exit=$?"
grep -v -i -E "warning|include|define|note:|__triton|POSIX|in file|from /|pyconfig|features.h" "/tmp/$(basename "$1" .py).out" | tail -40
