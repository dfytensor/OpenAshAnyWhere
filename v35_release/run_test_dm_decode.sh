#!/bin/bash
cd /mnt/f/OpenASH2605
wsl_venv/bin/python -u test_dm_decode.py > test_dm_decode.out 2>&1
echo "exit=$?"
# 去掉编译器噪声（C 预处理器警告），保留全部测试输出
grep -v -i -E "warning|include|define|note:|__triton|POSIX|in file|from /|pyconfig|features.h" test_dm_decode.out | tail -48
