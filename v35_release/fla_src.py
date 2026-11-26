# -*- coding: utf-8 -*-
"""fla_src.py — 读取安装的 FLA 源码：默认 scale、逐 token 递推语义、状态布局"""
import inspect
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from fla.ops.delta_rule import chunk_delta_rule
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

for fn in (chunk_delta_rule, chunk_gated_delta_rule):
    src = inspect.getsource(fn)
    print('=' * 20, fn.__name__, '(%d lines)' % len(src.splitlines()))
    # 只打印与 scale / 布局 / 递推语义相关的行
    for i, ln in enumerate(src.splitlines()):
        low = ln.lower()
        if any(k in low for k in ('scale', 'shape', 'layout', 'state', 'eigval', 'l2norm', 'sigmoid', 'recurrence', 's_t', 'initial_state')):
            print('%4d %s' % (i + 1, ln[:118]))

print()
print('=' * 20, 'chunk_fwd_h (gated) core: 逐 token 递推')
import fla.ops.gated_delta_rule.chunk_h as ch
src = inspect.getsource(ch)
lines = src.splitlines()
for i, ln in enumerate(lines):
    if 'def chunk_gated_delta_rule_fwd_h' in ln:
        for j in range(i, min(i + 95, len(lines))):
            if lines[j].strip() and not lines[j].strip().startswith(('#', '"""')) or 'S_t' in lines[j]:
                print('%4d %s' % (j + 1, lines[j][:118]))
        break
