# -*- coding: utf-8 -*-
"""fla_sig.py — 查 FLA 两个 kernel 的签名默认值（l2norm 默认会决定解码步是否要归一化 q）"""
import inspect
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from fla.ops.delta_rule import chunk_delta_rule
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

for fn in (chunk_delta_rule, chunk_gated_delta_rule):
    print('===', fn.__name__)
    for name, p in inspect.signature(fn).parameters.items():
        print('   %-28s default=%r' % (name, p.default if p.default is not inspect.Parameter.empty else '<required>'))
import fla
print('fla version:', getattr(fla, '__version__', '?'))
