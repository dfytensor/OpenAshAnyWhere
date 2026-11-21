# -*- coding: utf-8 -*-
"""measure_dme.py — v53dm 参数解剖 + 等规模裁剪组合测量"""
import sys
import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
from train_ab import build


def count(m, tag):
    tot = sum(p.numel() for p in m.parameters())
    print('%-28s %.2fM (GDN %.1f%%)' % (tag, tot / 1e6, tot / 106.3e6 * 100))
    return tot


torch.manual_seed(2024)
m = build('v53dm')
count(m, 'v53dm (base 111.2M)')

# 模块级解剖
groups = {}
for n, p in m.named_parameters():
    key = n.split('.')[0] + ('.' + n.split('.')[1] if 'decoder_layers' in n and len(n.split('.')) > 3 else '')
    top = n.split('.')[0]
    if top == 'decoder_layers':
        rest = n.split('.')
        sub = rest[2] if len(rest) > 2 else '?'
        if sub == 'self_attention_linear':
            s2 = rest[3] if len(rest) > 3 else 'attn_core'
            key2 = 'attn.' + (s2 if s2.startswith(('dm_', 'kv_', 'W_', 'q_', 'k_')) else 'core')
        elif sub == 'ffn':
            key2 = 'ffn'
        else:
            key2 = sub
        groups[key2] = groups.get(key2, 0) + p.numel()
    else:
        groups[top] = groups.get(top, 0) + p.numel()
for k in sorted(groups, key=groups.get, reverse=True):
    print('  %-24s %8.2fM' % (k, groups[k] / 1e6))

# dm 侧记忆逐层
dm_params = [(n, p.numel()) for n, p in m.named_parameters() if 'dm_' in n]
dm_total = sum(x[1] for x in dm_params)
print('  dm side memory total: %.2fM (per layer %.3fM)' % (dm_total / 1e6, dm_total / 1e6 / 7))
