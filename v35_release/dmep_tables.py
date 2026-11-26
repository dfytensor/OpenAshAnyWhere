# -*- coding: utf-8 -*-
"""dmep_tables.py — 激活度检查：位置表学到什么了"""
import sys
import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
sd = torch.load('/mnt/f/OpenASH2605/scale_v53dmep_15k/model.pt', map_location='cpu', weights_only=True)
for L in range(7):
    b = sd['decoder_layers.%d.self_attention_linear.beta_pos_table' % L]
    d = sd['decoder_layers.%d.self_attention_linear.decay_pos_table' % L]
    print('L%d beta |mean| %.4f max %.3f std %.4f | decay |mean| %.4f max %.3f std %.4f' % (
        L, b.abs().mean(), b.abs().max(), b.std(),
        d.abs().mean(), d.abs().max(), d.std()))
# 位置曲线（第 4 层 decay 表，按 64 桶）
d4 = sd['decoder_layers.4.self_attention_linear.decay_pos_table'].mean(dim=1)
buckets = [d4[i*64:(i+1)*64].mean().item() for i in range(32)]
print('L4 decay-table position curve (32 buckets of 64):')
print(' '.join('%+.2f' % v for v in buckets))
b4 = sd['decoder_layers.4.self_attention_linear.beta_pos_table'].mean(dim=1)
buckb = [b4[i*64:(i+1)*64].mean().item() for i in range(32)]
print('L4 beta-table position curve:')
print(' '.join('%+.2f' % v for v in buckb))
