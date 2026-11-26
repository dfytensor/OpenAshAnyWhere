# -*- coding: utf-8 -*-
"""measure_dme2.py — 等规模裁剪组合精确测量（直接实例化 OpenASH_V3）"""
import sys
import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
from open_ash_v3 import OpenASH_V3

VOC, HID, HEADS = 23004, 768, 8
BASE = dict(use_fused=True, use_decay=True, pre_norm=True, ffn_act='silu',
            decay_init=(0.0001, 0.35), decay_v_std=0.04,
            kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
            kv_dim=256, kv_backend='delta_fla', kv_beta_init=4.0,
            short_conv=True, moe_ffn=True, moe_experts=2, state_split=True,
            mem_heads=3, mem_force=True, ffn_hidden=2048, kv_out_gdn=True,
            kv_beta_tok=True, em_factor_r=192, use_read=False,
            dm_state=True, kv_decay_spread=True,
            kv_decay_spread_lo=0.90, kv_decay_spread_hi=0.999)

COMBOS = [
    ('dm (base)', {}),
    ('dme_A: ffn1792', dict(ffn_hidden=1792)),
    ('dme_B: ffn1856', dict(ffn_hidden=1856)),
    ('dme_C: ffn1856+em160', dict(ffn_hidden=1856, em_factor_r=160)),
    ('dme_D: ffn1920+em144', dict(ffn_hidden=1920, em_factor_r=144)),
    ('dme_E: ffn1856+kv224', dict(ffn_hidden=1856, kv_dim=224)),
    ('dme_F: ffn1792+kv240', dict(ffn_hidden=1792, kv_dim=240)),
]

for tag, over in COMBOS:
    torch.manual_seed(2024)
    kw = dict(BASE)
    kw.update(over)
    m = OpenASH_V3(VOC, HID, HEADS, 7, max_seq_len=2048, model_flag='train', **kw)
    tot = sum(p.numel() for p in m.parameters())
    ok = '✓' if tot <= 106.3e6 else '✗'
    print('%-26s %.2fM (GDN %.1f%%) %s' % (tag, tot / 1e6, tot / 106.3e6 * 100, ok))
    del m
