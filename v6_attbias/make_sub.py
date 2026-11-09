# -*- coding: utf-8 -*-
"""解析 1.27M 样本大缓存 → 截取前 110k 样本 → remap 到 top-2048 token → 保存 int32 流"""
import sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:\夸克\token_period_package\v5_minimind')
import core5 as C

NS = 110_000
V = 2048
data = C.load_pt(r'F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt')
print('loaded', len(data), 'samples; first:', data[0].dtype, data[0].shape)
lens = np.array([len(x) for x in data[:NS]])
print(f'前{NS}样本长度 mean {lens.mean():.0f} total {lens.sum():,}')

sub = [np.asarray(x, dtype=np.int64).ravel() for x in data[:NS]]
alltr = np.concatenate(sub)
cnt = np.bincount(alltr)
top = np.argsort(-cnt)[:V - 1]
remap = -np.ones(cnt.size, dtype=np.int64); remap[top] = np.arange(1, V)
print(f'vocab max id {cnt.size - 1}, top-{V-1} 覆盖率 {cnt[top].sum() / cnt.sum():.3f}')

def enc(arr):
    r = remap[np.minimum(arr, cnt.size - 1)]
    r[arr >= cnt.size] = 0
    r[r < 0] = 0
    return r.astype(np.int32)

stream = enc(alltr)
np.save(r'F:\OpenASH2605\v6_attbias\sub_stream.npy', stream)
np.save(r'F:\OpenASH2605\v6_attbias\sub_lens.npy', lens.astype(np.int64))
print('saved sub_stream.npy', stream.shape, stream.dtype)
