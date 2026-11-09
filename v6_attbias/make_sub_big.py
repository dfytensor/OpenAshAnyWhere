# -*- coding: utf-8 -*-
"""解析大缓存前 40 万样本 → 85M token 子词流 (top-2048)"""
import sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'F:/\u5938\u514b/token_period_package/v5_minimind')
import core5 as C

NS = 400_000
V = 2048
data = C.load_pt(r'F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt')
print('loaded', len(data), 'samples')
sub = [np.asarray(x, dtype=np.int64).ravel() for x in data[:NS]]
alltr = np.concatenate(sub)
print('tokens:', len(alltr))
cnt = np.bincount(alltr)
top = np.argsort(-cnt)[:V - 1]
remap = -np.ones(cnt.size, dtype=np.int64); remap[top] = np.arange(1, V)
print(f'覆盖率(top-{V-1}): {cnt[top].sum() / cnt.sum():.3f}')

def enc(arr):
    r = remap[np.minimum(arr, cnt.size - 1)]
    r[arr >= cnt.size] = 0
    r[r < 0] = 0
    return r.astype(np.int32)

stream = enc(alltr)
np.save(r'F:\OpenASH2605\v6_attbias\sub_stream_big.npy', stream)
print('saved sub_stream_big.npy', stream.shape)
