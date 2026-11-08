# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Microbenchmark: chunked decay-max scan vs the sequential naive reference."""

import time

import torch

from ops.decay_max_scan.chunk import chunk_decay_max_scan
from ops.decay_max_scan.naive import naive_decay_max_scan


def bench(fn, x, v, b, iters=20):
    for _ in range(3):
        fn(x=x, decay_v=v, decay_b=b)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn(x=x, decay_v=v, decay_b=b)
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


if __name__ == '__main__':
    for B, T, D, H in [(8, 512, 96, 8), (8, 2048, 96, 8), (32, 384, 96, 8)]:
        x = torch.randn(B, T, D, H, device='cuda')
        v = torch.randn(D, device='cuda') * 0.05
        b = torch.randn(D, device='cuda') * 0.05
        tc = bench(chunk_decay_max_scan, x, v, b)
        tn = bench(naive_decay_max_scan, x, v, b)
        tok = B * T
        print(f'B{B} T{T} D{D} H{H}: chunk {tc*1e3:7.2f} ms ({tok/tc/1e6:6.2f} Mtok/s) | '
              f'naive {tn*1e3:7.2f} ms ({tok/tn/1e6:6.2f} Mtok/s) | speedup {tn/tc:5.1f}x')
