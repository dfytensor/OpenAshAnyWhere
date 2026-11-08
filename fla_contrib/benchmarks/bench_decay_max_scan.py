# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Microbenchmark: Triton chunked decay-max scan vs torch semiring scan vs
the sequential naive reference."""

import time

import torch

from ops.decay_max_scan.chunk import TRITON_BACKEND, chunk_decay_max_scan
from ops.decay_max_scan.chunk import chunk_decay_max_scan as _chunk_entry
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
    # torch semiring path (what chunk_decay_max_scan falls back to off-CUDA)
    from ops.decay_max_scan.chunk import _scan_chunk

    def torch_scan(x, decay_v=None, decay_b=None, initial_state=None, cu_seqlens=None):
        import torch.nn.functional as F
        xf = x.float()
        pre = xf * decay_v.view(1, 1, -1, 1) + decay_b.view(1, 1, -1, 1)
        g = torch.exp(-F.softplus(pre))
        s = torch.full((x.shape[0], x.shape[2], x.shape[3]), -1e30,
                       dtype=torch.float32, device=x.device)
        e = torch.empty_like(xf)
        for ck in range(0, x.shape[1], 128):
            e[:, ck:ck + 128], s = _scan_chunk(xf[:, ck:ck + 128], g[:, ck:ck + 128], s)
        return e, s

    for B, T, D, H in [(8, 512, 96, 8), (8, 2048, 96, 8), (32, 384, 96, 8)]:
        x = torch.randn(B, T, D, H, device='cuda')
        v = torch.randn(D, device='cuda') * 0.05
        b = torch.randn(D, device='cuda') * 0.05
        tok = B * T
        parts = []
        if TRITON_BACKEND:
            tt = bench(_chunk_entry, x, v, b)
            parts.append(f'triton {tt*1e3:7.2f} ms ({tok/tt/1e6:5.2f} Mtok/s)')
        tc = bench(torch_scan, x, v, b)
        parts.append(f'torch {tc*1e3:7.2f} ms ({tok/tc/1e6:5.2f} Mtok/s)')
        tn = bench(naive_decay_max_scan, x, v, b)
        parts.append(f'naive {tn*1e3:7.2f} ms ({tok/tn/1e6:5.2f} Mtok/s)')
        print(f'B{B} T{T} D{D} H{H}: ' + ' | '.join(parts))

