# -*- coding: utf-8 -*-
"""crc_fix_dme40k.py — 用 CRC32 线性性定位单比特翻转并精确复原 dme@40k 的 o_read
原理：CRC(M ^ E) = CRC(M) ^ L(E)，L 为无初值/无终值的线性映射。
     D = CRC_expected ^ CRC_read；对每个字节位置 k、每个比特 b 计算 L(单比特@k,b)，匹配 D。
"""
import os
import struct
import sys
import zipfile
import zlib

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
p = r'F:\OpenASH2605\full_v53dme_40k\model.pt'
z = zipfile.ZipFile(p)
i = [x for x in z.infolist() if x.filename == 'model/data/158'][0]
L = i.file_size
with open(p, 'rb') as f:
    f.seek(i.header_offset)
    hdr = f.read(30)
    nlen, elen = struct.unpack('<HH', hdr[26:30])
    payload_start = i.header_offset + 30 + nlen + elen
    f.seek(payload_start)
    raw = f.read(L)
crc_read = zlib.crc32(raw) & 0xffffffff
D = np.uint32(i.CRC ^ crc_read)
print('expected %08x read %08x  diff %08x  (L=%d bytes)' % (i.CRC, crc_read, int(D), L))

# CRC32 表（标准 0xEDB88320 反射多项式）
tbl = np.zeros(256, dtype=np.uint32)
for n in range(256):
    c = np.uint32(n)
    for _ in range(8):
        c = np.uint32((c >> 1) ^ (0xEDB88320 if (c & 1) else 0))
    tbl[n] = c
bits = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.int64)

# S[v] = L(message: byte v at position k, zeros after) ；从最后一字节向前迭代
S = tbl.copy()                       # k = L-1
hits = []
for k in range(L - 1, -1, -1):
    m = S[bits] == D
    if m.any():
        for b in np.nonzero(m)[0]:
            hits.append((k, int(b)))
        if len(hits) >= 4:
            break
    if k > 0:
        S = (S >> np.uint32(8)) ^ tbl[S & np.uint32(0xff)]
print('single-bit candidates (byte_offset, bit):', hits)

if len(hits) == 1:
    k, b = hits[0]
    off = payload_start + k
    with open(p, 'r+b') as f:
        f.seek(off)
        old = f.read(1)
        f.seek(off)
        f.write(bytes([old[0] ^ (1 << b)]))
        f.flush()
        os.fsync(f.fileno())
    print('patched byte @%d: %02x -> %02x' % (off, old[0], old[0] ^ (1 << b)))
    z2 = zipfile.ZipFile(p)
    print('testzip after patch (None = all OK):', z2.testzip())
else:
    print('not a unique single-bit error -> multi-bit/burst corruption, not repairable by this method')
