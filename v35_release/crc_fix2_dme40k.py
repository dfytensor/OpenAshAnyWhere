# -*- coding: utf-8 -*-
"""crc_fix2_dme40k.py — 自检算法 + 放宽到单字节搜索"""
import os
import struct
import sys
import zipfile
import zlib

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
tbl = np.zeros(256, dtype=np.uint32)
for n in range(256):
    c = np.uint32(n)
    for _ in range(8):
        c = np.uint32((c >> 1) ^ (0xEDB88320 if (c & 1) else 0))
    tbl[n] = c


def search(D, L, max_hits=4):
    """返回所有 (byte_offset, xor_value) 使 L(单字节异或 xor_value @offset) == D"""
    S = tbl.copy()
    hits = []
    for k in range(L - 1, -1, -1):
        w = np.nonzero(S == np.uint32(D))[0]
        if w.size:
            hits.extend((k, int(x)) for x in w)
            if len(hits) >= max_hits:
                break
        if k > 0:
            S = (S >> np.uint32(8)) ^ tbl[S & np.uint32(0xff)]
    return hits


# ---- 自检：合成 buffer，翻转 1 个比特 / 改写 1 个字节 ----
rng = np.random.default_rng(0)
buf = bytearray(rng.integers(0, 256, 5000, dtype=np.uint8).tobytes())
c0 = zlib.crc32(bytes(buf)) & 0xffffffff
b1 = bytearray(buf)
b1[1234] ^= 0x10
h = search(c0 ^ (zlib.crc32(bytes(b1)) & 0xffffffff), 5000)
print('self-test single bit @1234 ^0x10 ->', h)
b2 = bytearray(buf)
b2[4711] ^= 0xA5
h2 = search(c0 ^ (zlib.crc32(bytes(b2)) & 0xffffffff), 5000)
print('self-test single byte @4711 ^0xA5 ->', h2)

# ---- 实搜：dme@40k o_read ----
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
D = i.CRC ^ (zlib.crc32(raw) & 0xffffffff)
print('real: diff %08x, searching single-byte patterns over %d bytes...' % (D, L))
hh = search(D, L)
print('real single-byte candidates:', hh)
if len(hh) == 1:
    k, v = hh[0]
    off = payload_start + k
    with open(p, 'r+b') as f:
        f.seek(off)
        old = f.read(1)
        f.seek(off)
        f.write(bytes([old[0] ^ v]))
        f.flush()
        os.fsync(f.fileno())
    print('patched byte @%d: %02x -> %02x' % (off, old[0], old[0] ^ v))
    print('testzip after patch (None = all OK):', zipfile.ZipFile(p).testzip())
else:
    print('=> not a single-byte error: multi-byte/burst corruption, unrecoverable without a reference copy')
