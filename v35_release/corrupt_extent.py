# -*- coding: utf-8 -*-
"""corrupt_extent.py — dme@40k o_read 损坏程度分析（按中央目录偏移直读原始字节）"""
import os
import struct
import sys
import zipfile

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
p = r'F:\OpenASH2605\full_v53dme_40k\model.pt'
z = zipfile.ZipFile(p)
infos = z.infolist()
i = [x for x in infos if x.filename == 'model/data/158'][0]
print('central: compress_type=%d csz=%d usz=%d crc=%s offset=%d' % (
    i.compress_type, i.compress_size, i.file_size, hex(i.CRC), i.header_offset))
with open(p, 'rb') as f:
    f.seek(i.header_offset)
    hdr = f.read(30)
    nlen, elen = struct.unpack('<HH', hdr[26:30])
    f.seek(i.header_offset + 30 + nlen + elen)
    raw = f.read(i.file_size)
print('read bytes:', len(raw), '| crc of read:', hex(zipfile.zlib.crc32(raw) & 0xffffffff))
a = np.frombuffer(raw, dtype=np.float32)
print('elements:', a.size)
fin = np.isfinite(a)
print('non-finite (NaN/Inf):', int((~fin).sum()))
absa = np.abs(a[fin])
print('finite |x|: median %.4f  mean %.4f  max %.4f' % (np.median(absa), absa.mean(), absa.max()))
print('count |x|>1:', int((absa > 1).sum()), '| |x|>10:', int((absa > 10).sum()), '| exact zeros:', int((a == 0).sum()))
# 损坏是否成块（4KB 扇区）：按 4096 字节块统计异常值
bs = 4096 // 4
nblk = a.size // bs
blocks = a[:nblk * bs].reshape(nblk, bs)
bad_per_blk = ((~np.isfinite(blocks)) | (np.abs(blocks) > 1)).sum(axis=1)
badblk = np.nonzero(bad_per_blk)[0]
print('blocks(4KB):', nblk, '| blocks with anomalies:', len(badblk), '| first/last bad block:', badblk[:5], badblk[-5:] if len(badblk) else '')
if len(badblk):
    b0 = badblk[0]
    print('first bad block sample floats:', blocks[b0, :8])
    print('a good block sample floats   :', blocks[max(0, b0 - 1), :8])
# 对比：同层其他参数的量级
print()
print('对照——相邻存储 data/157、data/159 的量级:')
for nm in ('model/data/157', 'model/data/159'):
    j = [x for x in infos if x.filename == nm][0]
    with open(p, 'rb') as f:
        f.seek(j.header_offset)
        h = f.read(30)
        nl, el = struct.unpack('<HH', h[26:30])
        f.seek(j.header_offset + 30 + nl + el)
        rr = f.read(j.file_size)
    b = np.frombuffer(rr, dtype=np.float32)
    print('  %s elems %d crc_ok=%s |x| median %.4f max %.4f' % (
        nm, b.size, (zipfile.zlib.crc32(rr) & 0xffffffff) == j.CRC, np.median(np.abs(b)), np.abs(b).max()))
