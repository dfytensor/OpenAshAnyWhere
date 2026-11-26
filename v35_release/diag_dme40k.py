# -*- coding: utf-8 -*-
"""diag_dme40k.py — full_v53dme_40k/model.pt 损坏性质诊断"""
import os
import struct
import sys
import zipfile
import zlib

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
p = r'F:\OpenASH2605\full_v53dme_40k\model.pt'
print('size:', os.path.getsize(p), 'mtime:', os.path.getmtime(p))
z = zipfile.ZipFile(p)
infos = z.infolist()
print('entries:', len(infos))
odd = [i for i in infos if ' ' in i.filename or any(ord(c) < 32 for c in i.filename)]
print('odd-named entries:', len(odd), [(repr(i.filename), i.file_size, i.header_offset) for i in odd[:5]])
tgt = [i for i in infos if i.filename.endswith('/158')]
print('entries ending /158:', [(repr(i.filename), i.file_size, i.header_offset, hex(i.CRC)) for i in tgt])

# 逐成员读 + CRC 校验，只报告坏的（带细节）
bad = []
for i in infos:
    try:
        with z.open(i) as f:
            data = f.read()
        if zlib.crc32(data) & 0xffffffff != i.CRC:
            bad.append((i.filename, 'CRC', len(data)))
    except Exception as e:
        bad.append((i.filename, repr(e)[:140], i.file_size))
print('bad members:', len(bad))
for bname in bad[:10]:
    print('  ', bname)

# 对第一个坏成员看本地头
if bad:
    name = bad[0][0]
    i = [x for x in infos if x.filename == name][0]
    with open(p, 'rb') as f:
        f.seek(i.header_offset)
        hdr = f.read(30)
        sig, ver, flag, comp, mt, md, crc, csz, usz, nlen, elen = struct.unpack('<IHHHHHIIIHH', hdr)
        lname = f.read(nlen)
        extra = f.read(elen)
        dstart = i.header_offset + 30 + nlen + elen
        print('local header: sig=%08x comp=%d crc=%s csz=%d usz=%d' % (sig, comp, hex(crc), csz, usz))
        print('local name :', lname[:80])
        print('central name:', i.filename[:80].encode())
        print('names equal:', lname.decode(errors='replace') == i.filename)
        with open(p, 'rb') as f:
            f.seek(dstart)
            raw = f.read(min(csz, 4096))
        print('payload first bytes:', raw[:16])
        print('payload all-zero in first 4KB:', raw.count(0) == len(raw))
