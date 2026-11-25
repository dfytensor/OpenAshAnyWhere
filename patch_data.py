# -*- coding: utf-8 -*-
"""patch_data.py — 外科修复数据缓存 zip 中央目录的单比特损坏（一个字节）
损坏：中央目录记录 data/386255 的名字 'pretrain_cached_127 238_384/...'（0x30→0x20）
依据：同一条记录的本地文件头名字为正确的 'pretrain_cached_1270238_384/data/386255'
"""
import os
import sys
import zipfile

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
p = r'F:\OpenASH2605\minimind_data\pretrain_cached_1270238_384.pt'
bad = b'pretrain_cached_127 238_384/data/386255'
size = os.path.getsize(p)
st0 = os.stat(p)
print('size:', size, 'mtime before:', st0.st_mtime)

tail = 300 * 1024 * 1024
start = max(0, size - tail)
with open(p, 'rb') as f:
    f.seek(start)
    data = f.read()
idx = data.find(bad)
assert idx != -1, 'corrupt name not found in tail'
assert data.find(bad, idx + 1) == -1, 'corrupt name found more than once'
abs_off = start + idx
flip = abs_off + len(b'pretrain_cached_127')
print('corrupt central-dir record @', abs_off, '| flipped byte @', flip)
with open(p, 'rb') as f:
    f.seek(flip)
    b0 = f.read(1)
print('byte before:', b0)
assert b0 == b' ', 'unexpected byte, abort'
with open(p, 'r+b') as f:
    f.seek(flip)
    n = f.write(b'0')
    f.flush()
    os.fsync(f.fileno())
print('patched bytes:', n)

# 全量校验
z = zipfile.ZipFile(p)
infos = z.infolist()
odd = [i for i in infos if ' ' in i.filename or any(ord(c) < 32 for c in i.filename)]
print('entries:', len(infos), '| odd-named after patch:', len(odd))
res = z.testzip()
print('testzip (None = all CRCs OK):', res)
