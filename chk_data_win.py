# -*- coding: utf-8 -*-
"""chk_data_win.py — Windows 侧直接读文件，判定损坏是真（磁盘）还是假（9p 层）"""
import zipfile
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
p = r'F:\OpenASH2605\minimind_data\pretrain_cached_1270238_384.pt'
z = zipfile.ZipFile(p)
infos = z.infolist()
print('entries:', len(infos))
bad_prefix = 'pretrain_cached_127'
# 找出名字异常的条目（含空格/控制字符）
odd = [i for i in infos if ' ' in i.filename or any(ord(c) < 32 for c in i.filename)]
print('odd-named entries:', len(odd), [repr(i.filename) for i in odd[:5]])
# 正常名下的 386255
tgt = [i for i in infos if i.filename.endswith('/data/386255')]
print('normal-named /data/386255:', [(repr(i.filename), i.file_size, i.header_offset) for i in tgt])
for i in odd[:3]:
    print('odd entry:', repr(i.filename), 'size', i.file_size, 'offset', i.header_offset, 'crc', hex(i.CRC))

# 重试读该记录 3 次（确定性 = 磁盘损坏；变化 = 瞬态）
import zlib
for attempt in range(3):
    try:
        for i in odd[:1] + tgt[:1]:
            with z.open(i) as f:
                data = f.read()
            print('attempt', attempt, repr(i.filename), 'read OK', len(data), 'bytes')
    except Exception as e:
        print('attempt', attempt, 'ERROR:', repr(e)[:200])

# 只校验这一条附近（header_offset 附近 4KB 原始字节）
if odd:
    i = odd[0]
    with open(p, 'rb') as f:
        f.seek(i.header_offset)
        raw = f.read(120)
    print('raw @header_offset:', raw[:100])
