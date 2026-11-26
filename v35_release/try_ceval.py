"""尝试下载 C-Eval dev 集（260 题含答案，52 科目）。多镜像/路径逐个尝试。"""
import io
import os
import sys
import zipfile

import urllib.request

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
OUT_DIR = r'F:\OpenASH2605\ceval_data'
os.makedirs(OUT_DIR, exist_ok=True)

URLS = [
    'https://hf-mirror.com/datasets/ceval/ceval-exam/resolve/main/dev.zip',
    'https://hf-mirror.com/datasets/ceval/ceval-exam/resolve/main/data/dev.zip',
    'https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/dev.zip',
    'https://hf-mirror.com/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip',
    'https://hf-mirror.com/datasets/ceval/ceval-exam/resolve/main/data.zip',
]

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for url in URLS:
    print('try:', url)
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        print(f'  got {len(data):,} bytes, head={data[:4]!r}')
        if data[:2] == b'PK':
            zf = zipfile.ZipFile(io.BytesIO(data))
            names = zf.namelist()
            print('  zip entries:', names[:10], f'... total {len(names)}')
            zf.extractall(OUT_DIR)
            print('extracted ->', OUT_DIR)
            sys.exit(0)
        else:
            print('  not a zip (may be html/error), content head:',
                  data[:200].decode('utf-8', 'replace'))
    except Exception as e:
        print('  fail:', repr(e)[:120])

print('\nALL URLS FAILED — C-Eval 下载不可达')
