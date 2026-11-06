"""SFT 数据预处理：sft_t2t_mini.jsonl -> 对话流拼接 token 缓存。

模板：用户：{u}\n助手：{a}\n （多轮循环，jieba 词表可直接编码）
流式拼接 -> 384+1 窗口（与预训练同构；loss 全序列，简化第一版）。
采样 N 条控制规模。
"""
import json
import sys

import torch

sys.path.insert(0, '.')
from open_ash_voc import OpenASHVoc
from config import agent_voc_path

SRC = r'minimind_data\sft_t2t_mini.jsonl'
OUT = r'minimind_data\sft_cached_60000_384.pt'
N_SAMPLES = 60000
WIN = 385          # L + 1

voc = OpenASHVoc(agent_voc_path=agent_voc_path)

lines = []
with open(SRC, encoding='utf-8') as f:
    for line in f:
        lines.append(line)
        if len(lines) >= N_SAMPLES:
            break
print(f'sampled {len(lines)} conversations')

stream = []
tok_count = 0
skipped = 0
for idx, line in enumerate(lines):
    try:
        d = json.loads(line)
    except Exception:
        skipped += 1
        continue
    conv = d.get('conversations')
    if not conv:
        skipped += 1
        continue
    text = ''
    for turn in conv:
        role = turn.get('role')
        content = (turn.get('content') or '').strip()
        if not content:
            continue
        if role == 'user':
            text += f'用户：{content}\n'
        elif role == 'assistant':
            text += f'助手：{content}\n'
    if len(text) < 8:
        skipped += 1
        continue
    ids = voc.encode(text)
    stream.extend(ids)
    if (idx + 1) % 10000 == 0:
        print(f'  {idx + 1}/{len(lines)} convs | tokens so far {len(stream):,}')

print(f'total tokens: {len(stream):,} | skipped: {skipped}')
t = torch.tensor(stream, dtype=torch.long)
windows = t.unfold(0, WIN, 1)
torch.save(windows.unsqueeze(0).chunk(1), OUT)
print(f'windows: {windows.shape[0]:,} -> {OUT}')
