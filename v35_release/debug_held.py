# -*- coding: utf-8 -*-
"""调试：留出段内容 + 兜底字符分布"""
import sys
from collections import Counter
sys.path.insert(0, r'F:\OpenASH2605')
sys.stdout.reconfigure(encoding='utf-8')
import new_openash_voc as M

_, _, held = M.load_ranges(n_mine=30000, n_freq=120000, n_held=4000)
print(f"留出 {len(held)} 条")
for i, t in enumerate(held[:4]):
    print(f"--- [{i}] {t[:120]!r}")

sents = M.to_sents(held)
zh = [s for s in sents if sum(1 for c in s if ord(c) < 128) / len(s) < 0.3]
en = [s for s in sents if sum(1 for c in s if ord(c) < 128) / len(s) > 0.7]
mid = [s for s in sents if s not in zh and s not in en]
print(f"\nzh {len(zh)} 句 / en {len(en)} 句 / mid {len(mid)} 句")
print("en 样例:", repr(en[0][:80]) if en else '-')
print("en 样例:", repr(en[1][:80]) if len(en) > 1 else '-')
print("zh 样例:", repr(zh[0][:80]) if zh else '-')

# 兜底字符分布：留出段字符 vs 频次段字符表
freq_texts = M.load_ranges(n_mine=30000, n_freq=120000, n_held=4000)[1]
fchars = Counter()
for t in freq_texts[:20000]:
    fchars.update(c for c in t if ord(c) > 127)
held_chars = Counter()
for s in sents:
    held_chars.update(c for c in s if ord(c) > 127)
miss = {c: n for c, n in held_chars.items() if c not in fchars}
tot_miss = sum(miss.values())
tot_held = sum(held_chars.values())
print(f"\n频次段(2万条采样)非ASCII字符 {len(fchars)} 种；"
      f"留出段非ASCII字符 {len(held_chars)} 种")
print(f"留出段字符不在频次段的: {len(miss)} 种 / {tot_miss} 次"
      f"（占留出段非ASCII字符 {tot_miss/max(1,tot_held):.1%}）")
print("缺失TOP30:", ' '.join(f"{c}×{n}" for c, n in
      sorted(miss.items(), key=lambda x: -x[1])[:30]))
