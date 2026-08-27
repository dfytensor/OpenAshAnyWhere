"""从原始数据前 120k 行生成代表性过滤训练集 (打乱, 无类别排序偏置).
训练范围: raw 0-120k; 评估范围: raw 150k+; 两者无重叠 -> 无泄漏."""
import sys, json, random
sys.path.insert(0, r'F:\OpenASH2605\frsm37_threeq_60m')
from three_q_filter import classify_three_q
from collections import Counter

random.seed(42)
SRC = r'F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl'
OUT = r'F:\OpenASH2605\frsm37_threeq_60m\filtered_data\pretrain_filt_train120k.jsonl'

lines = open(SRC, encoding='utf-8').readlines()[:120000]
a, b, c = [], [], []
for line in lines:
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except Exception:
        continue
    t = obj.get('text', '')
    if len(t) < 20:
        continue
    cat = classify_three_q(t)['category']
    obj['tqf_cat'] = cat
    {'A': a, 'B': b, 'C': c}[cat].append(obj)

core = len(a) + len(b)
budget = int(core * 0.2 / 0.8)
c = c[:budget]
kept = a + b + c
random.shuffle(kept)
with open(OUT, 'w', encoding='utf-8') as f:
    for o in kept:
        f.write(json.dumps(o, ensure_ascii=False) + '\n')
cc = Counter(o['tqf_cat'] for o in kept)
print(f'raw 0-120k filtered: A={cc["A"]} B={cc["B"]} C={cc["C"]} total={len(kept)} (shuffled)')
