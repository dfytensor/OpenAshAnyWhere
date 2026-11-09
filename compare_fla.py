"""FLA 基线对决汇总：v3.5 vs Gated DeltaNet vs DeltaNet（大数据等 token 预算）。"""
import json
import math
import os

RUNS = [('v3.5 (ours)', 'big_v35_3k'), ('gdn', 'big_gdn'), ('dn', 'big_dn')]
ROOT = r'F:\OpenASH2605'

print(f'{"model":16s} {"params":>12s} {"bs":>4s} {"steps":>7s} {"tokens":>12s} {"val PPL":>9s}')
print('-' * 60)
for tag, d in RUNS:
    p = os.path.join(ROOT, d, 'train_log.json')
    if not os.path.exists(p):
        print(f'{tag:16s} (missing)')
        continue
    log = json.load(open(p, encoding='utf-8'))
    steps = log['log'][-1]['step']
    bs = 16 if 'gdn' in d or 'dn' in d else 32
    toks = steps * bs * 384
    val = [s for s in log['log'] if 'val_loss' in s][-1]
    ppl = math.exp(min(val['val_loss'], 20))
    n = log.get('params', 0)
    print(f'{tag:16s} {n:>12,} {bs:>4d} {steps:>7,} {toks:>12,} {ppl:>9.2f}')
print('\n协议：等 token 预算 36.9M（FLA 基线 bs16×6000 步；v3.5 bs32×3000 步）；'
      '同 seed、同数据顺序协议、同 lr/warmup 结构')
