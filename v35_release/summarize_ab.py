"""汇总 A/B 训练结果: 各 arch 的最终 val PPL / 速度 / 参数。"""
import json
import math
import os

OUTS = [('v3.0 pure', 'ab_v30'), ('v3.1 decay', 'ab_v31'), ('v3.2 modern', 'ab_v32'),
        ('v3.3 layered-slow', 'ab_v33'), ('v3.4 +kv-channel', 'ab_v34'),
        ('tf baseline', 'ab_tf')]

print(f'{"arch":20s} {"params":>12s} {"val PPL":>9s} {"val loss":>9s} {"ms/step":>8s}')
print('-' * 62)
for tag, d in OUTS:
    p = os.path.join('.', d, 'train_log.json')
    if not os.path.exists(p):
        print(f'{tag:20s} {"(missing)":>12s}')
        continue
    log = json.load(open(p, encoding='utf-8'))
    steps = [s for s in log['log'] if 'val_loss' in s]
    last = steps[-1]
    ms = log['log'][-1].get('ms_per_step', 0)
    print(f'{tag:20s} {log["params"]:>12,} {math.exp(min(last["val_loss"],20)):>9.2f} '
          f'{last["val_loss"]:>9.4f} {ms:>8.0f}')
