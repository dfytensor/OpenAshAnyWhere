# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')
p = r'F:\OpenASH2605\v6_attbias\analyze_lc.py'
raw = open(p, 'rb').read().decode('utf-8')
old = 'import core5 as C'
new = chr(115) + 'ys.path.insert(0, ' + chr(39) + 'F:/\u5938\u514b/token_period_package/v5_minimind' + chr(39) + ')\nimport core5 as C'
assert old in raw
raw = raw.replace(old, new, 1)
open(p, 'wb').write(raw.encode('utf-8'))
print('patched')
