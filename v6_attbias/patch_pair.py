# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')
p = r'F:\OpenASH2605\v6_attbias\pair6.py'
raw = open(p, 'rb').read().decode('utf-8')
# 去掉 r 前缀使 \u 转义生效
old = chr(114) + chr(39) + 'F:/\u5938\u514b/token'
new = chr(39) + 'F:/\u5938\u514b/token'
if old in raw:
    raw = raw.replace(old, new)
    open(p, 'wb').write(raw.encode('utf-8'))
    print('prefix removed')
else:
    print('already ok')
# 验证
sys.path.insert(0, 'F:/\u5938\u514b/token_period_package/v5_minimind')
import core5
print('core5 import OK:', core5.__file__)
