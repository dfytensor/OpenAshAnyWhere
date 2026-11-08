import sys
sys.path.insert(0, 'fla_contrib')
from ops.decay_max_scan.chunk import TRITON_BACKEND
import torch, triton
print('TRITON_BACKEND:', TRITON_BACKEND, '| triton', triton.__version__, '| torch', torch.__version__)
assert TRITON_BACKEND, 'Triton backend did not engage on Linux!'
