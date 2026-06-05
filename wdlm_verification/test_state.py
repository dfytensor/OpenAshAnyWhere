import torch, sys
sys.path.insert(0,'F:/OpenASH2605/wdlm_verification')
from wdlm_neural import WaveDynamicsLanguageModel as WN

V=5000
m=WN(V,128,2).cuda().eval()
x=torch.arange(10,18).view(1,8).cuda()

# Full sequence
o_full, _ = m(x)

# Incremental with state
o_inc = []
state = None
for i in range(8):
    xi = x[:, i:i+1]
    oi, state = m(xi, state)
    o_inc.append(oi[:, -1, :])
o_inc = torch.stack(o_inc, 1)

diff = (o_full - o_inc).abs().max().item()
print(f'max diff: {diff:.2e}')
print('state OK' if diff < 1e-3 else 'state FAIL')
