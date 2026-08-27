import sys, time
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
sys.path.insert(0, r"F:\OpenASH2605")
import torch, torch.nn.functional as F
from convash30 import ConvASH30, VOCAB

torch.manual_seed(0)
m = ConvASH30().to('cuda')
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
B, S = 32, 256
x = torch.randint(2, VOCAB, (B, S), device='cuda')
y = x.clone()


def step():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out, _ = m(x)
        loss = F.cross_entropy(out.reshape(-1, VOCAB), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


for _ in range(2):
    step()
torch.cuda.synchronize()
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
    step()
torch.cuda.synchronize()
evs = {}
for e in prof.key_averages():
    if e.self_device_time_total > 0:
        evs[e.key] = e.self_device_time_total
tot = sum(evs.values())
print('total cuda: %.1f ms' % (tot / 1000))
for k, v in sorted(evs.items(), key=lambda kv: -kv[1])[:22]:
    print('  %8.1f ms (%4.1f%%)  %s' % (v / 1000, v / tot * 100, k[:90]))
