import sys, time, random
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
sys.path.insert(0, r"F:\OpenASH2605")
import torch, torch.nn.functional as F
from convash30 import ConvASH30, VOCAB

torch.manual_seed(0)
random.seed(0)
m = ConvASH30().to('cuda')
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
seqs = torch.load(r'F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt',
                  map_location='cpu', weights_only=True)[:10000]
B, S = 32, 256


def step(mod):
    xs = []
    for _ in range(B):
        s = seqs[random.randrange(len(seqs))][:S]
        xs.append(F.pad(s, (0, S - s.numel())))
    x = torch.stack(xs).to('cuda')
    y = x.clone(); y[:, :-1] = x[:, 1:]; y[:, -1] = 0
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out, _ = mod(x)
        loss = F.cross_entropy(out[:, :-1].reshape(-1, VOCAB), y[:, :-1].reshape(-1), ignore_index=0)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
    opt.step()
    return loss.item()


t0 = time.time()
for st in range(3):
    print('step %d loss=%.3f (%.1fs)' % (st, step(m), time.time() - t0), flush=True)

print('try torch.compile default...', flush=True)
mc = torch.compile(m, mode='default')
t0 = time.time()
for st in range(2):
    print('compiled step %d loss=%.3f (%.1fs)' % (st, step(mc), time.time() - t0), flush=True)
