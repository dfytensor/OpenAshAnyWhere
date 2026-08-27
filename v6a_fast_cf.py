"""V6a Fast CopyFirst 测试"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
import sys; sys.path.insert(0, 'F:/OpenASH2605')
from frsm_v6a_fast import FRSM_V6_Fast

device = torch.device("cuda")
V=32; E=0; I=1; H=128

def mf(bs,nl):
    t=torch.randint(2,V,(bs,)); n=torch.randint(2,V,(bs,nl))
    e=torch.full((bs,1),E,dtype=torch.long)
    x=torch.cat([t.unsqueeze(1),n,e],1); y=torch.full_like(x,I); y[:,-1]=t
    return x,y

torch.manual_seed(42)
m = FRSM_V6_Fast(vocab_size=V, d_model=H, num_scales=4).to(device)
n = sum(p.numel() for p in m.parameters())
print(f"V6a-Fast: {n:,} params")

opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 2500)
m.train(); best = float('inf'); t0 = time.time()
for st in range(1, 2501):
    x, y = mf(64, random.randint(4, 64)); x, y = x.to(device), y.to(device)
    log, _ = m(x, return_state=True); loss = F.cross_entropy(log[:, -1, :], y[:, -1], ignore_index=I)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
    if loss.item() < best: best = loss.item()
    if st % 500 == 0: print(f"step{st:5d} best={best:.5f} {time.time()-t0:.0f}s",flush=True)
print(f"best={best:.5f} {time.time()-t0:.0f}s",flush=True)

m.eval()
print("Dist | Acc")
for d in [4, 64, 256, 1024, 4096, 8192, 16384, 32768, 65536]:
    eb = 64 if d <= 4096 else 8; c = 0; total = 0
    for _ in range(4 if d <= 4096 else 2):
        x, y = mf(eb, d); x, y = x.to(device), y.to(device)
        log, _ = m(x, return_state=True); c += (log[:, -1, :].argmax(-1) == y[:, -1]).sum().item(); total += eb
    print(f"{d:5d} | {c/total*100:.1f}%",flush=True)
print("Done.")
