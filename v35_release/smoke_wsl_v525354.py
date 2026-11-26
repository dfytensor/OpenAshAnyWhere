import sys
sys.path.insert(0, '/mnt/f/OpenASH2605')
import torch
from train_ab import build

for arch in ('v52', 'v53', 'v54'):
    torch.manual_seed(2024)
    m = build(arch).cuda()
    n = sum(p.numel() for p in m.parameters())
    ids = torch.randint(0, 23004, (32, 384), device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, _ = m.forward_loss(ids[:, :-1], ids[:, 1:], fused_ce=True)
    loss.backward()
    print(f'[{arch}] WSL params {n:,} ({n/1e6:.1f}M) | fwd+bwd B32/T383 loss {loss.item():.3f} OK',
          flush=True)
print('WSL_SMOKE_ALL_OK', flush=True)
