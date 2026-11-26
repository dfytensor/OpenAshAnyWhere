import sys
sys.path.insert(0, '/mnt/f/OpenASH2605')
import torch
from train_ab import build
torch.manual_seed(2024)
m = build('v47').cuda()
n = sum(p.numel() for p in m.parameters())
ids = torch.randint(0, 23004, (2, 384), device='cuda')
with torch.autocast('cuda', dtype=torch.bfloat16):
    loss, _ = m.forward_loss(ids[:, :-1], ids[:, 1:], fused_ce=True)
loss.backward()
print(f'WSL v47 bf16 {n/1e6:.1f}M | fwd+bwd {loss.item():.3f} OK')
