import sys
sys.path.insert(0, r'F:\OpenASH2605')
import torch
from train_ab import build
torch.manual_seed(2024)
m = build('v48').cuda()
n = sum(p.numel() for p in m.parameters())
ids = torch.randint(0, 23004, (2, 384), device='cuda')
with torch.autocast('cuda', dtype=torch.bfloat16):
    loss, _ = m.forward_loss(ids[:, :-1], ids[:, 1:], fused_ce=True)
loss.backward()
m.eval()
ids2 = torch.randint(0, 23004, (1, 64), device='cuda')
with torch.no_grad():
    st = None
    lps = []
    for t in range(64):
        lt, st = m(ids2[:, t:t+1], state=st, pos_offset=t)
        lps.append(torch.log_softmax(lt[:, -1].float(), -1))
    ours = torch.stack(lps, 1)
    lg, _ = m(ids2)
    ref = torch.log_softmax(lg.float(), -1)[:, :64]
print(f'v48 {n/1e6:.1f}M | fwd+bwd {loss.item():.3f} | decode fp32: {(ours-ref).abs().max().item():.2e}')
