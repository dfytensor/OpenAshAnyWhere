import sys
sys.path.insert(0, r'F:\OpenASH2605')
import torch
import math
from train_ab import build

DEV = 'cuda'
DATA = r'/mnt/f/OpenASH2605/minimind_data/pretrain_cached_1270238_384.pt'
torch.manual_seed(2024)
raw = torch.load(DATA, map_location='cpu', weights_only=True)
stream = torch.cat([d.long() for d in raw]).cuda()
val_stream = stream[-200_000:]
val_win = val_stream.unfold(0, 385, 1)
print('val_win:', val_win.shape)

torch.manual_seed(2024)
m = build('v53k').cuda()
m.load_state_dict(torch.load(r'/mnt/f/OpenASH2605/full_v53kr_40k/model.pt',
                             map_location='cpu', weights_only=True), strict=False)
m.eval()


@torch.no_grad()
def eval_len(m, L, n_win=256, bs=32):
    n = min(n_win, (val_stream.shape[0] - 1) // (L + 1))
    stride = max(1, (val_stream.shape[0] - L - 1) // n)
    starts = (torch.arange(0, n, device=DEV) * stride)
    tot, cnt = 0.0, 0
    for s in range(0, n, bs):
        idx = starts[s:s + bs]
        win = torch.stack([val_stream[i:i + L + 1] for i in idx.tolist()]).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = m.forward_loss(win[:, :-1], win[:, 1:], fused_ce=True)
        tot += loss.item() * win[:, :-1].numel()
        cnt += win[:, :-1].numel()
    return tot / cnt


for L in (64, 384, 2048):
    ce = eval_len(m, L)
    print('kr held-out L=%d: ppl %.2f' % (L, math.exp(min(ce, 20))))
