# -*- coding: utf-8 -*-
"""probe_dme.py — dm-es@40k 等规模对决探针（执行预注册判据）
官方榜：dm-es vs f1（等规模冠军）；副判据：dm-es vs dm_v2 孪生（裁剪代价）；GDN 参照。"""
import sys
import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
from train_ab import build

DEV = 'cuda'
DATA = '/mnt/f/OpenASH2605/minimind_data/pretrain_cached_1270238_384.pt'
torch.manual_seed(2024)
raw = torch.load(DATA, map_location='cpu', weights_only=True)
stream = torch.cat([d.long() for d in raw]).cuda()
val_stream = stream[-200_000:]

exec(open('/mnt/f/OpenASH2605/recall_paired_base.py', encoding='utf-8').read().split('gaps = ')[0])
val_stream = stream[-200_000:]


@torch.no_grad()
def pos_buckets(m, L=384, n_win=256, bs=32):
    n = min(n_win, (val_stream.shape[0] - 1) // (L + 1))
    stride = max(1, (val_stream.shape[0] - L - 1) // n)
    starts = (torch.arange(0, n, device=DEV) * stride).tolist()
    buckets = [[] for _ in range(8)]
    for s in range(0, n, bs):
        idx = starts[s:s + bs]
        win = torch.stack([val_stream[i:i + L + 1] for i in idx]).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = m(win)
        lp = torch.log_softmax(logits[:, :-1].float(), -1)
        ce = -lp.gather(-1, win[:, 1:].unsqueeze(-1)).squeeze(-1)
        blen = ce.shape[1] // 8
        for b in range(8):
            lo, hi = b * blen, (b + 1) * blen if b < 7 else ce.shape[1]
            buckets[b].append(ce[:, lo:hi].mean().item())
    return [sum(b) / len(b) for b in buckets]


CK = [('dme@40k', '/mnt/f/OpenASH2605/full_v53dme_40k/model.pt', 'v53dme'),
      ('dm_v2@40k', '/mnt/f/OpenASH2605/full_v53dm_v2_40k/model.pt', 'v53dm'),
      ('f1@40k', '/mnt/f/OpenASH2605/full_v53f1_40k/model.pt', 'v53f1')]

print('=== paired recall (NB=12) ===')
rows = {}
for key, path, arch in CK:
    torch.manual_seed(2024)
    m = build(arch)
    m.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=False)
    m.eval()
    m = m.cuda()
    rr = recall_all(m, [32, 64, 128, 256, 384], seed=99)
    rr['pos'] = pos_buckets(m)
    rows[key] = rr
    del m
    torch.cuda.empty_cache()

print('gap'.rjust(6) + ''.join(k.rjust(20) for k in rows))
for gap in (32, 64, 128, 256, 384):
    line = f'{gap:6d}'
    for key in rows:
        rep, gain = rows[key][gap]
        line += f'  {rep:6.2f} {gain:+5.2f}'
    print(line)
print()
print('pos-bucket (b1..b8):')
for key in rows:
    print('  %-10s' % key, ' '.join(f'{x:5.2f}' for x in rows[key]['pos']))
print()
print('=== 裁剪代价（dme vs dm_v2 孪生） ===')
for gap in (32, 64, 128, 256, 384):
    d = rows['dme@40k'][gap][0] - rows['dm_v2@40k'][gap][0]
    print('  gap %-4d %+0.2f' % (gap, d))
print('（LM 代价以 bench_longrange 官方协议为准，本探针后运行）')
