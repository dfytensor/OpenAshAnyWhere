# -*- coding: utf-8 -*-
"""probe_dm40k_v2.py — dm_v2@40k 召回特化终局探针（执行预注册判据）
主对照 d7@40k（精确底座）+ 辅对照 wr@40k / GDN；LM 守门 + 配对召回 + 利用曲线。"""
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
def heldout_ppl(m, L=384, n_win=256, bs=32):
    n = min(n_win, (val_stream.shape[0] - 1) // (L + 1))
    stride = max(1, (val_stream.shape[0] - L - 1) // n)
    starts = (torch.arange(0, n, device=DEV) * stride).tolist()
    tot, cnt = 0.0, 0
    for s in range(0, n, bs):
        idx = starts[s:s + bs]
        win = torch.stack([val_stream[i:i + L + 1] for i in idx]).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = m.forward_loss(win[:, :-1], win[:, 1:], fused_ce=True)
        tot += loss.item() * win[:, :-1].numel()
        cnt += win[:, :-1].numel()
    return tot / cnt


@torch.no_grad()
def heldout_7len(m):
    out = {}
    for L in (64, 128, 256, 384, 512, 1024, 2048):
        n = min(128, (val_stream.shape[0] - 1) // (L + 1))
        stride = max(1, (val_stream.shape[0] - L - 1) // n)
        starts = (torch.arange(0, n, device=DEV) * stride).tolist()
        tot, cnt = 0.0, 0
        for s in range(0, n, 16):
            idx = starts[s:s + 16]
            win = torch.stack([val_stream[i:i + L + 1] for i in idx]).cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, _ = m.forward_loss(win[:, :-1], win[:, 1:], fused_ce=True)
            tot += loss.item() * win[:, :-1].numel()
            cnt += win[:, :-1].numel()
        out[L] = tot / cnt
    return out


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


CK = [('dm_v2@40k', '/mnt/f/OpenASH2605/full_v53dm_v2_40k/model.pt', 'v53dm'),
      ('d7@40k', '/mnt/f/OpenASH2605/full_v53d7_40k/model.pt', 'v53d7'),
      ('wr@40k', '/mnt/f/OpenASH2605/full_v53wr_40k/model.pt', 'v53w')]

print('=== LM held-out（七长度） ===')
rows = {}
for key, path, arch in CK:
    torch.manual_seed(2024)
    m = build(arch)
    m.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=False)
    m.eval()
    m = m.cuda()
    lens = heldout_7len(m)
    print('  %-10s' % key, ' '.join('%d:%.2f' % (L, pow(2.718281828, v)) for L, v in lens.items()))
    rr = recall_all(m, [32, 64, 128, 256, 384], seed=99)
    rr['pos'] = pos_buckets(m)
    rr['ce384'] = lens[384]
    rows[key] = rr
    del m
    torch.cuda.empty_cache()

# GDN
from baseline_fla import build_gdn, FlaLMWrapper
torch.manual_seed(2024)
m = FlaLMWrapper(build_gdn())
m.load_state_dict(torch.load('/mnt/f/OpenASH2605/full_gdn_80k_fixed/model.pt',
                             map_location='cpu', weights_only=True))
m.eval()
m = m.cuda()
lens = heldout_7len(m)
print('  %-10s' % 'GDN', ' '.join('%d:%.2f' % (L, pow(2.718281828, v)) for L, v in lens.items()))
rr = recall_all(m, [32, 64, 128, 256, 384], seed=99)
rr['pos'] = pos_buckets(m)
rr['ce384'] = lens[384]
rows['GDN'] = rr
del m
torch.cuda.empty_cache()

print()
print('=== paired recall (NB=12) ===')
print('gap'.rjust(6) + ''.join(k.rjust(22) for k in rows))
for gap in (32, 64, 128, 256, 384):
    line = f'{gap:6d}'
    for key in rows:
        rep, gain = rows[key][gap]
        line += f'  {rep:6.2f} {gain:+5.2f}'
    print(line)
print()
print('=== 预注册判据执行 ===')
dm, d7 = rows['dm_v2@40k'], rows['d7@40k']
wins = 0
for gap in (32, 64, 128, 256, 384):
    d = dm[gap][0] - d7[gap][0]
    ok = d <= -0.05
    wins += ok
    print('  gap %-4d dm-d7 delta %+0.2f %s' % (gap, d, '✓' if ok else '✗'))
lm_pen = dm['ce384'] / d7['ce384'] - 1
print('  召回存续: %d/5 gaps ≥+0.05 → %s' % (wins, '成立' if wins >= 4 else '不成立→线永久关闭'))
print('  LM 守门: dm/d7 @384 惩罚 %+.1f%% → %s' % (lm_pen * 100, '通过(≤5%)' if lm_pen <= 0.05 else '不过(归档档)'))
print()
print('pos-bucket (b1..b8):')
for key in rows:
    print('  %-10s' % key, ' '.join(f'{x:5.2f}' for x in rows[key]['pos']))
