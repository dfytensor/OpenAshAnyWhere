# -*- coding: utf-8 -*-
"""probe_gw.py — v53gw 落地裁决：γ_kv 学习终值（核心证据）+ LM + 配对召回 + 利用曲线"""
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

# ---------- 1) γ_kv 学习终值 ----------
sd = torch.load('/mnt/f/OpenASH2605/scale_v53gw_15k/model.pt', map_location='cpu', weights_only=True)
gammas = {k: v.item() for k, v in sd.items() if 'kv_gamma' in k}
print('=== γ_kv learned values (init 1.0) ===')
for k in sorted(gammas, key=lambda s: int(s.split('.')[1][:2])):
    print('  %-28s %.4f' % (k, gammas[k]))
gvals = list(gammas.values())
print('  mean %.4f  min %.4f  max %.4f' % (sum(gvals) / len(gvals), min(gvals), max(gvals)))

# ---------- 2) LM held-out ----------
exec(open('/mnt/f/OpenASH2605/recall_paired_base.py', encoding='utf-8').read().split('gaps = ')[0])
val_stream = stream[-200_000:]


def load(key, path, arch):
    torch.manual_seed(2024)
    m = build(arch)
    m.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=False)
    m.eval()
    return m.cuda()


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


CK = [('w@15k', '/mnt/f/OpenASH2605/scale_v53w_15k/model.pt', 'v53w'),
      ('gw@15k', '/mnt/f/OpenASH2605/scale_v53gw_15k/model.pt', 'v53gw'),
      ('GDN', '/mnt/f/OpenASH2605/full_gdn_80k_fixed/model.pt', None)]

print()
print('=== LM held-out @384 ===')
ppls = {}
for key, path, arch in CK:
    if arch is None:
        from baseline_fla import build_gdn, FlaLMWrapper
        torch.manual_seed(2024)
        m = FlaLMWrapper(build_gdn())
        m.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        m.eval()
        m = m.cuda()
    else:
        m = load(key, path, arch)
    lp = heldout_ppl(m)
    ppls[key] = lp
    print('  %-8s ppl %.2f' % (key, lp))
    res_key = key
    if key == 'GDN':
        globals()['gdn_m'] = None
    globals()['m_%s' % key] = m

print()
print('=== paired recall (NB=12) + pos-buckets ===')
rows = {}
for key, path, arch in CK:
    m = globals()['m_%s' % key]
    if arch is not None:
        rr = recall_all(m, [32, 64, 128, 256, 384], seed=99)
    else:
        rr = recall_all(m, [32, 64, 128, 256, 384], seed=99)
    rr['pos'] = pos_buckets(m)
    rows[key] = rr
    del m
    torch.cuda.empty_cache()

hdr = 'gap'.rjust(6) + ''.join(k.rjust(22) for k in rows)
print(hdr)
for gap in (32, 64, 128, 256, 384):
    line = f'{gap:6d}'
    for key in rows:
        rep, gain = rows[key][gap]
        line += f'  {rep:6.2f} {gain:+5.2f}'
    print(line)
print()
print('pos-bucket (b1..b8):')
for key in rows:
    print('  %-8s' % key, ' '.join(f'{x:5.2f}' for x in rows[key]['pos']))
