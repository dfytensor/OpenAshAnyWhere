import sys
sys.path.insert(0, r'F:\OpenASH2605')
src = open(r'/mnt/f/OpenASH2605/recall_paired_base.py', encoding='utf-8').read().split('gaps = ')[0]
src = src.replace("m = build('v53')", "m = build(arch_of(key))")
exec(src)
val_stream = stream[-200_000:]


def arch_of(key):
    if key.startswith('v53dm') or key.startswith('scale_v53dm'):
        return 'v53dm'
    if key.startswith('v53w') or key.startswith('scale_v53w'):
        return 'v53w'
    return 'v53'


def load(key, p):
    torch.manual_seed(2024)
    if key == 'full_gdn_80k_fixed':
        from baseline_fla import build_gdn, FlaLMWrapper
        m = FlaLMWrapper(build_gdn())
        m.load_state_dict(torch.load(p, map_location='cpu', weights_only=True))
        m.eval()
        return m.cuda()
    m = build(arch_of(key))
    m.load_state_dict(torch.load(p, map_location='cpu', weights_only=True), strict=False)
    m.eval()
    return m.cuda()


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


ck = [('v53w@15k', r'/mnt/f/OpenASH2605/scale_v53w_15k/model.pt'),
      ('v53dm_v2@15k', r'/mnt/f/OpenASH2605/scale_v53dm_15k_v2/model.pt'),
      ('full_gdn_80k_fixed', r'/mnt/f/OpenASH2605/full_gdn_80k_fixed/model.pt')]
res = {}
for key, p in ck:
    m = load(key, p)
    res[key] = recall_all(m, [32, 64, 128, 256, 384], seed=99)
    res[key]['pos'] = pos_buckets(m)
    del m
    torch.cuda.empty_cache()

print('gap'.rjust(6) + ''.join(k.rjust(24) for k, _ in ck))
for gap in (32, 64, 128, 256, 384):
    row = f'{gap:6d}'
    for key, _ in ck:
        rep, gain = res[key][gap]
        row += f'  {rep:6.2f} {gain:+5.2f}'
    print(row)
print()
print('pos-bucket (b1..b8):')
for key, _ in ck:
    print(f'  {key:24s}', ' '.join(f'{x:5.2f}' for x in res[key]['pos']))
