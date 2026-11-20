"""B 段协议修复版：每个模型重置生成器（同 span 严格配对）+ NB=12 批收紧方差。"""
import sys
sys.path.insert(0, r'/mnt/f/OpenASH2605')
import torch
from train_ab import build

DEV = 'cuda'
DATA = '/mnt/f/OpenASH2605/minimind_data/pretrain_cached_1270238_384.pt'
raw = torch.load(DATA, map_location='cpu', weights_only=True)
stream = torch.cat([d.long() for d in raw]).cuda()

CKPTS = [('full_v53_40k', r'/mnt/f/OpenASH2605\full_v53_40k\model.pt'),
         ('full_v53r_40k', r'/mnt/f/OpenASH2605\full_v53r_40k\model.pt'),
         ('full_v53r2_40k', r'/mnt/f/OpenASH2605\full_v53r2_40k\model.pt'),
         ('full_gdn_80k_fixed', r'/mnt/f/OpenASH2605\full_gdn_80k_fixed\model.pt')]


def load(key, p):
    torch.manual_seed(2024)
    if key == 'full_gdn_80k_fixed':
        from baseline_fla import build_gdn, FlaLMWrapper
        m = FlaLMWrapper(build_gdn())
    else:
        m = build('v53')
    m.load_state_dict(torch.load(p, map_location='cpu', weights_only=True), strict=False)
    m.eval()
    return m.cuda()


@torch.no_grad()
def recall_all(m, gaps, A_SPAN=24, PREFIX=64, BS=16, NB=12, seed=99):
    gen = torch.Generator(device='cpu').manual_seed(seed)
    out = {}
    for gap in gaps:
        seq_len = PREFIX + A_SPAN + gap + A_SPAN
        lo = PREFIX + A_SPAN + gap - 1
        hi = PREFIX + 2 * A_SPAN + gap - 1
        ce_r_sum, ce_f_sum, n = 0.0, 0.0, 0
        for _ in range(NB):
            starts = torch.randint(0, stream.shape[0] - seq_len - 1, (BS,), generator=gen)
            for kind in ('r', 'f'):
                batch = []
                for i in starts.tolist():
                    pre = stream[i:i + PREFIX]
                    A = stream[i + PREFIX:i + PREFIX + A_SPAN]
                    gap_tok = stream[i + PREFIX + A_SPAN:i + PREFIX + A_SPAN + gap]
                    if kind == 'r':
                        batch.append(torch.cat([pre, A, gap_tok, A]))
                    else:
                        j = torch.randint(0, stream.shape[0] - A_SPAN - 1, (1,), generator=gen).item()
                        batch.append(torch.cat([pre, A, gap_tok, stream[j:j + A_SPAN]]))
                w = torch.stack(batch).cuda()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits, _ = m(w)
                lp = torch.log_softmax(logits[:, :-1].float(), -1)
                ce = -lp.gather(-1, w[:, 1:].unsqueeze(-1)).squeeze(-1)[:, lo:hi]
                if kind == 'r':
                    ce_r_sum += ce.sum().item()
                else:
                    ce_f_sum += ce.sum().item()
            n += BS * (hi - lo)
        out[gap] = (ce_r_sum / n, ce_f_sum / n - ce_r_sum / n)
    return out


gaps = [32, 64, 128, 256, 384]
res = {}
for key, p in CKPTS:
    m = load(key, p)
    res[key] = recall_all(m, gaps)
    del m
    torch.cuda.empty_cache()

print('gap'.rjust(6) + ''.join(k.rjust(22) for k, _ in CKPTS))
for gap in gaps:
    row = f'{gap:6d}'
    for key, _ in CKPTS:
        rep, gain = res[key][gap]
        row += f'  {rep:6.2f} {gain:+5.2f}'
    print(row)
print('\n(同 span 严格配对：每模型同种子序列，NB=12)')
