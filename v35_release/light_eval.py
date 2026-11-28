# -*- coding: utf-8 -*-
"""light_eval.py — 轻量配对评测（与训练并行时使用）
显存纪律：本进程硬上限 10%（约 2.4GB），超限自己 OOM、不挤训练。三处省显存：
  1) 数据流留在内存，只把窗口搬上 GPU；
  2) LM 用融合交叉熵（我方模型）/ 输出头按 512 位置分块（GDN），不物化整段 logits；
  3) 召回只对用到的 24 个位置过输出头（原实现物化整段 logits 再取对数，峰值 ~2.2GB）。
数值上与 bench / recall_paired_base 等价（同一输出头作用于同一隐状态）。"""
import math
import sys
import time

import torch
import torch.nn.functional as F

torch.cuda.set_per_process_memory_fraction(0.10)
sys.path.insert(0, r'/mnt/f/OpenASH2605')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from train_ab import build
from ckpt_guard import assert_zip_integrity

DEV = 'cuda'
t0 = time.time()
DATA = '/mnt/f/OpenASH2605/minimind_data/pretrain_cached_1270238_384.pt'
assert_zip_integrity(DATA, 'data cache')
torch.manual_seed(2024)
raw = torch.load(DATA, map_location='cpu', weights_only=True)
stream = torch.cat([d.long() for d in raw])
val = stream[-200_000:]
print('data ready in %.0fs | stream on %s, %d tokens' % (time.time() - t0, stream.device, stream.shape[0]), flush=True)
del raw


def load(arch, path):
    assert_zip_integrity(path, path.split('/')[-2])
    torch.manual_seed(2024)
    if arch is None:
        from baseline_fla import build_gdn, FlaLMWrapper
        m = FlaLMWrapper(build_gdn())
        m.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    else:
        m = build(arch)
        m.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=False)
    return m.eval().cuda()


def hidden_and_head(m, ids, gdn):
    """(hidden (B,S,D), head_fn)。须在 autocast 上下文内调用。"""
    if gdn:
        h = m.model.model(input_ids=ids).last_hidden_state
        return h, m.model.lm_head
    h, _ = m._hidden(ids)
    if m.em_weights_tied or m.tied_head:
        w, ls = m._head_weight, m.logit_scale
        return h, (lambda hs: hs @ w.t() * ls)
    return h, m.head_score


@torch.no_grad()
def ppl_at(m, L, gdn, n=96):
    bs = min(16, max(1, 2048 // L))     # 每批 ≤2048 token（本进程显存上限 2.4GB；16×512 的前向峰值 >2.3GB）
    stride = max(1, (val.shape[0] - L - 1) // n)
    starts = (torch.arange(0, n) * stride).tolist()
    tot, cnt = 0.0, 0
    for s in range(0, n, bs):
        idx = starts[s:s + bs]
        win = torch.stack([val[i:i + L + 1] for i in idx]).cuda()
        ids, tgt = win[:, :-1], win[:, 1:]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if not gdn:
                loss, _ = m.forward_loss(ids, tgt, fused_ce=True)
                tot += loss.item() * tgt.numel()
            else:
                h, head = hidden_and_head(m, ids, True)
                for a in range(0, L, 128):     # 输出头按 128 位置分块（16×128×23004×4 ≈ 188MB）
                    lg = head(h[:, a:a + 128])
                    tot += F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]),
                                           tgt[:, a:a + 128].reshape(-1), reduction='sum').item()
        cnt += tgt.numel()
    return math.exp(tot / cnt)


@torch.no_grad()
def recall_all(m, gdn, gaps, A_SPAN=24, PREFIX=64, BS=16, NB=12, seed=99, MB=4):
    """与 recall_paired_base.recall_all 同种子同 span（抽样顺序逐次一致：每轮 starts(16) → f 段 16 个 j）；
    仅把每轮 16 条序列切成 MB 条一组送入模型（控显存）；CE 只在 [lo,hi) 位置上计算"""
    gen = torch.Generator(device='cpu').manual_seed(seed)
    out = {}
    for gap in gaps:
        seq_len = PREFIX + A_SPAN + gap + A_SPAN
        lo = PREFIX + A_SPAN + gap - 1
        hi = PREFIX + 2 * A_SPAN + gap - 1
        ce_r_sum, ce_f_sum, n = 0.0, 0.0, 0
        for _ in range(NB):
            starts = torch.randint(0, stream.shape[0] - seq_len - 1, (BS,), generator=gen).tolist()
            js = [torch.randint(0, stream.shape[0] - A_SPAN - 1, (1,), generator=gen).item() for _ in range(BS)]
            for kind in ('r', 'f'):
                for a in range(0, BS, MB):
                    batch = []
                    for b in range(a, min(a + MB, BS)):
                        i = starts[b]
                        pre = stream[i:i + PREFIX]
                        A = stream[i + PREFIX:i + PREFIX + A_SPAN]
                        gap_tok = stream[i + PREFIX + A_SPAN:i + PREFIX + A_SPAN + gap]
                        if kind == 'r':
                            batch.append(torch.cat([pre, A, gap_tok, A]))
                        else:
                            j = js[b]
                            batch.append(torch.cat([pre, A, gap_tok, stream[j:j + A_SPAN]]))
                    w = torch.stack(batch).cuda()
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        h, head = hidden_and_head(m, w, gdn)
                        lg = head(h[:, lo:hi])            # 预测目标 w[lo+1 : hi+1]
                    lp = torch.log_softmax(lg.float(), -1)
                    ce = -lp.gather(-1, w[:, lo + 1:hi + 1].unsqueeze(-1)).squeeze(-1)
                    if kind == 'r':
                        ce_r_sum += ce.sum().item()
                    else:
                        ce_f_sum += ce.sum().item()
            n += BS * (hi - lo)
        out[gap] = (ce_r_sum / n, ce_f_sum / n - ce_r_sum / n)
    return out


MODELS = [('dml@80k', 'v53dml', '/mnt/e/OpenASH_loom/full_v53dml_80k/model.pt', False),
          ('GDN@160k', None, '/mnt/f/OpenASH2605/full_gdn_160k/model.pt', True),
          ('dms@80k', 'v53dms', '/mnt/f/OpenASH2605/full_v53dms_80k/model.pt', False)]
Ls = (64, 128, 256, 384, 512, 1024, 2048)
GAPS = [32, 64, 128, 256, 384]
lm, rc = {}, {}
import json
import os
CACHE = '/mnt/e/OpenASH_loom/light_eval_cache.json'
if os.path.exists(CACHE):
    c = json.load(open(CACHE))
    lm = c.get('lm', {})
    rc = {k: {int(g): tuple(v) for g, v in d.items()} for k, d in c.get('rc', {}).items()}


def save_cache():
    json.dump({'lm': lm, 'rc': {k: {str(g): list(v) for g, v in d.items()} for k, d in rc.items()}},
              open(CACHE, 'w'))


for name, arch, path, gdn in MODELS:
    if name in lm and name in rc:
        print('%s cached: LM %s | recall %s' % (
            name, ' '.join('%.2f' % x for x in lm[name]),
            ' '.join('%.2f' % rc[name][g][0] for g in GAPS)), flush=True)
        continue
    m = load(arch, path)
    try:
        lm[name] = [ppl_at(m, L, gdn) for L in Ls]
        print('%s LM: %s [%.0fs, peak %.2f GB]' % (
            name, ' '.join('%.2f' % x for x in lm[name]), time.time() - t0,
            torch.cuda.max_memory_allocated() / 1e9), flush=True)
        rc[name] = recall_all(m, gdn, GAPS)
        print('%s recall: %s [%.0fs]' % (
            name, ' '.join('%.2f' % rc[name][g][0] for g in GAPS), time.time() - t0), flush=True)
    except Exception as e:
        print('%s FAILED: %s' % (name, repr(e)[:200]), flush=True)
        lm.pop(name, None)
        rc.pop(name, None)
    del m
    torch.cuda.empty_cache()
    save_cache()
missing = [n for n, _, _, _ in MODELS if n not in lm or n not in rc]
if missing:
    print('MISSING MODELS: %s — 不输出裁决' % missing, flush=True)
    print('LIGHT EVAL INCOMPLETE', flush=True)
    sys.exit(2)

print()
print('=== LM ppl（轻量协议，同窗口配对，n=96）===')
print('L'.rjust(6) + ''.join(n.rjust(12) for n, _, _, _ in MODELS) + 'dml/GDN'.rjust(11) + 'dml/dms'.rjust(11))
for i, L in enumerate(Ls):
    d = lm['dml@80k'][i]
    print(str(L).rjust(6) + ''.join(('%.2f' % lm[n][i]).rjust(12) for n, _, _, _ in MODELS)
          + ('%+.1f%%' % ((d / lm['GDN@160k'][i] - 1) * 100)).rjust(11)
          + ('%+.1f%%' % ((d / lm['dms@80k'][i] - 1) * 100)).rjust(11))
print()
print('=== 召回 rep / gain（同 span 配对，NB=12）===')
print('gap'.rjust(6) + ''.join(n.rjust(22) for n, _, _, _ in MODELS))
for g in GAPS:
    print(str(g).rjust(6) + ''.join(('%6.2f %+0.2f' % rc[n][g]).rjust(22) for n, _, _, _ in MODELS))
print()
i384 = Ls.index(384)
d = lm['dml@80k'][i384]
print('=== 预注册判据（dml@80k，轻量协议预判；正式以官方 bench 为准）===')
print('1 主判据 @384 %.2f < GDN %.2f : %s' % (d, lm['GDN@160k'][i384], 'PASS' if d < lm['GDN@160k'][i384] else 'FAIL'))
nw = sum(1 for i in range(len(Ls)) if lm['dml@80k'][i] <= lm['GDN@160k'][i] * 1.003)
print('2 全长度不劣于 GDN: %d/7 (需 >=5)' % nw)
print('3 规模存续 vs dms@80k @384: %+.1f%% (需 <= -1.0%%)' % ((d / lm['dms@80k'][i384] - 1) * 100))
print('LIGHT EVAL DONE [%.0fs]' % (time.time() - t0))
