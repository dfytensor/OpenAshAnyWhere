# -*- coding: utf-8 -*-
"""train_gdn160k.py — GDN@160k 星标消除战（982M tokens = 2 epochs，与 dme@80k 严格同 token 量）

复刻 full_gdn_80k_fixed 配方：FLA GatedDeltaNet 5 层（106.3M），bs16，warmup 800，
peak 3e-4 余弦到 3e-5，AdamW(0.9,0.95,wd=0.01)，seed 2024，同数据同窗口。
步数 80k→160k（token 量与 dme@80k 完全对齐）。
"""
import argparse
import json
import math
import time

import torch

sys_path = r'F:\OpenASH2605'
import sys
sys.path.insert(0, sys_path)
from baseline_fla import build_gdn, FlaLMWrapper

ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, default=160000)
ap.add_argument('--warmup', type=int, default=800)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--batch_size', type=int, default=16)
ap.add_argument('--eval_every', type=int, default=8000)
ap.add_argument('--data', type=str, default='/mnt/f/OpenASH2605/minimind_data/pretrain_cached_1270238_384.pt')
ap.add_argument('--out', type=str, default='/mnt/f/OpenASH2605/full_gdn_160k')
args = ap.parse_args()

DEV = 'cuda'
torch.manual_seed(2024)
raw = torch.load(args.data, map_location='cpu', weights_only=True)
stream = torch.cat([d.long() for d in raw])
train_win = stream[:len(stream) - 200_000].unfold(0, 385, 1)
val_win = stream[-200_000:].unfold(0, 385, 1)
print('train_win', tuple(train_win.shape), 'val_win', tuple(val_win.shape))
gen = torch.Generator().manual_seed(2024)


def crop(bs):
    while True:
        i = torch.randint(0, train_win.shape[0], (bs,), generator=gen)
        yield train_win[i].cuda()


import os
os.makedirs(args.out, exist_ok=True)

torch.manual_seed(2024)
model = FlaLMWrapper(build_gdn()).to(DEV)
n_params = sum(p.numel() for p in model.parameters())
print('[gdn] params: %s' % f'{n_params:,}')

n_win_eval = 256


@torch.no_grad()
def evaluate(bs=32):
    model.eval()
    n_win = min(n_win_eval, val_win.shape[0])
    stride = max(1, val_win.shape[0] // n_win)
    tot, cnt = 0.0, 0
    for s in range(0, n_win, bs):
        idx = (torch.arange(0, n_win, device=DEV) * stride)[s:s + bs]
        win = val_win[idx].cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = model.forward_loss(win[:, :-1], win[:, 1:], fused_ce=True)
        tot += loss.item() * win[:, :-1].numel()
        cnt += win[:, :-1].numel()
    model.train()
    return tot / cnt


def lr_at(step):
    if step < args.warmup:
        return args.lr * step / args.warmup
    p = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * p)) * 0.9 + args.lr * 0.1


opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
it = crop(args.batch_size)
log = []
t0 = time.time()
model.train()
for step in range(1, args.steps + 1):
    lr = lr_at(step)
    for gp in opt.param_groups:
        gp['lr'] = lr
    batch = next(it)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, _ = model.forward_loss(batch[:, :-1], batch[:, 1:], fused_ce=True)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 100 == 0 or step == 1:
        ms = (time.time() - t0) / 100 * 1000
        t0 = time.time()
        rec = {'step': step, 'loss': loss.item(), 'lr': lr, 'ms_per_step': ms}
        log.append(rec)
        print('[gdn] step %6d | loss %.3f | lr %.2e | %.0f ms/step' % (step, loss.item(), lr, ms))
        if step % args.eval_every == 0 or step == args.steps:
            v = evaluate()
            ppl = math.exp(v)
            rec['val'] = v
            print('[gdn]   >> val %.4f | ppl %.2f' % (v, ppl))

torch.save(model.state_dict(), os.path.join(args.out, 'model.pt'))
json.dump({'arch': 'gdn', 'params': n_params, 'steps': args.steps, 'log': log},
          open(os.path.join(args.out, 'train_log.json'), 'w'))
print('[gdn] done -> %s' % args.out)
