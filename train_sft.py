"""SFT 训练：从预训练 ckpt 继续，在对话数据（sft_cached_60000_384.pt）上微调。

用法: python train_sft.py --base big_v35/model.pt --arch v35 --out sft_v35 --epochs 3
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, r'F:\OpenASH2605')
from train_ab import build

DEV = 'cuda'
SFT_DATA = r'F:\OpenASH2605\minimind_data\sft_cached_60000_384.pt'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--arch', default='v35')
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=150)
    ap.add_argument('--out', required=True)
    ap.add_argument('--eval_every', type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    L = 384

    torch.manual_seed(2024)
    sft = torch.load(SFT_DATA, map_location='cpu', weights_only=True)
    if isinstance(sft, (tuple, list)):
        sft = sft[0]
    if sft.dim() == 3:
        sft = sft.squeeze(0)
    sft_win = sft.long()
    print(f'sft windows: {sft_win.shape[0]:,}')
    gen = torch.Generator(device='cpu').manual_seed(31)

    def crop(bs):
        while True:
            i = torch.randint(0, sft_win.shape[0], (bs,), generator=gen)
            yield sft_win[i].cuda()

    torch.manual_seed(2024)
    model = build(args.arch).to(DEV)
    sd = torch.load(args.base, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[{args.arch}] base loaded | missing={len(missing)} unexpected={len(unexpected)}')
    if missing:
        print('  missing e.g.:', missing[:3])

    @torch.no_grad()
    def evaluate(n_win=256, bs=32):
        model.eval()
        stride = max(1, sft_win.shape[0] // n_win)
        starts = (torch.arange(0, n_win, device=DEV) * stride)
        tot, cnt = 0.0, 0
        for s in range(0, n_win, bs):
            win = sft_win[starts[s:s + bs].tolist()].cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, _ = model.forward_loss(win[:, :-1], win[:, 1:], fused_ce=True)
            tot += loss.item() * win[:, :-1].numel()
            cnt += win[:, :-1].numel()
        model.train()
        return tot / cnt

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * p)) * 0.9 + args.lr * 0.1

    it = crop(args.batch_size)
    log, t0 = [], time.time()
    model.train()
    for step in range(1, args.steps + 1):
        lr = lr_at(step)
        for gp in opt.param_groups:
            gp['lr'] = lr
        batch = next(it)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = model.forward_loss(batch[:, :-1], batch[:, 1:], fused_ce=True)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            print(f'[sft] step {step:5d} | loss {loss.item():.3f} | lr {lr:.2e} | '
                  f'{(time.time() - t0) / step * 1000:.0f} ms/step')
        if step % args.eval_every == 0 or step == args.steps:
            vl = evaluate()
            log.append({'step': step, 'loss': loss.item(), 'val_loss': vl, 'lr': lr})
            print(f'[sft]   >> sft val {vl:.4f} | ppl {math.exp(min(vl, 20)):.2f}')
            json.dump(log, open(os.path.join(args.out, 'sft_log.json'), 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=1)

    torch.save(model.state_dict(), os.path.join(args.out, 'model.pt'))
    json.dump(log, open(os.path.join(args.out, 'sft_log.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'[sft] done -> {args.out}/')


if __name__ == '__main__':
    main()
