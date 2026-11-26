"""OpenASH v3 在 minimind 数据上的训练验证。

数据: minimind_data/pretrain_cached_30000_384.pt
      (30,000 条 × 358 token, OpenASH 代理词表 id ∈ [0, 23003])
验证内容:
  1. kernel 一致性: fused(全优化) vs eager 前 400 步 loss 轨迹
  2. 语言建模: train/val loss 下降 + 生成样本定性检查
用法: python train_minimind_v3.py [--steps 8000] [--batch_size 32]
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from open_ash_v3 import OpenASH_V3
from open_ash_voc import OpenASHVoc
from config import agent_voc_path

DEV = 'cuda'
DATA = r'F:\OpenASH2605\minimind_data\pretrain_cached_30000_384.pt'


def make_crop(win, L, batch_size, seed):
    """从窗口视图随机取 (B, L+1) 批并搬到 GPU，固定种子可复现。"""
    g = torch.Generator().manual_seed(seed)
    while True:
        starts = torch.randint(0, win.shape[0], (batch_size,), generator=g)
        yield win[starts].to(DEV, non_blocking=True)


@torch.no_grad()
def evaluate(model, val_win, batch_size=32):
    model.eval()
    total, count = 0.0, 0
    g = torch.Generator().manual_seed(99)
    n = val_win.shape[0]
    for _ in range(64):  # 固定 64 批评估
        starts = torch.randint(0, n, (batch_size,), generator=g)
        batch = val_win[starts].to(DEV, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = model.forward_loss(batch[:, :-1], batch[:, 1:], fused_ce=True)
        total += loss.item()
        count += 1
    model.train()
    return total / count


@torch.no_grad()
def generate(model, voc, prompt, n_new=80, temperature=0.8, top_k=50):
    model.eval()
    ids = voc.encode(prompt)
    if not ids:
        return '(empty prompt)'
    x = torch.tensor([ids], dtype=torch.long, device=DEV)
    state = None
    out, state = model(x, state=state)
    generated = []
    for _ in range(n_new):
        logits = out[0, -1].float() / temperature
        k = min(top_k, logits.shape[-1])
        v, _ = torch.topk(logits, k)
        logits[logits < v[-1]] = -float('inf')
        probs = torch.softmax(logits, -1)
        nxt = torch.multinomial(probs, 1).item()
        generated.append(nxt)
        out, state = model(torch.tensor([[nxt]], device=DEV), state=[s.detach() for s in state],
                           pos_offset=len(ids) + len(generated))
    model.train()
    return prompt + voc.decode(generated)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=150)
    ap.add_argument('--control_steps', type=int, default=400)
    ap.add_argument('--decay', type=int, default=1, help='1: 衰减最大值 (v3.1)；0: 纯 cummax (v3.0)')
    ap.add_argument('--out', type=str, default='minimind_v3_out')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(2024)
    raw = torch.load(DATA, map_location='cpu', weights_only=True)
    stream = torch.cat([d.long() for d in raw])
    L = 384
    n_val_tokens = 200_000
    val_win = stream[-n_val_tokens:].unfold(0, L + 1, 1)
    train_win = stream[:-n_val_tokens].unfold(0, L + 1, 1)
    print(f'data: tokens {stream.shape[0]:,} | train windows {train_win.shape[0]:,} | '
          f'val windows {val_win.shape[0]:,} | crop len {L}')

    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    voc_size = 23004
    hidden, heads, layers = 768, 8, 12

    def build(use_fused):
        torch.manual_seed(2024)
        return OpenASH_V3(voc_size, hidden, heads, layers, max_seq_len=512,
                          model_flag='train', use_fused=use_fused, use_decay=bool(args.decay)).to(DEV)

    # ---------- 阶段 1：kernel 一致性（fused vs eager，同种子前 N 步） ----------
    n_ctrl = args.control_steps
    if n_ctrl > 0:
        traces = {}
        for tag, uf in (('eager', False), ('fused', True)):
            torch.manual_seed(7)
            model = build(uf)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
            it = make_crop(train_win, L, args.batch_size, seed=11)
            tr = []
            for i in range(n_ctrl):
                batch = next(it)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, _ = model.forward_loss(batch[:, :-1], batch[:, 1:], fused_ce=uf)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tr.append(loss.item())
            traces[tag] = tr
            del model
            torch.cuda.empty_cache()
        diffs = [abs(a - b) for a, b in zip(traces['fused'], traces['eager'])]
        print(f'[一致性] 前 {n_ctrl} 步 max|fused-eager| = {max(diffs):.4f} | '
              f'loss {traces["fused"][0]:.3f} -> {traces["fused"][-1]:.3f}')
        json.dump({'fused': traces['fused'], 'eager': traces['eager']},
                  open(os.path.join(args.out, 'control_traces.json'), 'w', encoding='utf-8'))

    # ---------- 阶段 2：正式训练（全优化） ----------
    torch.manual_seed(2024)
    model = build(True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'model params: {n_params:,}')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    warmup, total = args.warmup, args.steps

    def lr_at(step):
        if step < warmup:
            return args.lr * step / warmup
        p = (step - warmup) / max(1, total - warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * p)) * 0.9 + args.lr * 0.1

    it = make_crop(train_win, L, args.batch_size, seed=23)
    log, t0 = [], time.time()
    model.train()
    for step in range(1, args.steps + 1):
        lr = lr_at(step)
        for gparam in opt.param_groups:
            gparam['lr'] = lr
        batch = next(it)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = model.forward_loss(batch[:, :-1], batch[:, 1:], fused_ce=True)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            log.append({'step': step, 'loss': loss.item(), 'lr': lr,
                        'tps': step * args.batch_size * L / (time.time() - t0)})
            print(f'step {step:5d} | loss {loss.item():.3f} | lr {lr:.2e} | '
                  f'{(time.time() - t0) / step * 1000:.0f} ms/step')
        if step % 500 == 0 or step == args.steps:
            vl = evaluate(model, val_win)
            log[-1]['val_loss'] = vl
            print(f'  >> val loss {vl:.3f} | ppl {math.exp(min(vl, 20)):.2f}')

    torch.save(model.state_dict(), os.path.join(args.out, 'v3_minimind.pt'))
    json.dump(log, open(os.path.join(args.out, 'train_log.json'), 'w', encoding='utf-8'))

    # ---------- 阶段 3：生成样本 ----------
    print('\n[生成样本]')
    samples = []
    for prompt in ('人工智能是', '中国的首都是', '春天来了，', '中国经济'):
        text = generate(model, voc, prompt)
        samples.append({'prompt': prompt, 'text': text})
        print(f'  [{prompt}] -> {text[:120]}')
    json.dump(samples, open(os.path.join(args.out, 'samples.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print(f'\ncheckpoint/log/samples -> {args.out}/')


if __name__ == '__main__':
    main()
