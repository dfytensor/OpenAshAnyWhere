"""统一 A/B 训练入口：同 seed / 同数据顺序 / 同预算（2500 步 × bs32 × L384）。

用法: python train_ab.py --arch v32 --out ab_v32
arch: v30 (pure cummax) | v31 (v3.1 decay) | v32 (v3.2 decay+现代化) | tf (Transformer 基线)
"""
import argparse
import json
import math
import os
import time

import torch

from open_ash_v3 import OpenASH_V3
from baseline_tf import TransformerLM

DEV = 'cuda'
DATA = r'F:\OpenASH2605\minimind_data\pretrain_cached_30000_384.pt'
VOC, HID, HEADS, LAYERS = 23004, 768, 8, 12


def build(arch):
    if arch in ('m2', 'gdn', 'dn'):
        import baseline_fla
        return baseline_fla.FlaLMWrapper(baseline_fla.BUILDERS[arch]())
    if arch == 'v36b1':
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=48, kv_backend='delta_fla', kv_beta_init=4.0)
    if arch == 'v39':
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=48, kv_backend='delta_chunk', kv_beta_init=4.0,
                          short_conv=True)
    if arch == 'v38':
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=48, kv_backend='delta_fla', kv_beta_init=4.0,
                          short_conv=True, kv_beta_tok=True)
    if arch == 'v37':
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=48, kv_backend='delta_fla', kv_beta_init=4.0,
                          short_conv=True)
    if arch == 'v36':
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=48, kv_backend='delta_fla')
    if arch == 'v30':
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=512, model_flag='train',
                          use_fused=True, use_decay=False)
    if arch == 'v31':
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=512, model_flag='train',
                          use_fused=True, use_decay=True)
    if arch == 'v32':
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04)
    if arch == 'v33':
        m = OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                       use_fused=True, use_decay=True,
                       pre_norm=True, ffn_act='silu',
                       decay_init=(0.0001, 0.35), decay_v_std=0.04)
        # 分层多尺度：顶层 6 层改为慢通道（γ ∈ [0.995, 0.9995]，半衰期 138~1386 tok）
        # —— 保留 v3.2 的快通道学习，同时给远程逐字记忆专用的状态载体
        with torch.no_grad():
            for layer in m.decoder_layers[6:]:
                s = torch.linspace(0.0005, 0.005, HID)
                layer.self_attention_linear.decay_b.copy_(torch.log(torch.expm1(s)))
        return m
    if arch == 'v34':
        m = OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                       use_fused=True, use_decay=True,
                       pre_norm=True, ffn_act='silu',
                       decay_init=(0.0001, 0.35), decay_v_std=0.04,
                       kv_channel=True, kv_decay=0.98)
        # 分层多尺度（同 v33）：顶层 6 层慢通道
        with torch.no_grad():
            for layer in m.decoder_layers[6:]:
                s = torch.linspace(0.0005, 0.005, HID)
                layer.self_attention_linear.decay_b.copy_(torch.log(torch.expm1(s)))
        return m
    if arch == 'v35':
        m = OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                       use_fused=True, use_decay=True,
                       pre_norm=True, ffn_act='silu',
                       decay_init=(0.0001, 0.35), decay_v_std=0.04,
                       kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True)
        with torch.no_grad():
            for layer in m.decoder_layers[6:]:
                s = torch.linspace(0.0005, 0.005, HID)
                layer.self_attention_linear.decay_b.copy_(torch.log(torch.expm1(s)))
        return m
    if arch == 'v35s':
        m = OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                       use_fused=True, use_decay=True,
                       pre_norm=True, ffn_act='silu',
                       decay_init=(0.0001, 0.35), decay_v_std=0.04,
                       kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                       kv_dim=48)
        with torch.no_grad():
            for layer in m.decoder_layers[6:]:
                s = torch.linspace(0.0005, 0.005, HID)
                layer.self_attention_linear.decay_b.copy_(torch.log(torch.expm1(s)))
        return m
    if arch == 'v35d':
        m = OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                       use_fused=True, use_decay=True,
                       pre_norm=True, ffn_act='silu',
                       decay_init=(0.0001, 0.35), decay_v_std=0.04,
                       kv_channel=True, kv_decay=0.98, kv_norm_keys=True,
                       kv_beta=False, kv_delta=True)
        with torch.no_grad():
            for layer in m.decoder_layers[6:]:
                s = torch.linspace(0.0005, 0.005, HID)
                layer.self_attention_linear.decay_b.copy_(torch.log(torch.expm1(s)))
        return m
    if arch == 'tf':
        return TransformerLM(VOC, HID, HEADS, 11, max_seq_len=2048)
    raise ValueError(arch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', required=True)
    ap.add_argument('--steps', type=int, default=2500)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=150)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data', default=r'F:\OpenASH2605\minimind_data\pretrain_cached_30000_384.pt')
    ap.add_argument('--eval_every', type=int, default=2000)
    args = ap.parse_args()
    arch = args.arch
    os.makedirs(args.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    L = 384

    torch.manual_seed(2024)
    raw = torch.load(args.data, map_location='cpu', weights_only=True)
    stream = torch.cat([d.long() for d in raw]).cuda()
    n_val = 200_000
    val_stream = stream[-n_val:]
    train_stream = stream[:-n_val]
    val_win = val_stream.unfold(0, L + 1, 1)
    train_win = train_stream.unfold(0, L + 1, 1)
    gen = torch.Generator(device='cpu').manual_seed(23)

    def crop(bs):
        while True:
            i = torch.randint(0, train_win.shape[0], (bs,), generator=gen)
            yield train_win[i].cuda()

    torch.manual_seed(2024)
    model = build(arch).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[{args.arch}] params: {n_params:,}')

    n_win_eval = 256
    @torch.no_grad()
    def evaluate(bs=32):
        model.eval()
        n_win = min(256, val_win.shape[0])
        stride = max(1, val_win.shape[0] // n_win)
        starts = (torch.arange(0, n_win, device=DEV) * stride)
        tot, cnt = 0.0, 0
        for s in range(0, n_win, bs):
            idx = starts[s:s + bs]
            win = val_win[idx]
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
            log.append({'step': step, 'loss': loss.item(), 'lr': lr,
                        'ms_per_step': (time.time() - t0) / step * 1000})
            print(f'[{args.arch}] step {step:5d} | loss {loss.item():.3f} | lr {lr:.2e} | '
                  f'{(time.time() - t0) / step * 1000:.0f} ms/step')
        if step % args.eval_every == 0 or step == args.steps:
            vl = evaluate()
            log[-1]['val_loss'] = vl
            print(f'[{args.arch}]   >> val {vl:.4f} | ppl {math.exp(min(vl, 20)):.2f}')

    torch.save(model.state_dict(), os.path.join(args.out, 'model.pt'))
    json.dump({'arch': args.arch, 'params': n_params, 'log': log},
              open(os.path.join(args.out, 'train_log.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'[{args.arch}] done -> {args.out}/')


if __name__ == '__main__':
    main()
