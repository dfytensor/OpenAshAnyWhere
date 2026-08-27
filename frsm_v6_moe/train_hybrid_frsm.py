"""
HybridFRSM 全量预训练 — 快慢尺度 1:1 并行架构
使用已有缓存 pretrain_cached_1270238_384.pt
"""
import os, sys, time, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys as _sys; _sys.stdout.reconfigure(line_buffering=True)

import torch, torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

torch.set_float32_matmul_precision('high')

def run():
    parser = argparse.ArgumentParser(description="HybridFRSM Pretrain")
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--num_fast", type=int, default=1)
    parser.add_argument("--num_slow", type=int, default=1)
    parser.add_argument("--slow_update_freq", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=88)
    parser.add_argument("--max_seq_len", type=int, default=384)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_lines", type=int, default=1270238)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="hybrid_frsm_pretrain")
    parser.add_argument("--data_dir", type=str, default="../minimind_data")
    parser.add_argument("--warmup_steps", type=int, default=500)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载词表
    from config import agent_voc_path
    from open_ash_voc import OpenASHVoc
    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    vs = len(voc.token_to_id) + 1
    print(f"Vocab size: {vs}")

    # 加载模型
    from frsm_linear import HybridFRSM_LM
    model = HybridFRSM_LM(
        vocab_size=vs, d_model=args.d_model,
        num_fast=args.num_fast, num_slow=args.num_slow,
        slow_update_freq=args.slow_update_freq,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} ({param_count/1e6:.1f}M)")
    print(f"Config: d={args.d_model} {args.num_fast}F+{args.num_slow}S K={args.slow_update_freq}")

    # 加载缓存数据集
    data_dir_abs = Path(args.data_dir).resolve()
    cache_file = data_dir_abs / f"pretrain_cached_{args.max_lines}_{args.max_seq_len}.pt"
    if cache_file.exists():
        print(f"Loading cached dataset from {cache_file}")
        cached = torch.load(cache_file, weights_only=True)
        print(f"Loaded {len(cached)} samples from cache")
    else:
        print(f"Cache not found at {cache_file}, loading from jsonl...")
        from frsm.dataset import PretrainDataset
        dataset = PretrainDataset(
            str(data_dir_abs / "pretrain_t2t_mini.jsonl"), voc,
            max_len=args.max_seq_len, max_lines=args.max_lines,
        )
        cached = dataset.data
        torch.save(cached, cache_file)
        print(f"Loaded {len(cached)} samples, cached to {cache_file}")

    # DataLoader (复用 PretrainDataset 的 collate_fn)
    class CachedDataset(torch.utils.data.Dataset):
        def __init__(self, data, max_len):
            self.data = data; self.max_len = max_len
        def __len__(self): return len(self.data)
        def __getitem__(self, i):
            ids = self.data[i]
            if len(ids) > self.max_len + 1: ids = ids[:self.max_len + 1]
            return ids
        @staticmethod
        def collate_fn(items):
            padded = pad_sequence(items, batch_first=True, padding_value=0)
            return padded[:, :-1], padded[:, 1:]

    dataset = CachedDataset(cached, args.max_seq_len)
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=CachedDataset.collate_fn, drop_last=True,
    )

    steps_per_epoch = len(dataset) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    print(f"Batch: {args.batch_size}, Seq: {args.max_seq_len}")
    print(f"Steps/epoch: {steps_per_epoch}, Total: {total_steps}, Epochs: {args.epochs}")

    # 优化器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    lr_lambda = lambda s: min(s / max(1, args.warmup_steps), 1.0) if s < args.warmup_steps \
        else max(0, 0.5 * (1 + math.cos(math.pi * (s - args.warmup_steps) / max(1, total_steps - args.warmup_steps))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 检查点
    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_path = ckpt_dir / "latest.pt"
    final_path = ckpt_dir / "final.pt"

    model.train()
    global_step = 0
    epoch = 0
    loss_accum = 0.0
    start_time = time.time()
    best_loss = float('inf')

    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        global_step = ckpt['step']
        epoch = ckpt.get('epoch', 0)
        print(f"Resumed at step {global_step}, epoch {epoch}")

    print(f"\nTraining {total_steps} steps ({args.epochs} epochs)...")
    print("=" * 70)

    while epoch < args.epochs:
        epoch += 1
        data_iter = iter(train_loader)
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        while True:
            try:
                x, t = next(data_iter)
            except StopIteration:
                break

            x, t = x.to(device), t.to(device)
            logits = model(x)
            lm_loss = F.cross_entropy(logits.reshape(-1, vs), t.reshape(-1), ignore_index=0)

            if torch.isnan(lm_loss) or torch.isinf(lm_loss):
                print(f"  NaN/Inf at step {global_step}, skipping"); continue

            optimizer.zero_grad()
            lm_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            loss_accum += lm_loss.item()

            if global_step % args.log_interval == 0:
                avg_loss = loss_accum / args.log_interval
                elapsed = time.time() - start_time
                tok_s = global_step * x.size(0) * x.size(1) / elapsed
                lr = optimizer.param_groups[0]['lr']
                ppl = math.exp(min(avg_loss, 20))
                print(f"  step {global_step:>6d}/{total_steps} | loss: {avg_loss:.4f} | ppl: {ppl:.1f} | lr: {lr:.2e} | {tok_s:.0f} tok/s")
                loss_accum = 0.0

            if global_step % args.save_interval == 0:
                tmp = ckpt_dir / "latest.tmp"
                torch.save({
                    'step': global_step, 'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, tmp)
                os.replace(tmp, latest_path)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save({'step': global_step, 'model_state_dict': model.state_dict()}, ckpt_dir / "best.pt")
                print(f"  Saved checkpoint at step {global_step}")

            if global_step >= total_steps:
                break

    torch.save({'step': global_step, 'epoch': epoch, 'model_state_dict': model.state_dict()}, final_path)
    elapsed = time.time() - start_time
    print(f"\nDone! {elapsed/3600:.1f}h, step {global_step}, best_loss={best_loss:.4f}")

if __name__ == "__main__":
    run()
