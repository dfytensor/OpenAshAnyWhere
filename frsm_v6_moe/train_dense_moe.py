"""
FRSM V6 Dense MoE — 全量预训练
使用 minimind_data/pretrain_t2t_mini.jsonl 全部数据训练 3 个 epoch
"""
import os, sys, time, math, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys as _sys; _sys.stdout.reconfigure(line_buffering=True)

import torch, torch.nn.functional as F
from torch.optim import AdamW
from frsm.config import FRSMConfig
from frsm.dataset import create_dataloaders
from frsm_v6a_dense_moe import FRSM_V6_DenseMoE
from config import agent_voc_path
from open_ash_voc import OpenASHVoc
from pathlib import Path
import json, pickle

torch.set_float32_matmul_precision('high')

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_experts", type=int, default=16)
    parser.add_argument("--n_shared", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--chunk_size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_lines", type=int, default=1270238)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="dense_moe_pretrain")
    parser.add_argument("--data_dir", type=str, default="../minimind_data")
    parser.add_argument("--top_k", type=int, default=4, help="soft top-k routing (0=all experts)")
    parser.add_argument("--chunk_pattern", type=str, default="", help="mixed chunk e.g. 1,4,4,4,4")
    parser.add_argument("--chunk_wave", type=str, default="", help="triangular wave e.g. 1,16")
    args = parser.parse_args()

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    vs = len(voc.token_to_id) + 1
    print(f"Vocab size: {vs}")

    cp = [int(x) for x in args.chunk_pattern.split(',')] if args.chunk_pattern else None
    cw = tuple(int(x) for x in args.chunk_wave.split(',')) if args.chunk_wave else None
    model = FRSM_V6_DenseMoE(
        vocab_size=vs, d_model=args.d_model, num_scales=4,
        n_experts=args.n_experts, n_shared=args.n_shared,
        chunk_size=args.chunk_size, top_k=(args.top_k if args.top_k>0 else None),
        chunk_pattern=cp, chunk_wave=cw
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} ({param_count/1e6:.1f}M)")

    config = FRSMConfig(
        d_model=args.d_model, num_scales=4,
        batch_size=args.batch_size, max_seq_len=args.max_seq_len,
        max_steps=args.epochs * (args.max_lines // args.batch_size + 1),
        learning_rate=args.lr,
        max_pretrain_lines=args.max_lines,
        output_dir=args.output_dir, data_dir=args.data_dir,
    )
    config.n_experts = args.n_experts
    config.n_shared = args.n_shared
    config.chunk_size = args.chunk_size

    steps_per_epoch = (args.max_lines // args.batch_size) + 1
    total_steps = steps_per_epoch * args.epochs
    print(f"Batch: {args.batch_size}, Seq: {args.max_seq_len}, Chunk: {args.chunk_size or 'auto'}")
    print(f"Lines: {args.max_lines}, Steps/epoch: {steps_per_epoch}, Total: {total_steps}")
    print(f"Epochs: {args.epochs}")

    # 缓存 tokenized 数据集
    data_dir_abs = Path(args.data_dir).resolve()
    cache_file = data_dir_abs / f"pretrain_cached_{args.max_lines}_{args.max_seq_len}.pt"
    if cache_file.exists():
        print(f"Loading cached dataset from {cache_file}")
        cached = torch.load(cache_file)
        from frsm.dataset import PretrainDataset
        dataset = PretrainDataset.__new__(PretrainDataset)
        dataset.max_len = args.max_seq_len
        dataset.data = cached
        from torch.utils.data import DataLoader
        from torch.nn.utils.rnn import pad_sequence
        train_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=0, collate_fn=PretrainDataset.collate_fn, drop_last=True,
        )
        print(f"Loaded {len(dataset.data)} samples from cache")
    else:
        print("No cache found, loading dataset (this may take a while)...")
        train_loader = create_dataloaders(voc, mode='pretrain', config=config)
        # 缓存 tokenized 结果
        dataset = train_loader.dataset
        torch.save(dataset.data, cache_file)
        print(f"Dataset cached to {cache_file}")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    lr_schedule = lambda s: min(s / 500, 1.0) if s < 500 else max(0, 0.5*(1+math.cos(math.pi*(s-500)/(total_steps-500))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_path = ckpt_dir / "latest.pt"
    final_path = ckpt_dir / "final.pt"

    model.train()
    global_step = 0
    epoch = 0
    loss_accum = 0.0
    aux_accum = 0.0
    start_time = time.time()
    best_loss = float('inf')

    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        global_step = ckpt['step']
        epoch = ckpt.get('epoch', 0)
        print(f"Resumed at step {global_step}, epoch {epoch}")

    print(f"\nTraining {total_steps} steps ({args.epochs} epochs)...")
    print("="*70)

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
            lm_loss = model(x, targets=t)
            loss = lm_loss + 0.01 * model.aux_loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  NaN/Inf at step {global_step}, skipping"); continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            loss_accum += lm_loss.item()
            aux_accum += float(model.aux_loss.detach())

            if global_step % args.log_interval == 0:
                avg_loss = loss_accum / args.log_interval
                avg_aux = aux_accum / args.log_interval
                elapsed = time.time() - start_time
                tok_s = global_step * x.size(0) * x.size(1) / elapsed
                lr = optimizer.param_groups[0]['lr']
                print(f"  step {global_step:>6d}/{total_steps} | loss: {avg_loss:.4f} | aux: {avg_aux:.4f} | lr: {lr:.2e} | {tok_s:.0f} tok/s")
                loss_accum = 0.0
                aux_accum = 0.0

            if global_step % args.save_interval == 0:
                tmp = ckpt_dir / "latest.tmp"
                torch.save({
                    'step': global_step, 'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, tmp)
                import os as _os; _os.replace(tmp, latest_path)
                if 'avg_loss' in dir() and avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save({'step': global_step, 'model_state_dict': model.state_dict()}, ckpt_dir / "best.pt")

            if global_step >= total_steps:
                break

    torch.save({'step': global_step, 'epoch': epoch, 'model_state_dict': model.state_dict()}, final_path)
    elapsed = time.time() - start_time
    print(f"\nDone! {elapsed/3600:.1f}h, step {global_step}")

if __name__ == "__main__":
    run()
