"""
FRSM Pretraining Script
使用 OpenASHVoc 词表进行分形递归状态机预训练。
"""
import os
import sys
import time
import math
import argparse

import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frsm.config import FRSMConfig
from frsm.model import FractalRecursiveStateMachine
from frsm.dataset import create_dataloaders
from config import agent_voc_path
from open_ash_voc import OpenASHVoc


def get_lr_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_loss(model, x, t, vs, compute_critical=True):
    logits, final_states, critical_loss = model(
        x, return_state=True, compute_critical_loss=compute_critical
    )
    lm_loss = F.cross_entropy(
        logits.reshape(-1, vs), t.reshape(-1), ignore_index=0
    )
    total_loss = lm_loss + model.critical_reg_coeff * critical_loss
    return total_loss, lm_loss, critical_loss


def train(config):
    print("=" * 60)
    print("FRSM Pretraining")
    print("=" * 60)
    print(f"Config: d_model={config.d_model}, num_scales={config.num_scales}")
    print(f"Config: batch_size={config.batch_size}, max_seq_len={config.max_seq_len}")
    print(f"Config: lr={config.learning_rate}, max_steps={config.max_steps}")

    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    vs = len(voc.token_to_id) + 1
    print(f"Vocabulary size: {vs}")

    model = FractalRecursiveStateMachine(
        vocab_size=vs,
        d_model=config.d_model,
        num_scales=config.num_scales,
        expansion_factor=config.expansion_factor,
        spectral_radius_target=config.spectral_radius_target,
    )
    model.critical_reg_coeff = config.critical_reg_coeff

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Model parameters: {param_count:,}")

    train_loader = create_dataloaders(voc, mode='pretrain', config=config)

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    scheduler = get_lr_schedule(optimizer, config.warmup_steps, config.max_steps)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    global_step = 0
    total_loss_accum = 0.0
    total_lm_loss_accum = 0.0
    total_crit_loss_accum = 0.0
    best_loss = float('inf')
    start_time = time.time()

    # 断点续接
    resume_path = config.output_dir / "frsm_pretrain_latest.pt"
    if resume_path.exists():
        print(f"  Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        global_step = ckpt['step']
        print(f"  Resumed at step {global_step}")

    print(f"\nStarting pretraining ({len(train_loader.dataset)} samples, {config.max_steps} steps)...")
    print("-" * 60)

    data_iter = iter(train_loader)

    while global_step < config.max_steps:
        try:
            x, t = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, t = next(data_iter)

        x = x.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)

        total_loss, lm_loss, crit_loss = compute_loss(
            model, x, t, vs, compute_critical=True
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        global_step += 1
        total_loss_accum += total_loss.item()
        total_lm_loss_accum += lm_loss.item()
        total_crit_loss_accum += crit_loss.item()

        if global_step % config.log_interval == 0 or global_step == 1:
            avg_loss = total_loss_accum / config.log_interval
            avg_lm_loss = total_lm_loss_accum / config.log_interval
            avg_crit_loss = total_crit_loss_accum / config.log_interval
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            tok_per_sec = global_step * x.size(1) / elapsed
            print(f"  step {global_step:5d}/{config.max_steps} | "
                  f"loss: {avg_loss:.4f} | lm: {avg_lm_loss:.4f} | "
                  f"crit: {avg_crit_loss:.6f} | lr: {lr:.2e} | "
                  f"{tok_per_sec:.0f} tok/s")
            total_loss_accum = 0.0
            total_lm_loss_accum = 0.0
            total_crit_loss_accum = 0.0

        if global_step % config.save_interval == 0 and global_step > 0:
            save_path = config.output_dir / f"frsm_pretrain_step{global_step}.pt"
            latest_path = config.output_dir / "frsm_pretrain_latest.pt"
            state = {
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config_d_model': config.d_model,
                'config_num_scales': config.num_scales,
            }
            torch.save(state, save_path)
            torch.save(state, latest_path)
            print(f"  Saved checkpoint to {save_path}")

    final_path = config.output_dir / "frsm_pretrain_final.pt"
    torch.save({
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'config_d_model': config.d_model,
        'config_num_scales': config.num_scales,
    }, final_path)
    elapsed_total = time.time() - start_time
    print(f"\nPretraining complete! ({elapsed_total:.0f}s)")
    print(f"Final model saved to {final_path}")

    return model, voc


def main():
    parser = argparse.ArgumentParser(description="FRSM Pretraining")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_scales", type=int, default=4, help="Number of temporal scales")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_seq_len", type=int, default=384, help="Max sequence length")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max training steps")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--max_lines", type=int, default=50000, help="Max pretrain lines to load")
    args = parser.parse_args()

    config = FRSMConfig(
        d_model=args.d_model,
        num_scales=args.num_scales,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        max_pretrain_lines=args.max_lines,
    )

    train(config)


if __name__ == "__main__":
    main()
