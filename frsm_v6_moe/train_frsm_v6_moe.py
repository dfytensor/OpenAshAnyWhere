"""
FRSM V6a MoE Pretraining Script
Sparse Mixture-of-Experts Content-Gated Multi-Scale State Machine
"""
import os
import sys
import time
import math
import argparse

import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 允许从工作区根目录导入 frsm / config / open_ash_voc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frsm.config import FRSMConfig
from frsm.dataset import create_dataloaders
from frsm_v6a_moe import FRSM_V6_MoE
from config import agent_voc_path
from open_ash_voc import OpenASHVoc


def get_lr_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_loss(model, x, t, vs, aux_weight, amp_dtype):
    # bf16 混合精度前向（bf16 动态范围大，无需 GradScaler）
    with torch.autocast(device_type='cuda', dtype=amp_dtype,
                        enabled=(amp_dtype is not None)):
        logits, _ = model(x, return_state=True)
    # cross_entropy 在 float32 下计算，保证数值稳定
    lm_loss = F.cross_entropy(
        logits.float().reshape(-1, vs), t.reshape(-1), ignore_index=0
    )
    total = lm_loss + aux_weight * model.aux_loss.float()
    return total, lm_loss


def train(config):
    print("=" * 60)
    print("FRSM V6a MoE Pretraining (Sparse Mixture-of-Experts)")
    print("=" * 60)
    print(f"Config: d_model={config.d_model}, num_scales={config.num_scales}")
    print(f"Config: n_experts={config.n_experts}, n_activated={config.n_activated}, "
          f"n_shared={config.n_shared}")
    print(f"Config: batch_size={config.batch_size}, max_seq_len={config.max_seq_len}")
    print(f"Config: lr={config.learning_rate}, max_steps={config.max_steps}")
    print(f"Config: amp_dtype={config.amp_dtype}, use_checkpoint={config.use_checkpoint}")

    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    vs = len(voc.token_to_id) + 1
    print(f"Vocabulary size: {vs}")

    model = FRSM_V6_MoE(
        vocab_size=vs,
        d_model=config.d_model,
        num_scales=config.num_scales,
        n_experts=config.n_experts,
        n_activated=config.n_activated,
        n_shared=config.n_shared,
        aux_loss_weight=config.aux_loss_weight,
        use_checkpoint=config.use_checkpoint,
    )

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    torch.set_float32_matmul_precision('high')  # 启用 TF32（Ampere+），加速 float32 矩阵乘
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Model parameters (total): {param_count:,}")
    print(f"Activated experts per token: {config.n_activated}/{config.n_experts}")

    train_loader = create_dataloaders(voc, mode='pretrain', config=config)

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    scheduler = get_lr_schedule(optimizer, config.warmup_steps, config.max_steps)

    ckpt_dir = config.output_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_path = ckpt_dir / "frsm_v6_moe_latest.pt"
    final_path = ckpt_dir / "frsm_v6_moe_final.pt"

    model.train()
    global_step = 0
    total_loss_accum = 0.0
    total_aux_accum = 0.0
    start_time = time.time()
    last_log_step = 0

    if latest_path.exists():
        print(f"  Resuming from {latest_path}")
        ckpt = torch.load(latest_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        global_step = ckpt['step']
        print(f"  Resumed at step {global_step}")

    print(f"\nStarting pretraining ({len(train_loader.dataset)} samples, "
          f"{config.max_steps} steps)...")
    print("-" * 60)

    data_iter = iter(train_loader)

    # Print initial loss for verification
    with torch.no_grad():
        x0, t0 = next(data_iter)
        x0 = x0.to(device, non_blocking=True)
        t0 = t0.to(device, non_blocking=True)
        init_loss, _ = compute_loss(model, x0, t0, vs, config.aux_loss_weight, config.amp_dtype)
        print(f"  Initial loss: {init_loss.item():.4f}")
    del x0, t0, init_loss
    data_iter = iter(train_loader)

    while global_step < config.max_steps:
        try:
            x, t = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, t = next(data_iter)

        x = x.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)

        total_loss, lm_loss = compute_loss(model, x, t, vs, config.aux_loss_weight, config.amp_dtype)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"  WARNING: NaN/Inf loss at step {global_step}, skipping batch")
            continue

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        global_step += 1
        total_loss_accum += lm_loss.item()
        total_aux_accum += float(model.aux_loss)

        if global_step % config.log_interval == 0 or global_step == 1:
            steps_since_log = global_step - last_log_step
            avg_loss = total_loss_accum / steps_since_log
            avg_aux = total_aux_accum / steps_since_log
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            tok_per_sec = global_step * x.size(1) * x.size(0) / elapsed if elapsed > 0 else 0
            print(f"  step {global_step:5d}/{config.max_steps} | "
                  f"loss: {avg_loss:.4f} | aux: {avg_aux:.4f} | "
                  f"lr: {lr:.2e} | {tok_per_sec:.0f} tok/s")
            total_loss_accum = 0.0
            total_aux_accum = 0.0
            last_log_step = global_step

        if global_step % max(1, config.save_interval) == 0 and global_step > 0:
            tmp_path = ckpt_dir / "frsm_v6_moe_latest.tmp"
            torch.save({
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config_d_model': config.d_model,
                'config_num_scales': config.num_scales,
                'config_n_experts': config.n_experts,
        'config_n_activated': config.n_activated,
        'config_n_shared': config.n_shared,
                'config_n_shared': config.n_shared,
            }, tmp_path)
            tmp_path.rename(latest_path)
            print(f"  Saved resume checkpoint to {latest_path}")

    torch.save({
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'config_d_model': config.d_model,
        'config_num_scales': config.num_scales,
        'config_n_experts': config.n_experts,
        'config_n_activated': config.n_activated,
    }, final_path)
    elapsed_total = time.time() - start_time
    print(f"\nPretraining complete! ({elapsed_total:.0f}s)")
    print(f"Final model saved to {final_path}")

    return model, voc


def main():
    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="FRSM V6a MoE Pretraining")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_scales", type=int, default=4, help="Scales per expert")
    parser.add_argument("--n_experts", type=int, default=16, help="Number of experts")
    parser.add_argument("--n_activated", type=int, default=4, help="Top-k experts per token")
    parser.add_argument("--n_shared", type=int, default=1,
                        help="Number of shared experts (always active, 0 to disable)")
    parser.add_argument("--aux_loss_weight", type=float, default=0.01,
                        help="Load-balancing loss weight")
    parser.add_argument("--router_noise", type=float, default=1.0,
                        help="Router Gaussian noise std (0 to disable)")
    parser.add_argument("--no_checkpoint", action="store_true",
                        help="Disable gradient checkpointing (faster but more memory)")
    parser.add_argument("--no_bf16", action="store_true",
                        help="Disable bfloat16 mixed precision (use pure float32)")
    parser.add_argument("--batch_size", type=int, default=80, help="Batch size")
    parser.add_argument("--max_seq_len", type=int, default=384, help="Max sequence length")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max training steps")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--max_lines", type=int, default=50000, help="Max pretrain lines")
    parser.add_argument("--cache_dir", type=str, default="", help="Dataset cache directory")
    parser.add_argument("--output_dir", type=str, default="frsm_v6_moe_checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--data_dir", type=str, default="minimind_data", help="Data directory")
    args = parser.parse_args()

    config = FRSMConfig(
        d_model=args.d_model,
        num_scales=args.num_scales,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        max_pretrain_lines=args.max_lines,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
    )
    config.n_experts = args.n_experts
    config.n_activated = args.n_activated
    config.n_shared = args.n_shared
    config.aux_loss_weight = args.aux_loss_weight
    config.router_noise = args.router_noise
    config.use_checkpoint = not args.no_checkpoint
    config.amp_dtype = None if args.no_bf16 else torch.bfloat16

    train(config)


if __name__ == "__main__":
    main()
