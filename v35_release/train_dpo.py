import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from open_ash import OpenASH
from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from open_ash_dataset import DPODataset
from trainer_utils import (
    get_lr, Logger, is_main_process, open_ash_checkpoint,
    init_distributed_mode, setup_seed, SkipBatchSampler
)

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def get_sequence_log_probs(model, input_ids, target_ids, device_type='cuda'):
    ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if device_type == 'cuda' else nullcontext()
    with ctx:
        outputs, _ = model(input_ids)
    log_probs = F.log_softmax(outputs, dim=-1)
    per_token_logps = log_probs.gather(-1, target_ids.clamp(0).unsqueeze(-1)).squeeze(-1)
    mask = (target_ids != 0).float()
    return (per_token_logps * mask).sum(dim=-1)


def train_epoch(epoch, model, ref_model, loader, optimizer, scaler, args,
                voc_size, hidden_size, num_layers, beta=0.1, start_step=0):
    model.train()
    ref_model.eval()
    device_type = "cuda" if "cuda" in args.device else "cpu"
    start_time = time.time()
    iters = len(loader)
    avg_loss = 0
    avg_chosen_rw = 0
    avg_rejected_rw = 0
    steps = 0

    bar = tqdm(loader, initial=start_step, total=iters, desc=f'Epoch {epoch + 1}/{args.epochs}')
    last_step = start_step

    for step, batch in enumerate(bar, start=start_step + 1):
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        chosen_input, chosen_target, rejected_input, rejected_target = batch
        chosen_input = chosen_input.to(args.device, non_blocking=True)
        chosen_target = chosen_target.to(args.device, non_blocking=True)
        rejected_input = rejected_input.to(args.device, non_blocking=True)
        rejected_target = rejected_target.to(args.device, non_blocking=True)

        if chosen_input.numel() == 0 or chosen_input.size(0) == 0:
            continue

        autocast_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if device_type == 'cuda' else nullcontext()
        with autocast_ctx:
            policy_chosen_logps = get_sequence_log_probs(model, chosen_input, chosen_target, device_type)
            policy_rejected_logps = get_sequence_log_probs(model, rejected_input, rejected_target, device_type)

            with torch.no_grad():
                ref_chosen_logps = get_sequence_log_probs(ref_model, chosen_input, chosen_target, device_type)
                ref_rejected_logps = get_sequence_log_probs(ref_model, rejected_input, rejected_target, device_type)

            chosen_rewards = policy_chosen_logps - ref_chosen_logps
            rejected_rewards = policy_rejected_logps - ref_rejected_logps

            loss = -F.logsigmoid(beta * (chosen_rewards - rejected_rewards)).mean()
            loss = loss / args.accumulation_steps

        steps += 1
        avg_loss += loss.item()
        avg_chosen_rw += chosen_rewards.mean().item()
        avg_rejected_rw += rejected_rewards.mean().item()
        bar.set_description(
            "loss: {:.4f} c_rw: {:.4f} r_rw: {:.4f}".format(
                avg_loss / steps, avg_chosen_rw / steps, avg_rejected_rw / steps))

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            torch.cuda.empty_cache()
            spend_time = time.time() - start_time
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'lr: {current_lr:.8f}, loss: {avg_loss / steps:.4f}, '
                f'chosen_rw: {avg_chosen_rw / steps:.4f}, rejected_rw: {avg_rejected_rw / steps:.4f}, '
                f'eta: {eta_min:.1f}min')

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            ckp = f'{args.save_dir}/{args.save_weight}_{hidden_size}_{num_layers}.pth'
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            open_ash_checkpoint(voc_size, hidden_size, num_layers,
                                weight=args.save_weight, model=model, optimizer=optimizer,
                                scaler=scaler, epoch=epoch, step=step, save_dir='../checkpoints')
            model.train()

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenASH DPO Training")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='dpo', type=str)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument('--max_seq_len', default=768, type=int)
    parser.add_argument("--data_path", type=str, default="dpo.jsonl")
    parser.add_argument('--from_weight', default='full_sft', type=str)
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    parser.add_argument("--voc_size", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)

    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    voc_size = len(voc.token_to_id) + 1
    hidden_size = args.hidden_size if args.hidden_size else 768
    num_layers = args.num_layers if args.num_layers else 12
    num_heads = args.num_heads if args.num_heads else 8
    Logger(f'DPO Training: Voc={voc_size}, Hidden={hidden_size}, Layers={num_layers}, Heads={num_heads}')

    ckp_data = open_ash_checkpoint(voc_size, hidden_size, num_layers,
                                   weight=args.save_weight,
                                   save_dir='../checkpoints') if args.from_resume == 1 else None

    weight_path = f'{args.save_dir}/{args.from_weight}_{hidden_size}_{num_layers}.pth'
    Logger(f'Loading SFT weights: {weight_path}')

    model = OpenASH(voc_size=voc_size, hidden_size=hidden_size,
                    num_heads=num_heads, num_layers=num_layers)
    model.load_state_dict(torch.load(weight_path, map_location=args.device), strict=False)
    model.to(args.device)

    ref_model = OpenASH(voc_size=voc_size, hidden_size=hidden_size,
                        num_heads=num_heads, num_layers=num_layers)
    ref_model.load_state_dict(torch.load(weight_path, map_location=args.device), strict=False)
    ref_model.to(args.device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f'Trainable Params: {params:,}')

    train_ds = DPODataset(args.data_path, voc)
    Logger(f'Dataset: {len(train_ds)} samples from {args.data_path}')

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        if ckp_data.get('scaler'):
            scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=4,
                            pin_memory=True, persistent_workers=True, prefetch_factor=4,
                            collate_fn=train_ds.dpo_padding_func)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: skip {start_step} steps')
            train_epoch(epoch, model, ref_model, loader, optimizer, scaler, args,
                        voc_size, hidden_size, num_layers, args.beta, start_step)
        else:
            train_epoch(epoch, model, ref_model, loader, optimizer, scaler, args,
                        voc_size, hidden_size, num_layers, args.beta, 0)

    if is_main_process():
        model.eval()
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        ckp = f'{args.save_dir}/{args.save_weight}_{hidden_size}_{num_layers}.pth'
        state_dict = raw_model.state_dict()
        torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
        Logger(f'Final model saved to {ckp}')

    if dist.is_initialized():
        dist.destroy_process_group()
