import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import json
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from open_ash import OpenASH
from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from open_ash_dataset import SFTDataset, split_sequence
from trainer_utils import (
    get_lr, Logger, is_main_process, open_ash_checkpoint,
    init_distributed_mode, setup_seed, SkipBatchSampler
)

warnings.filterwarnings('ignore')

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def train_epoch(epoch, model, loader, optimizer, scaler, criterion, args, voc_size, hidden_size, num_layers,
                start_step=0):
    start_time = time.time()
    model.train()
    iters = len(loader)
    last_step = start_step

    bar = tqdm(loader, initial=start_step, total=iters, desc=f'Epoch {epoch + 1}/{args.epochs}')
    avg_loss = 0
    steps = 0
    for step, batch_tokens in enumerate(bar, start=start_step + 1):
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        inputs, targets = batch_tokens

        inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        targets = pad_sequence(targets, batch_first=True, padding_value=0)
        # seq_s = 0
        state = None
        #
        # chunks = split_sequence(inputs.shape[-1], args.max_seq_len)
        # for chunk_idx, seq_len in enumerate(chunks):
        #     if state is not None:
        #         state = [i.detach() for i in state]
        data_input_one = inputs[:, :args.max_seq_len]
        data_target_one = targets[:, :args.max_seq_len]
        # seq_s += seq_len

        inputs_one = data_input_one.to(args.device, non_blocking=True)
        targets_one = data_target_one.to(args.device, non_blocking=True)

        if inputs_one.numel() == 0 or inputs_one.size(0) == 0:
            continue

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs, state = model(inputs_one, state=state)
            B, S, V = outputs.size()
            outputs = outputs.view(B * S, V)
            targets_flat = targets_one.view(B * S)
            loss = criterion(outputs, targets_flat) / args.accumulation_steps
        del outputs
        steps += 1
        avg_loss += loss.item()
        bar.set_description("loss: {:.4f}".format(avg_loss / steps))


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
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')

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
    parser = argparse.ArgumentParser(description="OpenASH Full SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=20, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="chunk最大序列长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="兼容参数（未使用）")
    # parser.add_argument('--num_hidden_layers', default=8, type=int, help="兼容参数（自动计算）")
    # parser.add_argument('--hidden_size', default=512, type=int, help="兼容参数（自动计算）")
    parser.add_argument("--data_path", type=str,
                        default=r"sft_t2t.jsonl", help="SFT数据路径")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速")
    parser.add_argument("--voc_size", type=int, default=None, help="词表大小")
    parser.add_argument("--hidden_size", type=int, default=None, help="隐藏层大小")
    parser.add_argument("--num_layers", type=int, default=None, help="层数")
    parser.add_argument("--num_heads", type=int, default=None, help="注意力头数")
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
    Logger(f'Voc: {voc_size}, Hidden: {hidden_size}, Layers: {num_layers}, Heads: {num_heads}')

    ckp_data = open_ash_checkpoint(voc_size, hidden_size, num_layers,
                                   weight=args.save_weight,
                                   save_dir='../checkpoints') if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=torch.bfloat16)

    model = OpenASH(voc_size=voc_size, hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers)
    if args.from_weight != 'none':
        weight_path = f'{args.save_dir}/{args.from_weight}_{hidden_size}_{num_layers}.pth'
        print(weight_path)
        model.load_state_dict(torch.load(weight_path, map_location=args.device), strict=False)

    params = sum(p.numel() for p in model.parameters() if p.shape != torch.Size([]))
    Logger(f'Params: {params:,}')
    # model.load_state_dict(torch.load("../out/full_sft_768_12.pth", map_location=args.device), strict=False)
    model.to(args.device)


    train_ds = SFTDataset(args.data_path, voc, )

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
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
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=16, pin_memory=True,
                            persistent_workers=True, prefetch_factor=6, collate_fn=train_ds.sft_padding_func)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, model, loader, optimizer, scaler, criterion, args, voc_size, hidden_size, num_layers,
                        start_step)
        else:
            train_epoch(epoch, model, loader, optimizer, scaler, criterion, args, voc_size, hidden_size, num_layers, 0)

    if dist.is_initialized():
        dist.destroy_process_group()
