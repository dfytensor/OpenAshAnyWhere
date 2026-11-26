import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from open_ash import OpenASH
from open_ash_voc import OpenASHVoc
from config import agent_voc_path


def get_open_ash_config(voc_size):
    num_layers = 8
    for i in range(16):
        if voc_size > 8192 * 2 ** i:
            num_layers = 8 * i
    if num_layers == 0:
        num_layers = 8
    hidden_size = 2 ** 6 * num_layers
    num_heads = num_layers
    return voc_size, hidden_size, num_heads, num_layers


def get_model_params(model):
    params = sum(p.numel() for p in model.parameters() if p.shape != torch.Size([]))
    print(f'Model Params: {params:,}')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def open_ash_checkpoint(voc_size, hidden_size, num_layers, weight='pretrain', model=None, optimizer=None, scaler=None, epoch=0, step=0, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    ckp_path = f'{save_dir}/{weight}_{hidden_size}_{num_layers}.pth'
    resume_path = f'{save_dir}/{weight}_{hidden_size}_{num_layers}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler else None,
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'voc_size': voc_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu',weights_only=False)
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_open_ash_model(voc_size, from_weight='none', save_dir='../out', device='cuda'):
    voc_size, hidden_size, num_heads, num_layers = get_open_ash_config(voc_size)
    model = OpenASH(
        voc_size=voc_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers
    )
    if from_weight != 'none':
        weight_path = f'{save_dir}/{from_weight}_{hidden_size}_{num_layers}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
    get_model_params(model)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    return model.to(device), hidden_size, num_layers


def init_voc():
    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    voc_size = len(voc.token_to_id) + 1
    Logger(f'Voc size: {voc_size}')
    return voc, voc_size


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
