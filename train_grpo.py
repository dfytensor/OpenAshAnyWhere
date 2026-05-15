import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import json
import random
import warnings
import math
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from open_ash import OpenASH
from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from open_ash_dataset import pre_processing_chat
from trainer_utils import (
    get_lr, Logger, is_main_process, open_ash_checkpoint,
    init_distributed_mode, setup_seed
)

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class GRPODataset(Dataset):
    def __init__(self, jsonl_path, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        with open(jsonl_path, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        self.samples = []
        for line in raw_lines:
            data = json.loads(line)
            convs = data.get("conversations", [])
            gt = data.get("gt", [])
            if not convs:
                continue
            last = convs[-1]
            if last.get("role") == "assistant" and (not last.get("content") or last.get("content") == ""):
                prompt_convs = convs[:-1]
            else:
                prompt_convs = convs
            if not prompt_convs:
                continue
            self.samples.append({"conversations": prompt_convs, "gt": gt})

        self.im_start = self.tokenizer.token_to_id.get("<|im_start|>")
        self.im_end = self.tokenizer.token_to_id.get("<|im_end|>")
        self.think_start = self.tokenizer.token_to_id.get("<|think|>")
        self.think_end = self.tokenizer.token_to_id.get("<|end_think|>")
        self.tool_call = self.tokenizer.token_to_id.get("special_token_tc")
        self.tool_call_end = self.tokenizer.token_to_id.get("special_token_tce")
        self.tools_tag = self.tokenizer.token_to_id.get("<|tools|>")
        self.tools_end_tag = self.tokenizer.token_to_id.get("<|end_tools|>")

        _sp = self._get_sp()
        self.user_tok = _sp["user"]
        self.agent_tok = _sp["agent"]
        self.system_tok = _sp["system"]

    def _get_sp(self):
        return {
            "user": self.tokenizer.token_to_id.get("user"),
            "agent": self.tokenizer.token_to_id.get("<|agent|>"),
            "system": self.tokenizer.token_to_id.get("system"),
        }

    def __len__(self):
        return len(self.samples)

    def _encode_role(self, role):
        if role == "user":
            return self.user_tok
        elif role == "assistant":
            return self.agent_tok
        elif role == "system":
            return self.system_tok
        return None

    def create_chat_prompt(self, conversations):
        messages = []
        for message in conversations:
            message = dict(message)
            role = message.get("role")
            role_tok = self._encode_role(role)
            if role_tok is None:
                continue
            messages += [self.im_start, role_tok]
            if role == "system":
                if message.get("content") != "":
                    messages += self.tokenizer.encode(message.get("content"))
                if "tools" in message:
                    messages += (
                        [self.tools_tag]
                        + self.tokenizer.encode(message.get("tools"))
                        + [self.tools_end_tag]
                    )
            elif role == "user":
                messages += self.tokenizer.encode(message.get("content"))
            elif role == "assistant":
                if "reasoning_content" in message and message["reasoning_content"]:
                    messages += (
                        [self.think_start]
                        + self.tokenizer.encode(message["reasoning_content"])
                        + [self.think_end]
                    )
                if "tool_calls" in message:
                    messages += (
                        [self.tool_call]
                        + self.tokenizer.encode(message["tool_calls"])
                        + [self.tool_call_end]
                    )
                if message.get("content") != "":
                    messages += self.tokenizer.encode(message.get("content"))
            messages += [self.im_end]
        return messages

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample["conversations"])
        prompt_ids = self.create_chat_prompt(conversations)
        gt = sample["gt"]
        return torch.tensor(prompt_ids, dtype=torch.long), gt

    def grpo_collate_fn(self, items):
        prompts = [item[0] for item in items]
        gt_list = [item[1] for item in items]
        padded_prompts = pad_sequence(prompts, batch_first=True, padding_value=0)
        return padded_prompts, gt_list


@torch.no_grad()
def batch_generate(model, prompt_ids, max_new_tokens, tokenizer, temperature=0.7,
                   top_k=30, top_p=0.85, repetition_penalty=1.35, repetition_window=64):
    model.eval()
    device = prompt_ids.device
    B = prompt_ids.size(0)

    sp = {
        "im_end": tokenizer.token_to_id.get("<|im_end|>"),
        "pad": tokenizer.token_to_id.get("<|pad|>"),
    }
    stop_ids = {sp["im_end"], sp["pad"]}

    outputs, states = model(prompt_ids, state=None)
    logits = outputs[:, -1, :]

    all_ids = []
    all_logps = []
    done = torch.zeros(B, dtype=torch.bool, device=device)

    for step in range(max_new_tokens):
        logits_curr = logits.clone()

        if repetition_penalty > 1.0:
            for b in range(B):
                if done[b]:
                    continue
                recent_ids = all_ids[-repetition_window:] if all_ids else []
                if recent_ids:
                    recent = set()
                    for prev in recent_ids:
                        recent.add(prev[b].item())
                    for tid in recent:
                        if tid < logits_curr.size(1):
                            if logits_curr[b, tid] > 0:
                                logits_curr[b, tid] /= repetition_penalty
                            else:
                                logits_curr[b, tid] *= repetition_penalty

        if temperature > 1e-6:
            logits_curr = logits_curr / temperature

        if top_k is not None and top_k > 0:
            top_k_val = min(top_k, logits_curr.size(-1))
            topk_vals, topk_idx = torch.topk(logits_curr, top_k_val)
            filtered = torch.full_like(logits_curr, float('-inf'))
            filtered.scatter_(1, topk_idx, topk_vals)
            logits_curr = filtered

        if top_p is not None and 0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits_curr, descending=True)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove_mask = cumulative_probs > top_p
            remove_mask[:, 1:] = remove_mask[:, :-1].clone()
            remove_mask[:, 0] = False
            sorted_logits[remove_mask] = float('-inf')
            logits_curr.scatter_(1, sorted_idx, sorted_logits)

        log_probs = F.log_softmax(logits_curr, dim=-1)

        if temperature < 1e-6:
            next_tokens = logits_curr.argmax(dim=-1)
        else:
            probs = torch.softmax(logits_curr, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        token_logps = log_probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)
        token_logps[done] = 0.0

        newly_done = torch.zeros(B, dtype=torch.bool, device=device)
        for sid in stop_ids:
            if sid is not None:
                newly_done |= (next_tokens == sid)
        newly_done &= ~done
        done |= newly_done

        next_tokens[done] = 0
        all_ids.append(next_tokens)
        all_logps.append(token_logps)

        if done.all():
            break

        next_input = next_tokens.unsqueeze(1)
        outputs, states = model(next_input, state=states)
        states = [s.detach() for s in states]
        logits = outputs[:, -1, :]

    if len(all_ids) == 0:
        max_len = 1
        gen_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
        gen_logps = torch.zeros(B, max_len, dtype=torch.float, device=device)
    else:
        gen_ids = torch.stack(all_ids, dim=1)
        gen_logps = torch.stack(all_logps, dim=1)

    return gen_ids, gen_logps


def compute_rewards(gen_ids, tokenizer, gt_list):
    B = gen_ids.size(0)
    rewards = torch.zeros(B, dtype=torch.float)

    for i in range(B):
        valid_ids = gen_ids[i][gen_ids[i] != 0].tolist()
        if not valid_ids:
            rewards[i] = 0.0
            continue

        try:
            text = tokenizer.decode(valid_ids)
        except (TypeError, KeyError):
            text = ""
        gt = gt_list[i] if i < len(gt_list) else []
        r = 0.0

        if gt and len(gt) > 0:
            for gt_ans in gt:
                if gt_ans and str(gt_ans) in text:
                    r += 2.0
                    break

        if len(valid_ids) > 5:
            r += 0.2
        elif len(valid_ids) > 2:
            r += 0.1

        r += min(len(valid_ids) / 100.0, 0.3)

        rewards[i] = r

    return rewards


def compute_grpo_token_logps(model, prompt_ids, gen_ids, device_type='cuda'):
    B, S_p = prompt_ids.shape
    S_g = gen_ids.size(1)

    full_ids = torch.cat([prompt_ids, gen_ids], dim=1)

    ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if device_type == 'cuda' else nullcontext()
    with ctx:
        outputs, _ = model(full_ids)

    gen_logits = outputs[:, S_p - 1:S_p - 1 + S_g, :]
    gen_log_probs = F.log_softmax(gen_logits, dim=-1)
    gen_targets = full_ids[:, S_p:S_p + S_g]
    token_logps = gen_log_probs.gather(-1, gen_targets.clamp(0).unsqueeze(-1)).squeeze(-1)

    gen_mask = (gen_ids != 0).float()
    return token_logps, gen_mask


def grpo_loss_fn(new_token_logps, old_token_logps, ref_token_logps,
                 advantages, gen_mask, clip_eps=0.2, beta_kl=0.01):
    ratio = torch.exp(new_token_logps - old_token_logps)
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

    adv_expanded = advantages.unsqueeze(-1)

    surr1 = ratio * adv_expanded
    surr2 = clipped_ratio * adv_expanded
    policy_loss = -torch.min(surr1, surr2)

    kl_penalty = beta_kl * (new_token_logps - ref_token_logps)

    total_loss = (policy_loss + kl_penalty) * gen_mask
    loss = total_loss.sum() / gen_mask.sum().clamp(min=1)

    with torch.no_grad():
        approx_kl = ((new_token_logps - ref_token_logps) * gen_mask).sum() / gen_mask.sum().clamp(min=1)
        clip_ratio = ((ratio - 1.0).abs() > clip_eps).float()
        clip_frac = (clip_ratio * gen_mask).sum() / gen_mask.sum().clamp(min=1)

    return loss, approx_kl.item(), clip_frac.item()


def train_grpo(args, model, ref_model, dataset, optimizer, scaler, tokenizer,
               voc_size, hidden_size, num_layers):
    device_type = "cuda" if "cuda" in args.device else "cpu"
    model.train()
    ref_model.eval()

    sp = {
        "im_end": tokenizer.token_to_id.get("<|im_end|>"),
        "pad": tokenizer.token_to_id.get("<|pad|>"),
        "im_start": tokenizer.token_to_id.get("<|im_start|>"),
        "agent": tokenizer.token_to_id.get("<|agent|>"),
    }

    total_steps = args.epochs * (len(dataset) // args.batch_size)
    global_step = 0

    for epoch in range(args.epochs):
        setup_seed(42 + epoch)
        indices = torch.randperm(len(dataset)).tolist()
        epoch_loss = 0
        epoch_steps = 0
        start_time = time.time()

        pbar = tqdm(range(0, len(indices), args.batch_size), desc=f'Epoch {epoch + 1}/{args.epochs}')

        for batch_start in pbar:
            batch_indices = indices[batch_start:batch_start + args.batch_size]
            B = len(batch_indices)
            if B < 2:
                continue

            batch_prompts = []
            batch_gt = []
            for idx in batch_indices:
                prompt_ids, gt = dataset[idx]
                batch_prompts.append(prompt_ids)
                batch_gt.append(gt)

            prompt_tensor = pad_sequence(batch_prompts, batch_first=True, padding_value=0).to(args.device)
            S_p = prompt_tensor.size(1)

            if S_p >= args.max_seq_len - 20:
                continue

            expanded_prompts = prompt_tensor.repeat_interleave(args.num_generations, dim=0)
            expanded_gt = []
            for gt in batch_gt:
                expanded_gt.extend([gt] * args.num_generations)
            BG = expanded_prompts.size(0)

            max_gen_tokens = min(args.max_gen_len, args.max_seq_len - S_p - 1)

            gen_ids, old_logps = batch_generate(
                model, expanded_prompts,
                max_new_tokens=max_gen_tokens,
                tokenizer=tokenizer,
                temperature=args.gen_temperature,
                top_k=args.gen_top_k,
                top_p=args.gen_top_p,
                repetition_penalty=args.gen_repetition_penalty,
            )

            rewards = compute_rewards(gen_ids, tokenizer, expanded_gt)
            rewards = rewards.to(args.device)

            rewards_grouped = rewards.reshape(B, args.num_generations)
            mean_r = rewards_grouped.mean(dim=1, keepdim=True)
            std_r = rewards_grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
            advantages = ((rewards_grouped - mean_r) / std_r).reshape(BG)

            for ppo_epoch in range(args.num_ppo_epochs):
                new_token_logps, gen_mask = compute_grpo_token_logps(
                    model, expanded_prompts, gen_ids, device_type)

                with torch.no_grad():
                    ref_token_logps, _ = compute_grpo_token_logps(
                        ref_model, expanded_prompts, gen_ids, device_type)

                old_token_logps_detached = old_logps.detach()

                loss, approx_kl, clip_frac = grpo_loss_fn(
                    new_token_logps, old_token_logps_detached, ref_token_logps,
                    advantages, gen_mask,
                    clip_eps=args.clip_eps,
                    beta_kl=args.beta_kl,
                )

                if approx_kl > args.target_kl:
                    Logger(f'  KL {approx_kl:.4f} > target_kl {args.target_kl}, skip PPO update')
                    break

                loss = loss / args.accumulation_steps

                scaler.scale(loss).backward()

                if (ppo_epoch + 1) % args.accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item()
                epoch_steps += 1
                global_step += 1

            avg_loss = epoch_loss / max(epoch_steps, 1)
            avg_reward = rewards.mean().item()
            pbar.set_description(
                f"loss: {avg_loss:.4f} reward: {avg_reward:.2f} kl: {approx_kl:.4f}")

            if global_step % args.log_interval == 0:
                spend_time = time.time() - start_time
                current_lr = optimizer.param_groups[-1]['lr']
                Logger(
                    f'[GRPO] Epoch:{epoch + 1}/{args.epochs} step:{global_step}, '
                    f'lr: {current_lr:.8f}, loss: {avg_loss:.4f}, '
                    f'reward: {avg_reward:.2f}, kl: {approx_kl:.4f}, '
                    f'clip_frac: {clip_frac:.3f}')

            if global_step % args.save_interval == 0 and is_main_process():
                model.eval()
                raw_model = model.module if isinstance(model, DistributedDataParallel) else model
                raw_model = getattr(raw_model, '_orig_mod', raw_model)
                ckp = f'{args.save_dir}/{args.save_weight}_{hidden_size}_{num_layers}.pth'
                state_dict = raw_model.state_dict()
                torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
                Logger(f'Checkpoint saved to {ckp}')
                model.train()

            torch.cuda.empty_cache()

    if is_main_process():
        model.eval()
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        ckp = f'{args.save_dir}/{args.save_weight}_{hidden_size}_{num_layers}.pth'
        state_dict = raw_model.state_dict()
        torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
        Logger(f'Final model saved to {ckp}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenASH GRPO Training")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='grpo', type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--max_seq_len', default=1024, type=int)
    parser.add_argument("--data_path", type=str, default="agent_rl.jsonl",
                        help="GRPO data: agent_rl.jsonl or rlaif.jsonl")
    parser.add_argument('--from_weight', default='full_sft', type=str)
    parser.add_argument("--voc_size", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)

    parser.add_argument("--num_generations", type=int, default=4,
                        help="G: completions per prompt for GRPO")
    parser.add_argument("--num_ppo_epochs", type=int, default=2,
                        help="PPO update epochs per generation batch")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--beta_kl", type=float, default=0.3, help="KL penalty coefficient")
    parser.add_argument("--target_kl", type=float, default=0.2, help="Early stop KL threshold")
    parser.add_argument("--max_gen_len", type=int, default=256,
                        help="Max generation length per response")
    parser.add_argument("--gen_temperature", type=float, default=0.7)
    parser.add_argument("--gen_top_k", type=int, default=30)
    parser.add_argument("--gen_top_p", type=float, default=0.85)
    parser.add_argument("--gen_repetition_penalty", type=float, default=1.2)
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
    Logger(f'GRPO Training: Voc={voc_size}, Hidden={hidden_size}, Layers={num_layers}, Heads={num_heads}')
    Logger(f'G={args.num_generations}, PPO_epochs={args.num_ppo_epochs}, clip={args.clip_eps}, kl_beta={args.beta_kl}')

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

    dataset = GRPODataset(args.data_path, voc)
    Logger(f'Dataset: {len(dataset)} samples from {args.data_path}')

    scaler = torch.amp.GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    train_grpo(args, model, ref_model, dataset, optimizer, scaler, voc,
               voc_size, hidden_size, num_layers)

    if dist.is_initialized():
        dist.destroy_process_group()
