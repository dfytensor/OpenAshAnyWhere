"""
WDLM 波动力学语言模型训练脚本
基于 train_full_sft.py 训练架构，使用 WDLM 模型替代 OpenASH
数据: F:\OpenASH2605\data\science.jsonl + if.jsonl
词表: OpenASHVoc (与 OpenASH 一致)
"""

import os
import sys
import time
import math
import json
import warnings
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings('ignore')

# 添加父目录到路径以导入 OpenASHVoc
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _parent_dir)

from open_ash_voc import OpenASHVoc
from config import agent_voc_path

os.chdir(_parent_dir)


# ============================================================
# 工具函数
# ============================================================
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# 数据集
# ============================================================
class WDLMTrainDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_paths, tokenizer, max_seq_len=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        for path in jsonl_paths:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.samples.append(line)
                print(f"[Dataset] 加载 {path}: {sum(1 for _ in open(path, 'r', encoding='utf-8'))} 行")

        self.im_start = tokenizer.token_to_id.get("<|im_start|>")
        self.im_end = tokenizer.token_to_id.get("<|im_end|>")
        self.user_id = tokenizer.token_to_id.get("<|user|>")
        self.agent_id = tokenizer.token_to_id.get("<|agent|>")
        self.system_id = tokenizer.token_to_id.get("<|system|>")
        self.think_start = tokenizer.token_to_id.get("<|think|>")
        self.think_end = tokenizer.token_to_id.get("<|end_think|>")
        print(f"[Dataset] 总样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _extract_think_and_response(self, text):
        """提取思考内容和回复内容"""
        think_content = None
        response_content = text

        # <think>...</think>
        if '<think>' in text:
            start = text.find('<think>') + len('<think>')
            end = text.find('</think>')
            if end != -1:
                think_content = text[start:end].strip()
                response_content = (text[:start - len('<think>')] + text[end + len('</think>'):]).strip()

        # <\s*response\s*>
        import re
        rm = re.search(r'<\s*response\s*>(.*?)$', response_content, re.DOTALL | re.IGNORECASE)
        if rm:
            response_content = rm.group(1).strip()

        return think_content, response_content

    def __getitem__(self, index):
        sample = json.loads(self.samples[index])
        conversations = sample.get("conversations", [])
        messages = []

        for msg in conversations:
            role = msg.get("from", msg.get("role", ""))
            content = msg.get("value", msg.get("content", ""))

            if role in ("human", "user"):
                messages += [self.im_start, self.user_id]
                messages += self.tokenizer.encode(content)
                messages += [self.im_end]
            elif role in ("gpt", "assistant", "agent"):
                messages += [self.im_start, self.agent_id]
                think, resp = self._extract_think_and_response(content)
                if think:
                    messages += [self.think_start]
                    messages += self.tokenizer.encode(think)
                    messages += [self.think_end]
                if resp:
                    messages += self.tokenizer.encode(resp)
                messages += [self.im_end]
            elif role == "system":
                messages += [self.im_start, self.system_id]
                messages += self.tokenizer.encode(content)
                messages += [self.im_end]

        if len(messages) > self.max_seq_len:
            messages = messages[:self.max_seq_len]

        return torch.tensor(messages, dtype=torch.long)

    @staticmethod
    def collate_fn(items):
        padded = pad_sequence(items, batch_first=True, padding_value=0)
        return padded[:, :-1], padded[:, 1:]


# ============================================================
# WDLM 模型导入
# ============================================================
def create_wdlm_model(vocab_size, hidden_dim=256, num_layers=4, n_heads=8):
    """创建 WDLM 模型"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wdlm_path = os.path.join(script_dir, 'wdlm.py')
    if os.path.exists(wdlm_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("wdlm", wdlm_path)
        wdlm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wdlm)
        # WaveDynamicsLanguageModel 有 generate 方法，基础但完整
        model = wdlm.WaveDynamicsLanguageModel(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            n_qubits=8,
            n_waves=4
        )
        return model
    return None


# ============================================================
# 训练循环
# ============================================================
def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[Init] 设备: {device}")

    setup_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载词表
    print("[Init] 加载词表...")
    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    voc_size = len(voc.token_to_id) + 1  # +1 for padding 0
    print(f"[Init] 词表大小: {voc_size}")

    # 模型配置
    hidden_dim = args.hidden_size if args.hidden_size else 256
    num_layers = args.num_layers if args.num_layers else 4
    n_heads = args.num_heads if args.num_heads else 8

    print(f"[Init] WDLM 配置: voc_size={voc_size}, hidden={hidden_dim}, layers={num_layers}, heads={n_heads}")

    # 创建模型
    print("[Init] 创建 WDLM 模型...")
    model = create_wdlm_model(voc_size, hidden_dim, num_layers, n_heads)
    if model is None:
        raise RuntimeError("无法导入 WDLM 模型，请确保 wdlm.py 存在")

    n_params = count_params(model)
    print(f"[Init] 模型参数: {n_params:,}")

    model.to(device)

    # 编译加速 (可选)
    if args.use_compile and device.type == 'cuda':
        print("[Init] torch.compile 加速...")
        model = torch.compile(model)

    # 数据集
    print("[Init] 加载数据集...")
    data_paths = [os.path.join(args.data_dir, p) for p in args.data_files.split(',')]
    dataset = WDLMTrainDataset(data_paths, voc, max_seq_len=args.max_seq_len)

    # 划分训练/验证集
    n_train = int(len(dataset) * 0.95)
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    print(f"[Init] 训练集: {n_train}, 验证集: {n_val}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=dataset.collate_fn,
        drop_last=True,
    )

    # 优化器和损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None

    total_steps = args.epochs * len(train_loader)
    global_step = 0

    print(f"[Train] 开始训练, {args.epochs} epochs, {len(train_loader)} steps/epoch")
    print(f"[Train] total_steps={total_steps}, batch_size={args.batch_size}, max_seq_len={args.max_seq_len}")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for step, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            global_step += 1
            lr = get_lr(global_step, total_steps, args.learning_rate)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # 前向传播
            if scaler:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, psi = model(inputs)
                    B, S, V = logits.shape
                    loss = criterion(logits.view(B * S, V), targets.view(B * S))
            else:
                logits, psi = model(inputs)
                B, S, V = logits.shape
                loss = criterion(logits.view(B * S, V), targets.view(B * S))

            # 反向传播
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            epoch_loss += loss.item()

            # 日志
            if step % args.log_interval == 0 or step == len(train_loader) - 1:
                avg_loss = epoch_loss / (step + 1)
                elapsed = time.time() - epoch_start
                tokens_per_sec = (step + 1) * args.batch_size * inputs.size(1) / max(elapsed, 0.001)
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | Step {step}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) | "
                    f"LR: {lr:.2e} | {tokens_per_sec:.0f} tok/s"
                )

            # 保存
            if step % args.save_interval == 0 and step > 0:
                save_path = f"{args.save_dir}/wdlm_e{epoch + 1}_s{step}_{hidden_dim}_{num_layers}.pth"
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'voc_size': voc_size,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                }, save_path)
                print(f"[Save] {save_path}")

        # Epoch 结束时保存
        epoch_avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - epoch_start
        print(f"[Epoch {epoch + 1}] avg_loss={epoch_avg_loss:.4f}, time={elapsed:.0f}s")

        save_path = f"{args.save_dir}/wdlm_epoch{epoch + 1}_{hidden_dim}_{num_layers}.pth"
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': global_step,
            'voc_size': voc_size,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
        }, save_path)
        print(f"[Save] {save_path}")

    print("[Done] 训练完成!")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WDLM 训练")
    parser.add_argument("--save_dir", type=str, default="./out", help="模型保存目录")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据目录")
    parser.add_argument("--data_files", type=str, default="science.jsonl,if.jsonl", help="数据文件(逗号分隔)")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--max_seq_len", type=int, default=512, help="最大序列长度")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--log_interval", type=int, default=50, help="日志间隔")
    parser.add_argument("--save_interval", type=int, default=500, help="保存间隔")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--device", type=str, default="cuda:0", help="训练设备")
    parser.add_argument("--hidden_size", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=4, help="WDLM 层数")
    parser.add_argument("--num_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--use_compile", type=int, default=0, choices=[0, 1], help="torch.compile")

    args = parser.parse_args()
    train(args)
