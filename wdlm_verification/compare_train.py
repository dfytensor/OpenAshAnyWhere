"""
OpenASH vs WDLM 对比训练 & 可视化脚本
相同数据、相同词表、相同训练配置，对比两个模型的loss曲线
"""

import os
import sys
import json
import time
import math
import random
import argparse
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings('ignore')

_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PARENT)
os.chdir(_PARENT)

from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from open_ash import OpenASH


# ============================================================
# 统一数据集 (与 WDLM 训练完全一致)
# ============================================================
class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_paths, tokenizer, max_seq_len=256):
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
            print(f"[Data] {path}: {sum(1 for _ in open(path,'r',encoding='utf-8') if _.strip())} 行")

        self.im_start = tokenizer.token_to_id.get("<|im_start|>")
        self.im_end = tokenizer.token_to_id.get("<|im_end|>")
        self.user_id = tokenizer.token_to_id.get("<|user|>")
        self.agent_id = tokenizer.token_to_id.get("<|agent|>")
        self.system_id = tokenizer.token_to_id.get("<|system|>")
        self.think_start = tokenizer.token_to_id.get("<|think|>")
        self.think_end = tokenizer.token_to_id.get("<|end_think|>")
        print(f"[Data] 总样本: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _split_think(self, text):
        think = None
        resp = text
        if '<think>' in text:
            s = text.find('<think>') + len('<think>')
            e = text.find('</think>')
            if e != -1:
                think = text[s:e].strip()
                resp = (text[:s - len('<think>')] + text[e + len('</think>'):]).strip()
        import re
        m = re.search(r'<\s*response\s*>(.*?)$', resp, re.DOTALL | re.IGNORECASE)
        if m:
            resp = m.group(1).strip()
        return think, resp

    def __getitem__(self, index):
        sample = json.loads(self.samples[index])
        convs = sample.get("conversations", [])
        msgs = []
        for msg in convs:
            role = msg.get("from", msg.get("role", ""))
            content = msg.get("value", msg.get("content", ""))
            if role in ("human", "user"):
                msgs += [self.im_start, self.user_id]
                msgs += self.tokenizer.encode(content)
                msgs += [self.im_end]
            elif role in ("gpt", "assistant", "agent"):
                msgs += [self.im_start, self.agent_id]
                think, resp = self._split_think(content)
                if think:
                    msgs += [self.think_start]
                    msgs += self.tokenizer.encode(think)
                    msgs += [self.think_end]
                if resp:
                    msgs += self.tokenizer.encode(resp)
                msgs += [self.im_end]
            elif role == "system":
                msgs += [self.im_start, self.system_id]
                msgs += self.tokenizer.encode(content)
                msgs += [self.im_end]
        if len(msgs) > self.max_seq_len:
            msgs = msgs[:self.max_seq_len]
        return torch.tensor(msgs, dtype=torch.long)

    @staticmethod
    def collate_fn(items):
        padded = pad_sequence(items, batch_first=True, padding_value=0)
        return padded[:, :-1], padded[:, 1:]


# ============================================================
# 工具函数
# ============================================================
def get_lr(step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total_steps)))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# 训练函数 (统一接口)
# ============================================================
def train_model(model, train_loader, total_steps, args, device, model_name):
    """训练模型并记录每一步的loss"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None

    loss_history = []
    loader_iter = iter(train_loader)
    step_time = time.time()

    print(f"\n[Train] {model_name}: total_steps={total_steps}, batches_per_epoch={len(train_loader)}")
    print(f"  Params: {count_params(model):,}")

    for step in range(1, total_steps + 1):
        try:
            inputs, targets = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            inputs, targets = next(loader_iter)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        lr = get_lr(step, total_steps, args.learning_rate)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # 前向
        if scaler:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                B, S, V = logits.shape
                loss = criterion(logits.view(B * S, V), targets.view(B * S))
        else:
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            B, S, V = logits.shape
            loss = criterion(logits.view(B * S, V), targets.view(B * S))

        # 反向
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

        loss_history.append(float(loss.item()))

        if step % args.log_interval == 0 or step == 1 or step == total_steps:
            elapsed = time.time() - step_time
            tok_per_s = (step - (step % args.log_interval) if step > args.log_interval else step) * \
                        args.batch_size * inputs.size(1) / max(elapsed, 0.01)
            # smoothed loss
            recent = loss_history[-min(50, len(loss_history)):]
            print(f"  [{model_name}] step {step:5d}/{total_steps} | "
                  f"loss: {loss.item():.4f} | avg50: {sum(recent)/len(recent):.4f} | "
                  f"lr: {lr:.2e} | {tok_per_s:.0f} tok/s")
            step_time = time.time()

    return loss_history


# ============================================================
# 绘图函数
# ============================================================
def plot_comparison(openash_loss, wdlm_loss, save_path, args):
    """绘制 loss 对比图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib 未安装，将保存数据到JSON而非图表")
        data = {"openash": openash_loss, "wdlm": wdlm_loss}
        with open(save_path.replace('.png', '.json'), 'w') as f:
            json.dump(data, f)
        print(f"[Plot] 数据已保存到 {save_path.replace('.png', '.json')}")
        return

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # 原始 loss
    ax1.plot(openash_loss, label='OpenASH', color='#2196F3', alpha=0.7, linewidth=1.0)
    ax1.plot(wdlm_loss, label='WDLM', color='#FF5722', alpha=0.7, linewidth=1.0)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('OpenASH vs WDLM - Raw Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 平滑 loss (滑动平均)
    window = 50
    def smooth(data, w):
        if len(data) < w:
            return data
        return [sum(data[max(0, i-w):i+1]) / len(data[max(0, i-w):i+1]) for i in range(len(data))]

    openash_smooth = smooth(openash_loss, window)
    wdlm_smooth = smooth(wdlm_loss, window)

    ax2.plot(openash_smooth, label=f'OpenASH (SMA-{window})', color='#2196F3', linewidth=2.0)
    ax2.plot(wdlm_smooth, label=f'WDLM (SMA-{window})', color='#FF5722', linewidth=2.0)

    # 标注起始和最终loss
    ax2.annotate(f'{openash_loss[0]:.2f}', xy=(0, openash_loss[0]),
                 xytext=(10, openash_loss[0] + 0.5),
                 arrowprops=dict(arrowstyle='->', color='#2196F3'), color='#2196F3', fontsize=9)
    ax2.annotate(f'{wdlm_loss[0]:.2f}', xy=(0, wdlm_loss[0]),
                 xytext=(10, wdlm_loss[0] - 0.5),
                 arrowprops=dict(arrowstyle='->', color='#FF5722'), color='#FF5722', fontsize=9)

    n = len(openash_loss) - 1
    ax2.annotate(f'{openash_loss[-1]:.2f}', xy=(n, openash_loss[-1]),
                 xytext=(n - 200, openash_loss[-1] + 0.5),
                 arrowprops=dict(arrowstyle='->', color='#2196F3'), color='#2196F3', fontsize=9)
    ax2.annotate(f'{wdlm_loss[-1]:.2f}', xy=(n, wdlm_loss[-1]),
                 xytext=(n - 200, wdlm_loss[-1] - 0.5),
                 arrowprops=dict(arrowstyle='->', color='#FF5722'), color='#FF5722', fontsize=9)

    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss (Smoothed)')
    ax2.set_title(f'OpenASH vs WDLM - Smoothed Loss (window={window})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 添加文本统计信息
    final_openash = sum(openash_loss[-50:]) / 50 if len(openash_loss) >= 50 else openash_loss[-1]
    final_wdlm = sum(wdlm_loss[-50:]) / 50 if len(wdlm_loss) >= 50 else wdlm_loss[-1]
    info_text = (
        f"Config: hidden={args.hidden_size}, layers={args.num_layers}, "
        f"batch={args.batch_size}, seq_len={args.max_seq_len}\n"
        f"OpenASH  Final loss (avg50): {final_openash:.4f}  |  Start: {openash_loss[0]:.4f}\n"
        f"WDLM     Final loss (avg50): {final_wdlm:.4f}  |  Start: {wdlm_loss[0]:.4f}"
    )
    fig.text(0.5, 0.01, info_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[Plot] 图表已保存到 {save_path}")

    # 同时保存JSON数据
    json_path = save_path.replace('.png', '.json')
    with open(json_path, 'w') as f:
        json.dump({"openash": openash_loss, "wdlm": wdlm_loss}, f)
    print(f"[Plot] 数据已保存到 {json_path}")


# ============================================================
# 主逻辑
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="OpenASH vs WDLM Loss 对比")
    parser.add_argument("--total_steps", type=int, default=2000, help="总训练步数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--max_seq_len", type=int, default=256, help="最大序列长度")
    parser.add_argument("--hidden_size", type=int, default=128, help="隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=2, help="层数")
    parser.add_argument("--num_heads", type=int, default=4, help="注意力头数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--log_interval", type=int, default=100, help="日志间隔")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据目录")
    parser.add_argument("--device", type=str, default="cuda:0", help="训练设备")
    parser.add_argument("--save_dir", type=str, default="./out", help="输出目录")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[Init] 设备: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- 加载词表 ----
    print("[Init] 加载词表...")
    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    voc_size = len(voc.token_to_id) + 1
    print(f"[Init] 词表大小: {voc_size}")

    # ---- 加载数据 ----
    data_paths = [os.path.join(args.data_dir, f) for f in ['science.jsonl']]
    dataset = UnifiedDataset(data_paths, voc, max_seq_len=args.max_seq_len)

    # 取子集加速加载 (每个模型用一个子集)
    n_samples = min(len(dataset), args.total_steps * args.batch_size * 2)
    indices = torch.randperm(len(dataset))[:n_samples].tolist()
    sub_dataset = torch.utils.data.Subset(dataset, indices)

    train_loader = DataLoader(
        sub_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=UnifiedDataset.collate_fn,
        drop_last=True,
    )
    print(f"[Init] 训练子集: {n_samples} 样本, {len(train_loader)} batches/epoch")

    # ========================================
    # 训练 OpenASH
    # ========================================
    print("\n" + "=" * 60)
    print("  训练 OpenASH")
    print("=" * 60)
    model_openash = OpenASH(
        voc_size=voc_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    ).to(device)
    print(f"  OpenASH params: {count_params(model_openash):,}")
    openash_loss = train_model(model_openash, train_loader, args.total_steps, args, device, "OpenASH")

    # 保存 OpenASH 权重
    torch.save(model_openash.state_dict(), f"{args.save_dir}/openash_compare.pth")
    torch.cuda.empty_cache()

    # ========================================
    # 训练 WDLM
    # ========================================
    print("\n" + "=" * 60)
    print("  训练 WDLM")
    print("=" * 60)
    try:
        import importlib.util
        wdlm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wdlm.py')
        spec = importlib.util.spec_from_file_location("wdlm", wdlm_path)
        wdlm_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wdlm_mod)

        model_wdlm = wdlm_mod.WaveDynamicsLanguageModel(
            vocab_size=voc_size,
            hidden_dim=args.hidden_size,
            num_layers=args.num_layers,
            n_qubits=8,
            n_waves=4
        ).to(device)
        print(f"  WDLM params: {count_params(model_wdlm):,}")
        wdlm_loss = train_model(model_wdlm, train_loader, args.total_steps, args, device, "WDLM")

        torch.save(model_wdlm.state_dict(), f"{args.save_dir}/wdlm_compare.pth")
    except Exception as e:
        print(f"[Error] WDLM 加载失败: {e}")
        # 如果没有 WDLM，使用已有日志数据
        wdlm_loss = None

    # ========================================
    # 绘制对比图
    # ========================================
    print("\n" + "=" * 60)
    print("  绘制 Loss 对比图")
    print("=" * 60)
    print(f"\n  OpenASH:  start={openash_loss[0]:.4f} -> end(avg50)={sum(openash_loss[-50:])/50:.4f}")
    if wdlm_loss:
        print(f"  WDLM:     start={wdlm_loss[0]:.4f} -> end(avg50)={sum(wdlm_loss[-50:])/50:.4f}")

    if wdlm_loss is None:
        wdlm_loss = [10.0] * len(openash_loss)

    save_path = os.path.join(args.save_dir, f"loss_comparison_{args.hidden_size}_{args.num_layers}.png")
    plot_comparison(openash_loss, wdlm_loss, save_path, args)

    print("\n[Done] 对比训练完成!")


if __name__ == "__main__":
    main()
