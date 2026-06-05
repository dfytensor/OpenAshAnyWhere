"""
PhaseGate vs ReLU FFN - Loss 曲线对比
相同: OpenASH 架构, 词表, 数据, 超参
唯一差异: FeedForward (ReLU gate) vs PhaseGatingFFN (sin+cos gate)
"""

import os, sys, json, time, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PARENT)
os.chdir(_PARENT)

from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from open_ash import OpenASH, DecoderLayer, MaxStateSuper


# ============================================================
class PhaseGatingFFN(nn.Module):
    """WDLM-inspired: sin/cos 替代 ReLU gate"""
    def __init__(self, hidden_size):
        super().__init__()
        h = hidden_size
        self.value_proj = nn.Linear(h, h, bias=False)
        self.gate_proj = nn.Linear(h, h, bias=False)
        self.out_proj = nn.Linear(h, h, bias=False)

    def forward(self, x):
        v = self.value_proj(x)
        g = self.gate_proj(x)
        return self.out_proj(v * (torch.sin(g) + torch.cos(g)) * 0.5)


def build_phasegate_openash(voc_size, hidden_size, num_heads, num_layers):
    """创建使用 PhaseGateFFN 的 OpenASH"""
    model = OpenASH(voc_size, hidden_size, num_heads, num_layers)
    for layer in model.decoder_layers:
        layer.ffn = PhaseGatingFFN(hidden_size)
    return model


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
                    self.samples.extend(l.strip() for l in f if l.strip())
        self.im_start = tokenizer.token_to_id.get("<|im_start|>")
        self.im_end = tokenizer.token_to_id.get("<|im_end|>")
        self.user_id = tokenizer.token_to_id.get("<|user|>")
        self.agent_id = tokenizer.token_to_id.get("<|agent|>")
        self.think_start = tokenizer.token_to_id.get("<|think|>")
        self.think_end = tokenizer.token_to_id.get("<|end_think|>")
        print(f"[Data] {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def _split_think(self, text):
        think = None; resp = text
        if '<think>' in text:
            s = text.find('<think>') + 7; e = text.find('</think>')
            if e != -1: think = text[s:e].strip(); resp = text[:s-7] + text[e+8:]; resp = resp.strip()
        import re
        m = re.search(r'<\s*response\s*>(.*?)$', resp, re.DOTALL | re.IGNORECASE)
        if m: resp = m.group(1).strip()
        return think, resp

    def __getitem__(self, index):
        sample = json.loads(self.samples[index])
        convs = sample.get("conversations", [])
        msgs = []
        for msg in convs:
            role = msg.get("from", msg.get("role", ""))
            content = msg.get("value", msg.get("content", ""))
            if role in ("human", "user"):
                msgs += [self.im_start, self.user_id] + self.tokenizer.encode(content) + [self.im_end]
            elif role in ("gpt", "assistant", "agent"):
                think, resp = self._split_think(content)
                msgs += [self.im_start, self.agent_id]
                if think: msgs += [self.think_start] + self.tokenizer.encode(think) + [self.think_end]
                if resp: msgs += self.tokenizer.encode(resp)
                msgs += [self.im_end]
        if len(msgs) > self.max_seq_len: msgs = msgs[:self.max_seq_len]
        return torch.tensor(msgs, dtype=torch.long)

    @staticmethod
    def collate_fn(items):
        padded = pad_sequence(items, batch_first=True, padding_value=0)
        return padded[:, :-1], padded[:, 1:]


# ============================================================
def get_lr(step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total_steps)))


def train_steps(model, loader, total_steps, device):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    loss_hist = []
    loader_iter = iter(loader)

    for step in range(1, total_steps + 1):
        try:
            inputs, targets = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            inputs, targets = next(loader_iter)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        lr = get_lr(step, total_steps, 5e-5)
        for pg in opt.param_groups: pg['lr'] = lr

        if scaler:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out, _ = model(inputs)
                loss = criterion(out.view(-1, out.size(-1)), targets.view(-1))
        else:
            out, _ = model(inputs)
            loss = criterion(out.view(-1, out.size(-1)), targets.view(-1))

        opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        loss_hist.append(float(loss.detach()))

        if step % 200 == 0 or step == 1 or step == total_steps:
            avg = sum(loss_hist[-50:]) / min(50, len(loss_hist))
            print(f"  step {step:5d}/{total_steps}  loss={loss.item():.4f}  avg50={avg:.4f}")

    return loss_hist


def plot_comparison(relu_loss, phase_loss, save_path):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except:
        print("no matplotlib")
        return

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.plot(relu_loss, label='OpenASH (ReLU gate)', color='#2196F3', alpha=0.6, linewidth=1.0)
    ax1.plot(phase_loss, label='OpenASH (sin+cos gate)', color='#FF5722', alpha=0.6, linewidth=1.0)
    ax1.set_xlabel('Step'); ax1.set_ylabel('Loss')
    ax1.set_title('OpenASH ReLU vs PhaseGate - Raw Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    w = 50
    def smooth(d, w):
        if len(d) < w: return d
        return [sum(d[max(0,i-w+1):i+1]) / len(d[max(0,i-w+1):i+1]) for i in range(len(d))]

    ax2.plot(smooth(relu_loss, w), label=f'ReLU gate (SMA-{w})', color='#2196F3', linewidth=2.0)
    ax2.plot(smooth(phase_loss, w), label=f'sin+cos gate (SMA-{w})', color='#FF5722', linewidth=2.0)
    ax2.set_xlabel('Step'); ax2.set_ylabel('Loss (Smoothed)')
    ax2.set_title('OpenASH ReLU vs PhaseGate - Smoothed Loss')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fr = sum(relu_loss[-50:])/50; fp = sum(phase_loss[-50:])/50
    fig.text(0.5, 0.01,
             f"ReLU final avg50: {fr:.4f}  |  PhaseGate final avg50: {fp:.4f}  |  delta: {fp-fr:+.4f}",
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[Plot] saved to {save_path}")

    # 保存数据
    with open(save_path.replace('.png', '.json'), 'w') as f:
        json.dump({'relu': relu_loss, 'phasegate': phase_loss}, f)


# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # 词表
    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    vs = len(voc.token_to_id) + 1
    print(f"voc_size: {vs}")

    # 数据
    ds = UnifiedDataset([os.path.join(args.data_dir, 'science.jsonl')], voc, args.seq)
    n = min(len(ds), args.steps * args.bs * 3)
    idx = torch.randperm(len(ds))[:n].tolist()
    sub = torch.utils.data.Subset(ds, idx)
    loader = DataLoader(sub, batch_size=args.bs, shuffle=True, num_workers=0,
                        collate_fn=UnifiedDataset.collate_fn, drop_last=True)
    print(f"training samples: {n}, batches: {len(loader)}")

    # ==== Train ReLU ====
    print("\n=== Training OpenASH (ReLU gate) ===")
    torch.manual_seed(42)
    m_relu = OpenASH(vs, args.hidden, args.heads, args.layers).to(device)
    p_relu = sum(p.numel() for p in m_relu.parameters() if p.requires_grad)
    print(f"params: {p_relu:,}")
    loss_relu = train_steps(m_relu, loader, args.steps, device)
    torch.cuda.empty_cache()

    # ==== Train PhaseGate ====
    print("\n=== Training OpenASH (sin+cos gate) ===")
    torch.manual_seed(42)
    m_phase = build_phasegate_openash(vs, args.hidden, args.heads, args.layers).to(device)
    p_phase = sum(p.numel() for p in m_phase.parameters() if p.requires_grad)
    print(f"params: {p_phase:,}")
    loss_phase = train_steps(m_phase, loader, args.steps, device)
    torch.cuda.empty_cache()

    # ==== Report ====
    lr_end = sum(loss_relu[-50:]) / 50
    lp_end = sum(loss_phase[-50:]) / 50
    print(f"\n=== Results ===")
    print(f"ReLU gate:     {loss_relu[0]:.2f} -> {lr_end:.4f}  ({p_relu:,} params)")
    print(f"PhaseGate:     {loss_phase[0]:.2f} -> {lp_end:.4f}  ({p_phase:,} params)")
    print(f"Final delta:   {lp_end-lr_end:+.4f}")
    print(f"Params delta:  {p_phase-p_relu:+d}")

    # ==== Plot ====
    save_path = f"./out/phasegate_vs_relu_{args.hidden}_{args.layers}.png"
    os.makedirs("./out", exist_ok=True)
    plot_comparison(loss_relu, loss_phase, save_path)
