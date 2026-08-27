"""
小型 FRSMASH v3.7 文本分类器
============================
用微型 frsmash3.7 骨干做 3 分类 (A强可验证 / B弱可验证 / C不可验证),
以正则分类器(three_q_filter)的标签为伪标签训练, 再用它复核正则的"边界样本".

博客映射: Q1(验证)需要一个能理解语义(而非仅模式匹配)的判别器——
正则抓"169厘米"这类显式信号, frsm 分类器抓"无数字但事实性表述"这类隐式信号。
两者互补: 正则召回率高, frsm 精度高。

架构: FRSMASH 骨干(SSM+SlowMemory+recall, H=96 L=2 ≈2.5M) -> masked mean pool -> 3类头.
"""
import os
import sys
import json
import time
import random
import argparse

if sys.platform == 'win32':
    os.environ.setdefault('PYTHONUTF8', '1')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'F:\OpenASH2605')

from open_ash_voc import OpenASHVoc
from model import SSMLayer, LinearSlowMemory, GlaRecall
from three_q_filter import classify_three_q

HERE = os.path.dirname(os.path.abspath(__file__))
VS = 23006
CAT2ID = {'A': 0, 'B': 1, 'C': 2}
ID2CAT = {0: 'A', 1: 'B', 2: 'C'}


class FRSMClassifier(nn.Module):
    """微型 FRSMASH v3.7 文本分类器: 骨干 -> masked mean pool -> 3 类."""

    def __init__(self, voc_size=VS, hidden=96, heads=4, layers=2, n_slots=4,
                 n_classes=3, max_pe=2048):
        super().__init__()
        self.D = hidden
        import math
        self.em = nn.Embedding(voc_size, hidden, padding_idx=0)
        _pe = torch.zeros(max_pe, hidden)
        _pos = torch.arange(max_pe).unsqueeze(1)
        _div = torch.exp(torch.arange(0, hidden, 2) * (-math.log(10000) / hidden))
        _pe[:, 0::2] = torch.sin(_pos * _div)
        _pe[:, 1::2] = torch.cos(_pos * _div)
        self.register_buffer('pe', _pe)

        self.layers = nn.ModuleList([SSMLayer(hidden, heads, n_slots) for _ in range(layers)])
        self.final_norm = nn.RMSNorm(hidden)
        self.slow_cell = LinearSlowMemory(hidden)
        self.mem_proj = nn.Linear(hidden, hidden, bias=False)
        self.mem_norm = nn.RMSNorm(hidden)
        self.recall = GlaRecall(hidden, heads=heads, d_h=64)
        self.recall_norm = nn.RMSNorm(hidden)
        self.fusion_norm = nn.RMSNorm(hidden)
        self.class_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_classes),
        )

    def forward(self, x, mask=None):
        B, T = x.shape
        dt = self.class_head[0].weight.dtype
        x_emb = self.em(x).to(dt) + self.pe[:T].to(dt)
        h_slow = torch.zeros(B, self.D, device=x.device, dtype=dt)
        h = x_emb
        for layer in self.layers:
            h, _ = layer(h)
        x_ash = self.final_norm(h)
        H_slow, _ = self.slow_cell(x_emb, h_slow)
        x_mem = self.mem_norm(self.mem_proj(H_slow))
        x_recall = self.recall_norm(self.recall(x_emb))
        fused = self.fusion_norm(x_ash + x_mem + x_emb) + x_recall
        # masked mean pool
        if mask is None:
            mask = (x != 0).float().unsqueeze(-1)
        else:
            mask = mask.float().unsqueeze(-1)
        pooled = (fused.float() * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.class_head(pooled.to(dt))


def build_train_data(tok, path, n_per_cat=12000, seq_len=256, seed=42):
    """从带 tqf 标签的过滤数据采样均衡训练集."""
    random.seed(seed)
    by_cat = {'A': [], 'B': [], 'C': []}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cat = obj.get('tqf_cat')
            if cat not in by_cat:
                continue
            text = obj.get('text', '')
            if len(text) < 16:
                continue
            by_cat[cat].append(text)
    for c in by_cat:
        random.shuffle(by_cat[c])
    samples = []
    for c in 'ABC':
        for text in by_cat[c][:n_per_cat]:
            ids = tok.encode(text)[:seq_len]
            if len(ids) >= 8:
                samples.append((ids, CAT2ID[c]))
    random.shuffle(samples)
    return samples


def train_classifier(args):
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tok = OpenASHVoc(agent_voc_path=os.path.join(r'F:\OpenASH2605', 'open_ash_voc_agent.json'))
    print('构建训练数据 (正则伪标签)...', flush=True)
    samples = build_train_data(tok, os.path.join(HERE, 'filtered_data', 'pretrain_filtered.jsonl'),
                               n_per_cat=args.n_per_cat, seq_len=args.seq_len)
    # 切分 train/val
    random.seed(0)
    random.shuffle(samples)
    n_val = min(2000, len(samples) // 10)
    val = samples[:n_val]
    train = samples[n_val:]
    print(f'训练 {len(train)} / 验证 {len(val)}', flush=True)

    model = FRSMClassifier(VS, hidden=args.hidden, heads=args.heads, layers=args.layers).to(dev)
    n = sum(p.numel() for p in model.parameters())
    print(f'FRSM 分类器: H={args.hidden} L={args.layers} heads={args.heads} = {n:,} 参数', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    def make_batch(items):
        xs = [torch.tensor(s, dtype=torch.long) for s, _ in items]
        ys = torch.tensor([y for _, y in items], dtype=torch.long)
        x = pad_sequence(xs, batch_first=True, padding_value=0)
        return x.to(dev), ys.to(dev)

    BS = 64
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train)
        total_loss = 0.0
        nb = 0
        t0 = time.time()
        for i in range(0, len(train), BS):
            batch = train[i:i + BS]
            x, y = make_batch(batch)
            x = x.clamp(0, VS - 1)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            nb += 1
            if nb % 100 == 0:
                print(f'  e{epoch+1} b{nb} loss={total_loss/nb:.4f} ({time.time()-t0:.0f}s)', flush=True)

        # 验证
        model.eval()
        correct = tot = 0
        cm = [[0, 0, 0] for _ in range(3)]
        with torch.no_grad():
            for i in range(0, len(val), BS):
                batch = val[i:i + BS]
                x, y = make_batch(batch)
                x = x.clamp(0, VS - 1)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x)
                pred = logits.argmax(-1)
                correct += (pred == y).sum().item()
                tot += y.size(0)
                for yi, pi in zip(y.tolist(), pred.tolist()):
                    cm[yi][pi] += 1
        acc = correct / max(tot, 1)
        print(f'[e{epoch+1}] val_acc={acc:.4f} loss={total_loss/max(nb,1):.4f}', flush=True)
        print(f'  混淆矩阵(行=真实 A/B/C, 列=预测): '
              f'A->{cm[0]} B->{cm[1]} C->{cm[2]}', flush=True)
        if acc > best_acc:
            best_acc = acc
            torch.save({'model': model.state_dict(), 'acc': acc,
                        'config': {'hidden': args.hidden, 'heads': args.heads, 'layers': args.layers}},
                       os.path.join(HERE, 'checkpoints', 'frsm_classifier.pth'))
            print(f'  [保存] best_acc={best_acc:.4f}', flush=True)
    print(f'\n完成. 最佳 val_acc={best_acc:.4f}', flush=True)
    return best_acc


@torch.no_grad()
def predict_batch(model, tok, texts, dev, seq_len=256, bs=64):
    """批量预测类别 + 置信度."""
    model.eval()
    results = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        xs = [torch.tensor(tok.encode(t)[:seq_len], dtype=torch.long) for t in chunk]
        x = pad_sequence(xs, batch_first=True, padding_value=0).to(dev).clamp(0, VS - 1)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
        prob = F.softmax(logits.float(), dim=-1)
        pred = prob.argmax(-1)
        conf = prob.max(-1).values
        for p, c in zip(pred.tolist(), conf.tolist()):
            results.append((ID2CAT[p], c))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden', type=int, default=96)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--n_per_cat', type=int, default=12000)
    ap.add_argument('--seq_len', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=5e-4)
    args = ap.parse_args()
    train_classifier(args)


if __name__ == '__main__':
    main()
