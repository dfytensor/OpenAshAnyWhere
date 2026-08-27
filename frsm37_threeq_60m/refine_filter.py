"""
精炼过滤 — 正则 + 小型 frsm 分类器 双重判别
==========================================
博客映射: Q1(验证)需要"理解语义"的判别器。正则只抓显式信号(169厘米),
会漏掉"无数字但事实性"的表述; frsm 分类器(val_acc 81%)补上语义判别。

策略 (降低误杀):
  1. 正则先分类 (高召回)
  2. 对正则判为 C 的样本, 用 frsm 分类器复核:
     - 若 frsm 高置信判为 A/B (prob_AB >= 阈值) -> 救援为 B (假阴性回收)
  3. 对正则判为 A 的高置信样本, 若 frsm 判为 C -> 降级 (假阳性纠正)
  4. 再按 C<=20% 配额过滤

产出 pretrain_refined.jsonl, 与纯正则版对比。
"""
import os
import sys
import json
import time
import argparse

if sys.platform == 'win32':
    os.environ.setdefault('PYTHONUTF8', '1')

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'F:\OpenASH2605')

from open_ash_voc import OpenASHVoc
from frsm_classifier import FRSMClassifier, ID2CAT
from three_q_filter import classify_three_q
from filter_dataset import _extract_text

HERE = os.path.dirname(os.path.abspath(__file__))
VS = 23006

RESCUE_THRESH = 0.55    # frsm 判 A/B 的置信度阈值, 超过则把 C 救援为 B
DEMOTE_THRESH = 0.75    # frsm 判 C 的高置信阈值, 把 A 降级


@torch.no_grad()
def frsm_predict(model, tok, texts, dev, seq_len=256, bs=64):
    """批量返回每条文本的 (pred_cat, prob_A, prob_B, prob_C)."""
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        xs = []
        for t in chunk:
            ids = tok.encode(t)[:seq_len]
            if len(ids) < 4:
                ids = ids + [0] * (4 - len(ids))
            xs.append(torch.tensor(ids, dtype=torch.long))
        x = pad_sequence(xs, batch_first=True, padding_value=0).to(dev).clamp(0, VS - 1)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
        prob = F.softmax(logits.float(), dim=-1).cpu()
        for p in prob:
            out.append((ID2CAT[p.argmax().item()], p[0].item(), p[1].item(), p[2].item()))
    return out


def refine(in_path, out_path, model, tok, dev, c_quota=0.20, max_lines=0, label='refine'):
    t0 = time.time()
    print(f'[{label}] 读取 {in_path} ...', flush=True)
    with open(in_path, encoding='utf-8') as f:
        lines = f.readlines()
    if max_lines:
        lines = lines[:max_lines]
    total = len(lines)

    # 1. 正则分类
    regex_results = []   # (cat, obj, text)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        text = _extract_text(obj)
        if len(text) < 20:
            continue
        cat = classify_three_q(text)['category']
        regex_results.append([cat, obj, text])

    from collections import Counter
    rc = Counter(r[0] for r in regex_results)
    print(f'[{label}] 正则分布: A={rc["A"]} B={rc["B"]} C={rc["C"]}', flush=True)

    # 2. frsm 复核 C 类 (批量)
    c_idx = [i for i, r in enumerate(regex_results) if r[0] == 'C']
    if c_idx:
        c_texts = [regex_results[i][2][:2000] for i in c_idx]
        print(f'[{label}] frsm 复核 {len(c_idx)} 条 C 类...', flush=True)
        preds = frsm_predict(model, tok, c_texts, dev)
        rescued = 0
        for j, i in enumerate(c_idx):
            pcat, pa, pb, pc = preds[j]
            if pa + pb >= RESCUE_THRESH:        # frsm 认为是 A/B -> 救援
                regex_results[i][0] = 'B'
                rescued += 1
        print(f'[{label}] 救援 {rescued} 条 C->B (阈值 prob_AB>={RESCUE_THRESH})', flush=True)

    # 3. frsm 复核 A 类高置信 -> 降级假阳性
    a_idx = [i for i, r in enumerate(regex_results) if r[0] == 'A']
    if a_idx:
        a_texts = [regex_results[i][2][:2000] for i in a_idx]
        preds = frsm_predict(model, tok, a_texts, dev)
        demoted = 0
        for j, i in enumerate(a_idx):
            pcat, pa, pb, pc = preds[j]
            if pc >= DEMOTE_THRESH:
                regex_results[i][0] = 'C'
                demoted += 1
        print(f'[{label}] 降级 {demoted} 条 A->C (阈值 prob_C>={DEMOTE_THRESH})', flush=True)

    fc = Counter(r[0] for r in regex_results)
    print(f'[{label}] 精炼后分布: A={fc["A"]} B={fc["B"]} C={fc["C"]}', flush=True)

    # 4. C 类配额过滤
    a = [r for r in regex_results if r[0] == 'A']
    b = [r for r in regex_results if r[0] == 'B']
    c = [r for r in regex_results if r[0] == 'C']
    for r in a + b + c:
        r[1]['tqf_cat'] = r[0]
    keep_core = len(a) + len(b)
    c_budget = int(keep_core * c_quota / (1 - c_quota)) if c_quota < 1 else len(c)
    c_budget = max(0, min(len(c), c_budget))
    c_kept = c[:c_budget]
    kept = [r[1] for r in a + b + c_kept]

    with open(out_path, 'w', encoding='utf-8') as f:
        for obj in kept:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    stats = {'total': total, 'A': len(a), 'B': len(b), 'C_total': len(c),
             'C_kept': len(c_kept), 'kept': len(kept),
             'c_ratio': round(len(c_kept) / max(len(kept), 1), 4),
             'time_s': round(time.time() - t0, 1)}
    print(f'[{label}] 写出 {len(kept)} 条, C占比={stats["c_ratio"]:.1%} -> {out_path}', flush=True)
    print(f'[{label}] 耗时 {stats["time_s"]}s', flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_lines', type=int, default=0)
    ap.add_argument('--c_quota', type=float, default=0.20)
    args = ap.parse_args()
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tok = OpenASHVoc(agent_voc_path=os.path.join(r'F:\OpenASH2605', 'open_ash_voc_agent.json'))

    ckpt = os.path.join(HERE, 'checkpoints', 'frsm_classifier.pth')
    cinfo = torch.load(ckpt, map_location=dev, weights_only=False)
    model = FRSMClassifier(VS, **cinfo['config']).to(dev)
    model.load_state_dict(cinfo['model'])
    print(f'加载 frsm 分类器 (val_acc={cinfo["acc"]:.4f})', flush=True)

    out_dir = os.path.join(HERE, 'filtered_data')
    stats = refine(
        r'F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl',
        os.path.join(out_dir, 'pretrain_refined.jsonl'),
        model, tok, dev, args.c_quota, args.max_lines, 'pretrain-refined')
    with open(os.path.join(out_dir, 'refine_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == '__main__':
    main()
