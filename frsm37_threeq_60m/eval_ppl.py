"""
困惑度评估 — 对比三问过滤 vs 未过滤
==================================
在保留的验证集上计算 perplexity. 重点看「事实类(A类)」文本上的 ppl,
验证博客论点: 过滤掉主观噪音后, 模型对可验证知识的掌握更精准 (ppl 更低).

用法:
  python eval_ppl.py --ckpt checkpoints/frsm37_60m_pretrain_final.pth --tag filtered
  python eval_ppl.py --ckpt <baseline_ckpt> --tag baseline
  python eval_ppl.py --compare   # 直接对比两个 checkpoint
"""
import os
import sys
import json
import argparse

if sys.platform == 'win32':
    os.environ.setdefault('PYTHONUTF8', '1')

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'F:\OpenASH2605')

from open_ash_voc import OpenASHVoc
from model import FRSMASHv37
from three_q_filter import classify_three_q

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'filtered_data')
VS = 23006
HIDDEN, LAYERS, HEADS = 448, 7, 8
EVAL_N = 1000           # 验证样本数
SEQ = 512


def load_model(ckpt_path, dev):
    model = FRSMASHv37(VS, HIDDEN, HEADS, LAYERS).to(dev)
    ckp = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model.load_state_dict(ckp['model'])
    model.eval()
    return model


def build_eval_set(tok, n_per_cat=250, seed=123, skip_train=0):
    """从 SFT 数据构建验证集 (预训练模型未见过, 无泄漏), 按 A/B/C 类别均衡.
    全量预训练模型都训练了完整 pretrain, 故评估必须用另一来源 (sft) 才无泄漏."""
    raw_path = r'F:\OpenASH2605\minimind_data\sft_t2t_mini.jsonl'
    with open(raw_path, encoding='utf-8') as f:
        lines = f.readlines()
    import random
    random.seed(seed)
    random.shuffle(lines)

    sets = {'all': [], 'A': [], 'B': [], 'C': []}
    budgets = {'A': n_per_cat, 'B': n_per_cat, 'C': n_per_cat}
    for line in lines:
        if all(v <= 0 for v in budgets.values()):
            break
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        # 拼接 user+assistant content 作为评估文本
        convs = obj.get('conversations', [])
        text = '\n'.join(m.get('content', '') for m in convs
                         if m.get('role') in ('user', 'assistant'))
        if len(text) < 24:
            continue
        cat = classify_three_q(text)['category']
        if budgets.get(cat, 0) <= 0:
            continue
        ids = tok.encode(text)[:SEQ + 1]
        if len(ids) < 16:
            continue
        t = torch.tensor(ids, dtype=torch.long)
        sets[cat].append(t)
        sets['all'].append(t)
        budgets[cat] -= 1
    return sets


@torch.no_grad()
def eval_ppl(model, samples, dev, bs=8):
    """计算平均 ppl (按 token 平均)."""
    total_loss = 0.0
    total_tok = 0
    for i in range(0, len(samples), bs):
        batch = samples[i:i + bs]
        x = pad_sequence(batch, batch_first=True, padding_value=0).to(dev)
        inp, tgt = x[:, :-1], x[:, 1:]
        inp = inp.clamp(0, VS - 1)
        tgt = tgt.clamp(0, VS - 1)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, VS), tgt.reshape(-1),
                               ignore_index=0, reduction='sum')
        ntok = (tgt != 0).sum().item()
        total_loss += loss.item()
        total_tok += ntok
    avg_nll = total_loss / max(total_tok, 1)
    return avg_nll, float(torch.exp(torch.tensor(avg_nll))), total_tok


def run_eval(ckpt_path, tag, dev):
    tok = OpenASHVoc(agent_voc_path=os.path.join(r'F:\OpenASH2605', 'open_ash_voc_agent.json'))
    model = load_model(ckpt_path, dev)
    sets = build_eval_set(tok)
    print(f'\n=== Eval [{tag}] ({os.path.basename(ckpt_path)}) ===', flush=True)
    results = {}
    for name in ['all', 'A', 'B', 'C']:
        if not sets[name]:
            continue
        nll, ppl, ntok = eval_ppl(model, sets[name], dev)
        results[name] = {'n': len(sets[name]), 'nll': round(nll, 4), 'ppl': round(ppl, 2)}
        print(f'  {name:>3}: n={len(sets[name]):>4}  nll={nll:.4f}  ppl={ppl:.2f}', flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, default=None)
    ap.add_argument('--tag', type=str, default='model')
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--filtered_ckpt', type=str,
                    default=os.path.join(HERE, 'checkpoints', 'frsm37_60m_pretrain_final.pth'))
    ap.add_argument('--baseline_ckpt', type=str,
                    default=os.path.join(HERE, 'checkpoints', 'frsm37_60m_baseline_pretrain_final.pth'))
    ap.add_argument('--refined_ckpt', type=str,
                    default=os.path.join(HERE, 'checkpoints', 'frsm37_60m_refined_pretrain_final.pth'))
    args = ap.parse_args()
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.compare:
        rf = run_eval(args.filtered_ckpt, '纯正则过滤', dev)
        rb = run_eval(args.baseline_ckpt, '未过滤基线', dev)
        has_refined = os.path.exists(args.refined_ckpt)
        rr = run_eval(args.refined_ckpt, '正则+frsm精炼', dev) if has_refined else None
        print('\n' + '=' * 68)
        hdr = f'{"指标":>8} | {"未过滤":>9} | {"纯正则":>9}'
        if rr:
            hdr += f' | {"+frsm精炼":>10}'
        print(hdr)
        print('-' * 68)
        for name in ['all', 'A', 'B', 'C']:
            if name not in rb or name not in rf:
                continue
            pb = rb[name]['ppl']
            pf = rf[name]['ppl']
            line = f'{name:>8} | {pb:>9.2f} | {pf:>9.2f}'
            if rr and name in rr:
                pr = rr[name]['ppl']
                line += f' | {pr:>10.2f}'
            line += '   (vs基线: '
            line += f'纯正则{(pf-pb)/pb*100:+.1f}%'
            if rr and name in rr:
                line += f'  精炼{(pr-pb)/pb*100:+.1f}%'
            line += ')'
            print(line)
        print('=' * 68)
        print('结论: A类(事实) ppl 越低, 模型对可验证知识掌握越精准。'
              '精炼版若在 A 类上进一步下降, 说明 frsm 复核有效。')
    elif args.ckpt:
        run_eval(args.ckpt, args.tag, dev)
    else:
        ap.error('需要 --ckpt 或 --compare')


if __name__ == '__main__':
    main()
