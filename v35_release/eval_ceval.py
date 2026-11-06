"""C-Eval 评测：48 科目，dev 5-shot + val 评测，选项字母 logit argmax。

用法: python eval_ceval.py --ckpt big_v35/model.pt --arch v35
      python eval_ceval.py --ckpt full_v35/model.pt --arch v35
"""
import argparse
import glob
import json
import math
import os
import sys

import pandas as pd
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
from open_ash_voc import OpenASHVoc
from open_ash_v3 import OpenASH_V3
from config import agent_voc_path

CEVAL = r'ceval_data'
DEV = 'cuda'
LETTERS = ['A', 'B', 'C', 'D']


def build(arch):
    if arch.startswith('tf'):
        from baseline_tf import TransformerLM
        return TransformerLM(23004, 768, 8, 11, max_seq_len=2048)
    kw = {}
    if arch in ('v35', 'v35s', 'v34'):
        kw = dict(pre_norm=True, ffn_act='silu', decay_init=(0.0001, 0.35), decay_v_std=0.04,
                  kv_channel=True, kv_decay=0.98)
        if arch in ('v35', 'v34'):
            kw.update(kv_norm_keys=True, kv_beta=True)
    m = OpenASH_V3(23004, 768, 8, 12, max_seq_len=2048, model_flag='train',
                   use_fused=True, use_decay=True, **kw)
    return m


def load_ckpt(m, path):
    sd = torch.load(path, map_location='cpu', weights_only=True)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        print(f'  [warn] missing keys: {len(missing)} (e.g. {missing[:2]})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--arch', default='v35')
    args = ap.parse_args()

    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    torch.manual_seed(2024)
    m = build(args.arch).to(DEV)
    load_ckpt(m, args.ckpt)
    m.eval()

    ans_tok = {L: voc.encode(L)[0] for L in LETTERS}
    print('letter token ids:', ans_tok)

    @torch.no_grad()
    def answer_letter(model, prompt):
        ids = voc.encode(prompt)[-1500:]
        x = torch.tensor([ids], device=DEV)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(x)
        last = logits[0, -1].float()
        return max(LETTERS, key=lambda L: last[ans_tok[L]].item())

    def fmt_row(r, with_ans):
        s = f"问题：{r['question']}\nA. {r['A']}\nB. {r['B']}\nC. {r['C']}\nD. {r['D']}\n答案："
        if with_ans:
            s += r['answer'] + '\n\n'
        return s

    files = sorted(glob.glob(os.path.join(CEVAL, '*_dev.parquet')))
    subjects = [os.path.basename(f).replace('_dev.parquet', '') for f in files]

    per_subject, total_correct, total_n = {}, 0, 0
    for si, subj in enumerate(subjects):
        dev = pd.read_parquet(os.path.join(CEVAL, f'{subj}_dev.parquet'))
        val_pq = os.path.join(CEVAL, f'{subj}_val.parquet')
        if not os.path.exists(val_pq):
            try:
                d = fetch = None
                import urllib.request
                req = urllib.request.Request(
                    f'https://hf-mirror.com/datasets/ceval/ceval-exam/resolve/main/{subj}/val-00000-of-00001.parquet',
                    headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=60) as rr:
                    open(val_pq, 'wb').write(rr.read())
            except Exception:
                continue
        val = pd.read_parquet(val_pq)
        if 'answer' not in val.columns:
            continue
        shots = ''.join(fmt_row(r, True) for _, r in dev.iterrows())
        correct = 0
        for _, r in val.iterrows():
            prompt = shots + fmt_row(r, False)
            pred = answer_letter(m, prompt)
            correct += (pred == r['answer'])
        acc = correct / len(val)
        per_subject[subj] = acc
        total_correct += correct
        total_n += len(val)
        if (si + 1) % 8 == 0 or si == len(subjects) - 1:
            print(f'  [{si + 1}/{len(subjects)}] running acc: {total_correct / total_n * 100:.2f}% '
                  f'({total_correct}/{total_n})')

    acc = total_correct / total_n
    print(f'\n===== C-Eval ({args.ckpt}) =====')
    print(f'total: {total_n} questions | acc: {acc * 100:.2f}%')
    for s, a in sorted(per_subject.items(), key=lambda x: -x[1])[:8]:
        print(f'  top: {s} {a * 100:.0f}%')
    for s, a in sorted(per_subject.items(), key=lambda x: x[1])[:8]:
        print(f'  low: {s} {a * 100:.0f}%')
    out = {'ckpt': args.ckpt, 'acc': acc, 'n': total_n, 'per_subject': per_subject}
    base = os.path.basename(os.path.dirname(args.ckpt)) or os.path.splitext(os.path.basename(args.ckpt))[0]
    json.dump(out, open(f'ceval_{base}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'saved ceval_{base}.json')


if __name__ == '__main__':
    main()
