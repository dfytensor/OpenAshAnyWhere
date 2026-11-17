"""train_grpo_v53.py v2 — GRPO（受限动作空间版）：动作 = 4 个字母 token（与 C-Eval 评测接口一致）。
每步 8 题 × G=6 采样；奖励 = 字母 == 金标；组归一化优势；损失 = -A × logπ(选中字母)。
dev 240 题训练 / val 1250 题终审（无泄漏）。"""
import argparse
import glob
import json
import math
import os
import sys
import time

import pandas as pd
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, r'F:\OpenASH2605')
from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from train_ab import build

DEV = 'cuda'
CEVAL = r'F:\OpenASH2605\ceval_data'
LETTERS = ['A', 'B', 'C', 'D']


def fmt_row(r, with_ans):
    s = f"问题：{r['question']}\nA. {r['A']}\nB. {r['B']}\nC. {r['C']}\nD. {r['D']}\n答案："
    if with_ans:
        s += r['answer'] + '\n\n'
    return s


def load_dev_pool(voc):
    files = sorted(glob.glob(os.path.join(CEVAL, '*_dev.parquet')))
    pool = []
    for f in files:
        dev = pd.read_parquet(f)
        shots = ''.join(fmt_row(r, True) for _, r in dev.iterrows())
        for _, r in dev.iterrows():
            prompt = shots + fmt_row(r, False)
            ids = voc.encode(prompt)[-1500:]
            pool.append({'ids': ids, 'gold': r['answer']})
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=r'F:\OpenASH2605\sft_v53f1\model.pt')
    ap.add_argument('--arch', default='v53f1')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--prompts_per_step', type=int, default=8)
    ap.add_argument('--group', type=int, default=6)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--out', default=r'F:\OpenASH2605\grpo_v53f1')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    ans_tok = {L: voc.encode(L)[0] for L in LETTERS}
    letter_ids = torch.tensor([ans_tok[L] for L in LETTERS], device=DEV)   # (4,)
    gold_idx = {L: i for i, L in enumerate(LETTERS)}
    pool = load_dev_pool(voc)
    print(f'dev pool: {len(pool)} questions | letter ids: {letter_ids.tolist()}')

    torch.manual_seed(2024)
    model = build(args.arch).to(DEV)
    sd = torch.load(args.base, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'base loaded | missing={len(missing)} unexpected={len(unexpected)}')
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.01)

    rng = torch.Generator(device='cpu').manual_seed(77)
    stats = {'reward': [], 'acc': [], 'loss': []}
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(pool), (args.prompts_per_step,), generator=rng).tolist()
        opt.zero_grad()
        step_loss = 0.0
        step_r, step_c, n_rows = 0.0, 0, 0
        for pi in idx:
            item = pool[pi]
            ids = torch.tensor([item['ids']], device=DEV)          # (1, Lp)
            rows = ids.repeat(args.group, 1)                        # (G, Lp)
            gold = gold_idx[item['gold']]
            # ---- rollout：prompt 前向（G 行批量），4 字母受限分布采样 ----
            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits, _ = model(rows)
            last = logits[:, -1].float()                            # (G,V)
            letter_logits = last[:, letter_ids]                     # (G,4)
            probs = torch.softmax(letter_logits / args.temp, -1)
            choices = torch.multinomial(probs, 1).squeeze(-1)       # (G,) ∈ [0,4)
            rewards = (choices == gold).float()                     # (G,)
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)                     # (G,)
            # ---- 策略梯度：带梯度重前向（rollout 的 no-grad logits 仅用于采样）----
            logits_g, _ = model(rows)
            lp_full = torch.log_softmax(logits_g[:, -1].float(), -1)  # (G,V)
            tok_lp = lp_full.gather(-1, letter_ids[choices].unsqueeze(-1)).squeeze(-1)  # logπ(选中)
            loss_pg = -(adv * tok_lp).sum() / args.group
            loss = loss_pg
            loss.backward()
            step_loss += loss.item()
            step_r += rewards.mean().item()
            step_c += (rewards == 1).float().sum().item()
            n_rows += args.group
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        stats['reward'].append(step_r / max(1, n_rows))
        stats['acc'].append(step_c / max(1, n_rows))
        stats['loss'].append(step_loss / args.prompts_per_step)
        if step % 10 == 0 or step == 1:
            print(f'[step {step:4d}] sample_acc {step_c / max(1, n_rows):.3f} '
                  f'mean_reward {step_r / max(1, n_rows):.3f} | '
                  f'loss {stats["loss"][-1]:+.4f} | {(time.time() - t0) / step:.1f} s/step')
            json.dump(stats, open(os.path.join(args.out, 'grpo_stats.json'), 'w'))

    torch.save(model.state_dict(), os.path.join(args.out, 'model.pt'))
    json.dump(stats, open(os.path.join(args.out, 'grpo_stats.json'), 'w'))
    print(f'done -> {args.out}/model.pt')


if __name__ == '__main__':
    main()
