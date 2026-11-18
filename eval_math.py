"""eval_math.py — 数学 CoT exact-match 评测（留出题）。
贪婪生成 → 解析最后一个『算式：..=N』的数值 → 对照金标。
报告：exact-match %、算式有效率、按步数分层。"""
import argparse
import json
import re
import sys
import time

import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, r'F:\OpenASH2605')
from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from train_ab import build

DEV = 'cuda'


def extract_num(text):
    """最后一个『算式：..=N』的 N；否则『答：N』。"""
    nums = re.findall(r'算式：[^\n=]+=(\d+)', text)
    if nums:
        return int(nums[-1]), True
    m = re.findall(r'答：(\d+)', text)
    if m:
        return int(m[-1]), False
    return None, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--arch', required=True)
    ap.add_argument('--data', default=r'F:\OpenASH2605\minimind_data\math_cot_eval.jsonl')
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--gen', type=int, default=96)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding='utf-8')][: args.n]
    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    torch.manual_seed(2024)
    m = build(args.arch).cuda()
    m.load_state_dict(torch.load(args.ckpt, map_location='cpu', weights_only=True),
                      strict=False)
    m.eval()

    n_ok = n_expr = n = 0
    by_steps = {}
    show = []
    t0 = time.time()
    with torch.no_grad():
        for ri, r in enumerate(rows):
            prompt = '用户：' + r['question'] + '\n助手：'
            ids = torch.tensor([voc.encode(prompt)[-1500:]], device=DEV)
            cur, st = ids, None
            _, st = m(cur)
            gen = []
            nxt = cur[:, -1:]
            for t in range(args.gen):
                lt, st = m(nxt, state=st, pos_offset=ids.shape[1] + t)
                nxt = lt[:, -1:].argmax(-1)
                gen.append(nxt.item())
            text = voc.decode(gen)
            num, has_expr = extract_num(text)
            ok = (num == r['gold'])
            n += 1
            n_ok += ok
            n_expr += has_expr
            by_steps.setdefault(r['steps'], [0, 0])
            by_steps[r['steps']][0] += ok
            by_steps[r['steps']][1] += 1
            if ri < 3:
                show.append((ri, r['gold'], num, ok, text[:90]))
            if (ri + 1) % 250 == 0:
                print('[%d/%d] em: %.2f%% | %d ms/题' % (ri + 1, len(rows),
                      n_ok / n * 100, (time.time() - t0) / n * 1000))

    for ri, gold, num, ok, text in show:
        print('--- %d gold=%s pred=%s ok=%s | GEN: %s' % (ri, gold, num, ok, text))
    print('===== Math CoT exact-match (%s) =====' % args.ckpt)
    print('EM: %d/%d = %.2f%% | 算式行有效率: %.1f%%' % (n_ok, n, n_ok / n * 100,
          n_expr / n * 100))
    for s in sorted(by_steps):
        a, b = by_steps[s]
        print('  %d步题: %d/%d = %.1f%%' % (s, a, b, a / b * 100))
    base = args.ckpt.replace('/', '_').replace('\\', '_')
    json.dump({'ckpt': args.ckpt, 'em': n_ok / n, 'n': n, 'by_steps': by_steps},
              open('matheval_' + base + '.json', 'w'), ensure_ascii=False)
    print('saved matheval_%s.json' % base)


if __name__ == '__main__':
    main()
