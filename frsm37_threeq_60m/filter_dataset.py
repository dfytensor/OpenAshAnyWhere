"""
三问过滤 — 数据集处理
====================
读取 minimind 数据, 用 three_q_filter 对每条文本打标签并执行 C类配额过滤.
输出过滤后的 jsonl (保留 text 字段, 附加 tqf 标签) + 统计报告.

用法:
  python filter_dataset.py            # 处理 pretrain + sft
  python filter_dataset.py --pretrain # 仅 pretrain
"""
import json
import os
import argparse
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from three_q_filter import classify_three_q

DATA_DIR = r'F:\OpenASH2605\minimind_data'
OUT_DIR = r'F:\OpenASH2605\frsm37_threeq_60m\filtered_data'
C_QUOTA = 0.20


def _extract_text(obj):
    """兼容 pretrain(text) 与 sft(conversations) 两种格式, 取用于分类的文本."""
    if 'text' in obj:
        return obj['text']
    convs = obj.get('conversations', [])
    return '\n'.join(m.get('content', '') for m in convs if m.get('role') in ('user', 'assistant'))


def _classify_line(line):
    """worker: 解析一行并分类 (用于多进程)"""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except Exception:
        return None
    text = _extract_text(obj)
    res = classify_three_q(text)
    return res['category'], res['verifiable_score'], res['operable_score'], res['noise_score'], obj


def process_file(in_path, out_path, c_quota=C_QUOTA, workers=8, label='pretrain'):
    """对单个 jsonl 文件执行三问过滤."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t0 = time.time()
    print(f'[{label}] 读取 {in_path} ...', flush=True)
    with open(in_path, encoding='utf-8') as f:
        raw_lines = f.readlines()
    total = len(raw_lines)
    print(f'[{label}] {total} 行, 多进程分类 (workers={workers}) ...', flush=True)

    results = []
    done = 0
    CHUNK = 4000
    with ProcessPoolExecutor(max_workers=workers) as ex:
        # 用 map + chunksize 批量处理, 避免 127万 Future 对象开销
        it = ex.map(_classify_line, raw_lines, chunksize=CHUNK)
        for r in it:
            if r is not None:
                results.append(r)
            done += 1
            if done % 200000 == 0:
                print(f'  [{label}] {done}/{total} ({100*done//total}%)', flush=True)
    # 收集有效结果, 按类分桶
    valid = results
    cat_count = Counter(r[0] for r in valid)
    print(f'[{label}] 分类分布: A={cat_count["A"]} B={cat_count["B"]} C={cat_count["C"]} '
          f'(耗时 {time.time()-t0:.0f}s)', flush=True)

    # 过短文本丢弃
    MIN_LEN = 20
    a, b, c = [], [], []
    for cat, v, o, n, obj in valid:
        if len(_extract_text(obj)) < MIN_LEN:
            continue
        obj['tqf_cat'] = cat
        obj['tqf_v'] = v
        obj['tqf_o'] = o
        obj['tqf_n'] = n
        if cat == 'A': a.append(obj)
        elif cat == 'B': b.append(obj)
        else: c.append(obj)

    # Q3: C类配额 (≤c_quota 占保留总量)
    keep_core = len(a) + len(b)
    c_budget = int(keep_core * c_quota / (1 - c_quota)) if c_quota < 1 else len(c)
    c_budget = max(0, min(len(c), c_budget))
    c.sort(key=lambda x: x['tqf_v'], reverse=True)   # 相对有价值的 C 优先
    c_kept = c[:c_budget]

    kept = a + b + c_kept
    stats = {
        'label': label, 'total': total, 'valid': len(valid),
        'A': len(a), 'B': len(b), 'C_total': len(c), 'C_kept': len(c_kept),
        'kept': len(kept),
        'c_ratio_kept': round(len(c_kept) / max(len(kept), 1), 4),
        'filter_rate': round(1 - len(kept) / max(total, 1), 4),
        'time_s': round(time.time() - t0, 1),
    }

    # 写过滤后数据
    with open(out_path, 'w', encoding='utf-8') as f:
        for obj in kept:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    print(f'[{label}] 写出 {len(kept)} 条 -> {out_path}', flush=True)
    print(f'[{label}] C类占比={stats["c_ratio_kept"]:.1%} 过滤率={stats["filter_rate"]:.1%}', flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pretrain', action='store_true')
    ap.add_argument('--sft', action='store_true')
    ap.add_argument('--c_quota', type=float, default=C_QUOTA)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    do_pre = args.pretrain or (not args.pretrain and not args.sft)
    do_sft = args.sft or (not args.pretrain and not args.sft)

    all_stats = []
    if do_pre:
        all_stats.append(process_file(
            f'{DATA_DIR}/pretrain_t2t_mini.jsonl',
            f'{OUT_DIR}/pretrain_filtered.jsonl',
            args.c_quota, args.workers, 'pretrain'))
    if do_sft:
        # sft 是 conversations 格式, 取首条 user+assistant 拼接文本做分类, 保留原结构
        all_stats.append(process_file(
            f'{DATA_DIR}/sft_t2t_mini.jsonl',
            f'{OUT_DIR}/sft_filtered.jsonl',
            args.c_quota, args.workers, 'sft'))

    with open(f'{OUT_DIR}/filter_stats.json', 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print('\n=== 三问过滤完成 ===')
    for s in all_stats:
        print(json.dumps(s, ensure_ascii=False))


if __name__ == '__main__':
    main()
