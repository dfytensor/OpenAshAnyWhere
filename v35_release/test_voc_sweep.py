# -*- coding: utf-8 -*-
"""扫描 HEAD_K：逼近词级上限（jieba+漏斗）

挖掘/频次语料扩量：N_MINE=30000, N_FREQ=120000
对比 K ∈ {20000, 50000, 100000}，最优者覆盖到 new_openash_voc.json
"""
import shutil
import sys
import time

sys.path.insert(0, r'F:\OpenASH2605')
sys.stdout.reconfigure(encoding='utf-8')
import new_openash_voc as M


def main():
    t0 = time.time()
    mine_texts, freq_texts, held_texts = M.load_ranges(n_mine=30000, n_freq=120000, n_held=4000)
    print(f"挖掘 {len(mine_texts)} 条 / {sum(len(t) for t in mine_texts)} 字，"
          f"频次 {len(freq_texts)} 条 / {sum(len(t) for t in freq_texts)} 字")

    phrases, _ = M.mine_phrases(M.to_sents(mine_texts))
    tok_freq, char_freq = M.count_freq(M.to_sents(freq_texts), freq_texts)

    rows = []
    for K in (20000, 50000, 100000):
        out = rf'F:\OpenASH2605\new_openash_voc_k{K}.json'
        M.assemble(tok_freq, char_freq, phrases, head_k=K, out=out)
        rows.append(M.evaluate(out, held_texts, tag=f'HEAD_K={K}'))

    print('\n' + '=' * 62)
    print(f"{'K':>8}{'词表':>9}{'token':>10}{'字/token':>10}{'上限':>8}")
    print('-' * 62)
    for r in rows:
        print(f"{r['path'].split('_k')[-1].split('.')[0]:>8}{r['vocab']:>9,}{r['tokens']:>10,}"
              f"{r['cpt']:>10.2f}{r['ceiling']:>8.2f}")

    best = max(rows, key=lambda r: r['cpt'])
    shutil.copyfile(best['path'], M.OUT)
    print(f"\n最优 K={best['path'].split('_k')[-1].split('.')[0]} "
          f"({best['cpt']:.2f} 字/token, 上限 {best['ceiling']:.2f})，"
          f"已达上限的 {best['cpt']/best['ceiling']*100:.1f}% → 已覆盖 {M.OUT}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
