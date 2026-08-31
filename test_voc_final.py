# -*- coding: utf-8 -*-
"""NewOpenASHVoc 编码/解码全面验收测试

  A. 结构校验（id 唯一、寻址空间、词表布局）
  B. 无损回灌：minimind 留出段 400 条原文（严格 ==）
  C. 边界用例：emoji / 繁体 / SPARQL / 多语言混合 / 类special串 / 空串 / 超长
  D. 跨域验证：科幻小说库 3 本抽样（回灌 + 压缩率）
"""
import sys
import time

sys.path.insert(0, r'F:\OpenASH2605')
sys.path.insert(0, r'F:\夸克\领域分词最终方案')
sys.stdout.reconfigure(encoding='utf-8')
import re
import new_openash_voc as M

HELD_TEXTS = 400
NOVELS = [r'F:\小说\科幻小说\鬼吹灯-天下霸唱.txt',
          r'F:\小说\科幻小说\盗墓笔记 (全本)-南派三叔.txt',
          r'F:\小说\科幻小说\御鬼者传奇-沙之愚者.txt']

ok = True


def check(name, cond, detail=''):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    ok = ok and cond


def roundtrip(voc, texts):
    bad = []
    for t in texts:
        if voc.decode(voc.encode(t)) != t:
            bad.append(t)
    return bad


def main():
    global ok
    voc = M.NewOpenASHVoc()
    print("=" * 70)
    print("A. 结构校验")
    print("=" * 70)
    check("词表大小", voc.vocab_size() == voc.digit_base + 2 * voc.D + 256,
          f"= {voc.vocab_size()}")
    check("id 无冲突", len(set(voc.token_to_id.values())) == len(voc.token_to_id))
    check("寻址空间 D^2 覆盖词典", voc.D * voc.D >= len(voc.tail),
          f"{voc.D}^2={voc.D*voc.D:,} >= {len(voc.tail):,}")
    check("字节兜底区就位", voc.byte_base == voc.digit_base + 2 * voc.D)

    print("\n" + "=" * 70)
    print(f"B. minimind 留出段无损回灌（{HELD_TEXTS} 条原文，严格 ==）")
    print("=" * 70)
    _, _, held_texts = M.load_ranges(n_mine=30000, n_freq=120000,
                                     n_held=HELD_TEXTS, stride=40)
    t0 = time.time()
    bad = roundtrip(voc, held_texts)
    enc_sps = sum(len(t) for t in held_texts) / max(1, time.time() - t0)
    check(f"回灌 {len(held_texts) - len(bad)}/{len(held_texts)}", not bad)
    nchar = sum(len(t) for t in held_texts)
    ntok = sum(len(voc.encode(t)) for t in held_texts)
    print(f"  压缩: {nchar:,} 字 → {ntok:,} token = {nchar/ntok:.2f} 字/token"
          f"（吞吐约 {enc_sps/10000:.0f} 万字/s）")
    if bad:
        print("  失败样例:", repr(bad[0][:80]))

    print("\n" + "=" * 70)
    print("C. 边界用例")
    print("=" * 70)
    cases = {
        "空串": "",
        "纯emoji": "🚀🔥𝄞𠀀",
        "繁体中文": "天津大學出過哪些科學家？誰於昨晚舉報了這件事。",
        "SPARQL混合": "select ?x where { ?x <职业> <科学家> . }",
        "英中混排": "The Eiffel Tower is a wrought iron lattice tower, 埃菲尔铁塔是锻铁格构塔。",
        "类special串": "文本中有 <|pad|> 和 ts1 te2 字样也不受影响",
        "数字标点": "3.1415926，……——？！【】（ ）%&*#",
        "韩文日文": "안녕하세요、こんにちは世界",
        "换行制表": "第一行\n第二行\t缩进\r\n回车",
        "超长文本": "重复压缩测试。" * 5000,
    }
    for name, t in cases.items():
        bad = roundtrip(voc, [t])
        n = len(voc.encode(t))
        check(f"{name}（{n} token）", not bad)

    print("\n" + "=" * 70)
    print("D. 跨域验证：科幻小说库")
    print("=" * 70)
    for p in NOVELS:
        raw = M.read_text if hasattr(M, 'read_text') else None
        try:
            text = open(p, 'rb').read().decode('utf-8')
        except UnicodeDecodeError:
            text = open(p, 'rb').read().decode('gb18030', 'replace')
        text = re.sub(r'\s+', '', text)[:200000]
        ntok = len(voc.encode(text))
        bad = roundtrip(voc, [text[10000:130000]])
        print(f"  [{'PASS' if not bad else 'FAIL'}] {p.split(chr(92))[-1][:22]:<24}"
              f"{len(text):>7} 字 → {ntok:>6} token = {len(text)/ntok:.2f} 字/token"
              f"  回灌{'OK' if not bad else 'FAIL'}")
        ok = ok and not bad

    print("\n" + "=" * 70)
    print("全部通过 ✔" if ok else "存在失败项 ✘")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
