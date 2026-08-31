# -*- coding: utf-8 -*-
"""全链条测试: text → token串 → token id → 还原token串 → 还原text

逐级断言:
  ① 切分无损:      ''.join(tokens) == text
  ② token→id 映射: 每个token按 头部1码/词典2码/字节N码 映射, 手工重算 == voc.encode
  ③ id 合法性:     所有 id < vocab_size; 数位 hi,lo < D; 无 unk
  ④ id→token 还原: 从 id 流独立还原出 token 串, == 原始 token 串
  ⑤ 文本还原:      ''.join(还原token) == text
"""
import sys

sys.path.insert(0, r'F:\OpenASH2605')
sys.stdout.reconfigure(encoding='utf-8')
import json
import new_openash_voc as M

voc = M.NewOpenASHVoc()


def map_tokens(tokens):
    """② token → id（独立重算映射，与 encode 同规则）"""
    ids = []
    kinds = []
    for tok in tokens:
        tid = voc.token_to_id.get(tok)
        if tid is not None:
            ids.append(tid); kinds.append('1码:' + tok); continue
        idx = voc.tail_index.get(tok)
        if idx is not None:
            hi, lo = divmod(idx, voc.D)
            assert hi < voc.D and lo < voc.D, '③ 数位越界'
            ids += [voc.digit_base + hi, voc.digit_base + voc.D + lo]
            kinds.append(f'2码:{tok}(词典#{idx})'); continue
        for ch in tok:
            tid = voc.token_to_id.get(ch)
            if tid is not None:
                ids.append(tid); kinds.append('1码:' + ch); continue
            idx = voc.tail_index.get(ch)
            if idx is not None:
                hi, lo = divmod(idx, voc.D)
                ids += [voc.digit_base + hi, voc.digit_base + voc.D + lo]
                kinds.append(f'2码:{ch}(字符词典)'); continue
            bs = ch.encode('utf-8')
            ids += [voc.byte_base + b for b in bs]
            kinds.append(f'{len(bs)}字节:{ch}')
    return ids, kinds


def pipeline(text, verbose=False):
    # ① text → token 串
    tokens = voc.jieba.lcut(text)
    assert ''.join(tokens) == text, '① 切分有损'

    # ② token → id
    ids, kinds = map_tokens(tokens)
    assert ids == voc.encode(text), '② 手工映射与 encode() 不一致'
    assert all(0 <= i < voc.vocab_size() for i in ids), '③ id 越界'
    assert voc.unk not in ids, '③ 出现 unk（不应发生）'

    # ④⑤ id → token 串 → text（独立解码）
    rt, buf, bb = [], None, bytearray()
    for i in ids:
        if voc.byte_base <= i < voc.byte_base + 256:
            bb.append(i - voc.byte_base); continue
        if bb:
            rt.append(bb.decode('utf-8')); bb = bytearray()
        if voc.digit_base <= i < voc.digit_base + voc.D:
            buf = i - voc.digit_base
        elif voc.digit_base + voc.D <= i < voc.digit_base + 2 * voc.D:
            rt.append(voc.tail[buf * voc.D + (i - voc.digit_base - voc.D)]); buf = None
        else:
            rt.append(voc.id_to_token[i])
    if bb:
        rt.append(bb.decode('utf-8'))
    assert ''.join(rt) == text, '⑤ 文本还原不一致'
    ids2, _ = map_tokens(rt)
    assert ids2 == ids, '④ id→token→id 不是不动点（还原token再编码应得到原id流）'

    if verbose:
        print(f'    text     : {text!r}')
        print(f'    token串  : {tokens}')
        print(f'    映射明细 : {" | ".join(kinds)}')
        print(f'    ids      : {ids[:40]}{"..." if len(ids) > 40 else ""}')
        print(f'    还原token: {rt}')
        print(f'    还原text : {text!r}  ✔ 五级全对')
    return len(tokens), len(ids)


CASES = [
    ('中英混合', 'RWKV模型的推理显存占用比同规模Transformer低很多'),
    ('短语回灌', '根据所提供的产品描述，编写一个50字左右的广告文案'),
    ('词典2码', '閃電炮擊了城牆，殭屍圍住了基地'),          # 少见词走词典寻址
    ('emoji字节', '点赞👍和火箭🚀'),
    ('SPARQL', 'select ?x where { ?x <职业> <科学家> . }'),
    ('纯英文', 'The quick brown fox jumps over the lazy dog.'),
]
print('=' * 72)
print('全链条逐级验证: text → token → id → token → text')
print('=' * 72)
for name, t in CASES:
    nt, ni = pipeline(t, verbose=True)
    print(f'  [{name}] {nt} tokens → {ni} ids, 压缩 {len(t)/ni:.2f} 字/token\n')

print('---- 批量回归：留出段 300 条五级断言 ----')
_, _, held = M.load_ranges(n_mine=30000, n_freq=120000, n_held=300, stride=40)
tot_t = tot_i = 0
for t in held:
    nt, ni = pipeline(t)
    tot_t += nt; tot_i += ni
print(f'300/300 条通过 ｜ {tot_t} tokens → {tot_i} ids ｜ {sum(len(x) for x in held)/tot_i:.2f} 字/token')
print('\n全链条测试通过 ✔')
