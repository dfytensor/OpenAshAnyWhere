# -*- coding: utf-8 -*-
"""NewOpenASHVoc：领域漏斗挖词 + OpenASH 数位寻址

数据：minimind pretrain_t2t_mini.jsonl（中文为主，含英文翻译任务）+ vocabulary_nnn.json(452万词)
结构：[special][chars][head words][ts×D][te×D]
  - special/字符/头部词   → 1 token
  - 词典尾部(数百万词)    → 一律 2 token（idx = hi*D + lo）
  - 全部 BMP 字符保证可编码（高频字符 1 token，低频字符在尾部 2 token）→ 无损
流水线：
  ① 漏斗挖短语：DomainMiner(jieba打底+频次+PMI+邻熵) 在 minimind 拟合段挖跨词搭配
  ② 回灌 jieba：短语 add_word，使切分与编码一致
  ③ 域频统计：minimind 频次段上统计所有 token 频次
  ④ 组装：头部词按域频取 TOP-K；词典 = voc_nnn ∪ jieba词表 ∪ 短语 ∪ 字符，按域频重排
  ⑤ 评测：留出段压缩率 / 覆盖率 / 无损回灌 / 中英文分段
"""
import json
import math
import re
import sys
import time
import unicodedata
from collections import Counter

sys.path.insert(0, r'F:\夸克\领域分词最终方案')
import jieba
import domain_miner as D

DATA = r'F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl'
VOC_NNN = r'F:\OpenASH2605\vocabulary_nnn.json'
OUT = r'F:\OpenASH2605\new_openash_voc.json'
SPLIT_RE = re.compile(r'[。！？；…\n\r]+')
N_MINE, N_FREQ, N_HELD = 12000, 60000, 4000     # 条数：挖掘 / 频次 / 留出
HEAD_K = 30000                                   # 头部词预算（默认）
PHRASE_K = 20000                                 # 漏斗短语预算
SPECIAL = ["<|pad|>", "<|im_start|>", "<|im_end|>", "<|think|>",
           "<|end_think|>", "<|user|>", "<|agent|>", "<|system|>",
           "<|func|>", "<|args|>", "<|unk|>",
           "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>",
           "<|quad_start|>", "<|quad_end|>",
           "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
           "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<tts_pad>", "<tts_text_bos>",
           "<tts_text_eod>", "<tts_text_bos_single>", "<|tools|>", "<|end_tools|>",
           "<tool_call>", "</tool_call>",
           "<tool_response>", "</tool_response>"]
sys.stdout.reconfigure(encoding='utf-8')


def to_sents(texts):
    out = []
    for t in texts:
        for s in SPLIT_RE.split(t):
            s = s.strip()
            if 4 <= len(s) <= 120:
                out.append(s)
    return out


def is_meaningful(ch):
    cat = unicodedata.category(ch)
    return cat[0] in 'LMNPS' or cat == 'Zs'


def meaningful_bmp():
    return [chr(c) for c in range(0x10000) if is_meaningful(chr(c))]


def load_ranges(n_mine=N_MINE, n_freq=N_FREQ, n_held=N_HELD, stride=40):
    """留出段按 stride 跨库抽样，避免数据集区域性偏斜"""
    texts, held = [], []
    with open(DATA, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n_freq + n_held * stride:
                break
            t = json.loads(line)['text']
            if i < n_freq:
                texts.append(t)
            elif (i - n_freq) % stride == 0 and len(held) < n_held:
                held.append(t)
    return texts[:n_mine], texts[n_mine:], held


# ==================== 流水线各步 ====================
def mine_phrases(sents_mine):
    """① 漏斗挖短语 + ② 回灌全局 jieba（须在干净 jieba 上做一次）"""
    t0 = time.time()
    m = D.DomainMiner()
    m.fit(sents_mine)
    passed = m._dedup_nested([g for g in m.candidates() if m.passes(g)])
    passed.sort(key=lambda g: -m.freq(g))
    grams = passed[:PHRASE_K]
    phrases = [''.join(g) for g in grams]
    ph_freq = {''.join(g): m.freq(g) for g in grams}
    for p in phrases:
        jieba.add_word(p, freq=max(1000, ph_freq[p] * 100), tag='n')
    print(f"① 漏斗挖掘：{len(passed)} 过漏斗 → 短语 {len(phrases)}，TOP10: {'、'.join(phrases[:10])}"
          f"  ({time.time()-t0:.0f}s)")
    return phrases, ph_freq


def count_freq(sents_freq, texts_freq):
    """③ 域频统计（短语已回灌，切分与编码一致）；字符表用原文扫描而非词级统计"""
    t0 = time.time()
    tok_freq = Counter()
    for s in sents_freq:
        tok_freq.update(jieba.lcut(s))
    char_freq = Counter()
    for t in texts_freq:
        char_freq.update(t)
    print(f"③ 域频统计：不同 token {len(tok_freq)}，总 {sum(tok_freq.values())}，"
          f"字符 {len(char_freq)} 种  ({time.time()-t0:.0f}s)")
    return tok_freq, char_freq


def assemble(tok_freq, char_freq, phrases, head_k=HEAD_K, out=OUT):
    """④ 组装词表并保存。字符表 = 频次段原文 ∪ 词典字符，保证通用覆盖"""
    t0 = time.time()
    voc_nnn = json.load(open(VOC_NNN, encoding='utf-8'))['voc']
    sp_set = set(SPECIAL)
    chars = {c for c, n in char_freq.items()
             if len(c) == 1 and is_meaningful(c)}
    for w in voc_nnn:                      # 词典里出现过的字符全部给 1-token
        for c in w:
            if len(c) == 1 and is_meaningful(c):
                chars.add(c)
    astral = {c for c in chars if ord(c) > 0xFFFF}
    chars = [c for c in chars if ord(c) <= 0xFFFF]
    chars.sort(key=lambda c: -char_freq.get(c, 0))
    chars = list(dict.fromkeys(chars + [chr(c) for c in range(32, 127)]))
    char_set = set(chars)
    head = [w for w, _ in tok_freq.most_common() if len(w) >= 2][:head_k]
    head_set = set(head)

    rank0 = {w: i for i, w in enumerate(voc_nnn)}
    tail_set = (set(voc_nnn) | set(jieba.dt.FREQ.keys()) | set(phrases)
                | set(tok_freq) | set(astral))
    tail = [w for w in tail_set
            if w and w not in sp_set and w not in char_set and w not in head_set]
    tail.sort(key=lambda w: (-tok_freq.get(w, 0), rank0.get(w, 1 << 60)))
    Dg = max(2240, int(math.ceil(math.sqrt(len(tail)) / 50) * 50))
    meta = {'special': SPECIAL, 'chars': chars, 'head': head, 'tail': tail,
            'D': Dg, 'phrases': phrases}
    json.dump(meta, open(out, 'w', encoding='utf-8'),
              ensure_ascii=False, separators=(',', ':'))
    print(f"④ 组装：chars {len(chars)} + head {len(head)} + tail {len(tail)} (D={Dg})"
          f" → 真实词表 {len(SPECIAL)+len(chars)+len(head)+2*Dg}  "
          f"({time.time()-t0:.0f}s) → {out}")
    return out


def build(head_k=HEAD_K, out=OUT):
    mine_texts, freq_texts, _ = load_ranges()
    print(f"数据：挖掘 {len(mine_texts)} 条 / 频次 {len(freq_texts)} 条")
    phrases, _ = mine_phrases(to_sents(mine_texts))
    tok_freq, char_freq = count_freq(to_sents(freq_texts), freq_texts)
    return assemble(tok_freq, char_freq, phrases, head_k, out)


# ==================== 编码器 ====================
class NewOpenASHVoc:
    def __init__(self, path=OUT):
        meta = json.load(open(path, encoding='utf-8'))
        self.special, self.chars, self.head = meta['special'], meta['chars'], meta['head']
        self.tail, self.D = meta['tail'], meta['D']
        self.phrases = meta['phrases']
        self.unk = self.special.index('<|unk|>')
        self.char_base = len(self.special)
        self.head_base = self.char_base + len(self.chars)
        self.digit_base = self.head_base + len(self.head)
        self.byte_base = self.digit_base + 2 * self.D     # 256 个隐式字节token
        self.token_to_id = {t: i for i, t in
                            enumerate(self.special + self.chars + self.head)}
        # 数位 token 不占用字符串名字，避免与真实词(如 "ts1")冲突：纯数值区间寻址
        self.tail_index = {w: i for i, w in enumerate(self.tail)}
        self._jieba = None

    @property
    def jieba(self):
        if self._jieba is None:
            self._jieba = jieba.Tokenizer()      # 独立实例，不污染全局
            self._jieba.initialize()
            for p in self.phrases:               # ★ 短语必须进编码器自己的切分器
                self._jieba.add_word(p, freq=1000, tag='n')
            self._jieba.lcut('预热')
        return self._jieba

    def _digits(self, idx):
        hi, lo = divmod(idx, self.D)
        return [self.digit_base + hi, self.digit_base + self.D + lo]

    def encode(self, text):
        ids = []
        t2i, D = self.token_to_id, self.D
        bb = self.byte_base
        for tok in self.jieba.lcut(text):
            tid = t2i.get(tok)
            if tid is not None:
                ids.append(tid)
                continue
            idx = self.tail_index.get(tok)
            if idx is not None:
                ids += self._digits(idx)
                continue
            for ch in tok:                        # 字符兜底
                tid = t2i.get(ch)
                if tid is not None:
                    ids.append(tid)
                    continue
                idx = self.tail_index.get(ch)
                if idx is not None:
                    ids += self._digits(idx)
                else:                             # ★ UTF-8 字节兜底：任意字符无损
                    ids.extend(bb + b for b in ch.encode('utf-8'))
        return ids

    def decode(self, ids):
        out, buf, bytes_buf = [], None, bytearray()
        tb, D, bb = self.digit_base, self.D, self.byte_base
        for i in ids:
            if bb <= i < bb + 256:                # 字节流：攒够一个字符再吐
                bytes_buf.append(i - bb)
                continue
            if bytes_buf:
                out.append(bytes_buf.decode('utf-8', 'replace'))
                bytes_buf = bytearray()
            if tb <= i < tb + D:
                buf = i - tb
            elif tb + D <= i < tb + 2 * D:
                if buf is not None:
                    out.append(self.tail[buf * D + (i - tb - D)])
                    buf = None
            else:
                t = self.id_to_token.get(i)
                if t is not None:
                    out.append(t)
        if bytes_buf:
            out.append(bytes_buf.decode('utf-8', 'replace'))
        return ''.join(out)

    @property
    def id_to_token(self):
        if not hasattr(self, '_itt'):
            self._itt = {i: t for t, i in self.token_to_id.items()}
        return self._itt

    def vocab_size(self):
        return self.byte_base + 256


# ==================== 评测 ====================
def evaluate(path, held_texts, tag=''):
    held_sents = to_sents(held_texts)
    nchar = sum(len(s) for s in held_sents)
    text = ''.join(held_sents)
    voc = NewOpenASHVoc(path)
    ids = voc.encode(text)

    n1 = n2 = nc = 0
    for tok in voc.jieba.lcut(text):
        if tok in voc.token_to_id:
            n1 += 1
        elif tok in voc.tail_index:
            n2 += 1
        else:
            nc += 1
    loss = sum(1 for t in held_texts[:300] if voc.decode(voc.encode(t)) != t)

    base = jieba.Tokenizer()
    base.initialize()
    jieba_n = sum(len(base.lcut(s)) for s in held_sents)     # 无短语，jieba 原生
    funnel_n = sum(len(jieba.lcut(s)) for s in held_sents)   # 全局含短语=词级上限

    import tiktoken
    cl = len(tiktoken.get_encoding('cl100k_base').encode(text))
    o2 = len(tiktoken.get_encoding('o200k_base').encode(text))

    print(f"\n===== 评测 {tag}  词表 {voc.vocab_size()} =====")
    print(f"语料 {len(held_sents)} 句 / {nchar} 字 | 命中: 头部 {n1} / 词典 {n2} / 兜底 {nc}"
          f" | 无损 {300-loss}/300")
    print(f"{'方法':<28}{'token':>10}{'字/token':>10}")
    print("-" * 50)
    for name, n in [('字符基线', nchar), ('BPE cl100k', cl), ('BPE o200k', o2),
                    ('jieba 原生', jieba_n), ('词级上限(jieba+漏斗)', funnel_n),
                    ('★ NewOpenASHVoc', len(ids))]:
        print(f"{name:<28}{n:>10}{nchar/n:>10.2f}")

    zh = ''.join(s for s in held_sents if sum(1 for c in s if ord(c) < 128) / len(s) < 0.3)
    en = ''.join(s for s in held_sents if sum(1 for c in s if ord(c) < 128) / len(s) > 0.7)
    if zh:
        print(f"  中文: {len(voc.encode(zh))/max(1,len(zh)):.2f} 字/token", end='')
    if en:
        print(f"   英文: {len(voc.encode(en))/max(1,len(en)):.2f} 字/token")
    return {'path': path, 'vocab': voc.vocab_size(), 'tokens': len(ids),
            'cpt': nchar / len(ids), 'ceiling': nchar / funnel_n, 'nchar': nchar}


def main():
    t0 = time.time()
    build()
    _, _, held_texts = load_ranges()
    evaluate(OUT, held_texts, tag='默认配置')
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
