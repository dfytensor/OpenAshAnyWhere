"""M2 数据: 每文档多事实 + 提问, 内容寻址回忆."""
import sys, random
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import torch
from open_ash_voc import OpenASHVoc
from openash_reg import MARK_OPEN, MARK_CLOSE, QUESTION

CACHE = r"F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt"

NAMES = ["张三", "李四", "王五", "赵六", "孙七", "周八"]
ATTRS = {
    "喜欢的数字": ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"],
    "喜欢的颜色": ["红", "黄", "蓝", "绿", "黑", "白", "紫"],
    "喜欢的动物": ["猫", "狗", "鸟", "鱼", "马", "牛"],
}


def make_encoder():
    return OpenASHVoc(agent_voc_path=r"F:\OpenASH2605\open_ash_voc_agent.json")


def build_multi_sample(enc, seqs, n_fact=3, max_len=256, rng=None):
    """n_fact 个事实 (不同人名), 提问其中一个."""
    if rng is None:
        rng = random
    names = rng.sample(NAMES, n_fact)
    facts = []
    for nm in names:
        attr = rng.choice(list(ATTRS))
        val = rng.choice(ATTRS[attr])
        facts.append((nm, attr, val))
    ask = rng.randrange(n_fact)
    nm_q, attr_q, val_q = facts[ask]
    q_toks = enc.encode("%s%s是什么" % (nm_q, attr_q))
    ans = enc.encode("答案是%s" % val_q)
    tail = [QUESTION] + q_toks + ans
    seq = seqs[rng.randrange(len(seqs))]
    pieces = []
    span = max(4, (max_len - len(tail)) // (n_fact + 1))
    pos = 0
    for i, (nm, attr, val) in enumerate(facts):
        f = enc.encode("%s%s是%s" % (nm, attr, val))
        chunk = seq[pos:pos + span]
        pieces += chunk.tolist() + [MARK_OPEN] + f + [MARK_CLOSE]
        pos += span
    pieces += seq[pos:pos + span].tolist()
    tokens = (pieces + tail)[:max_len]
    v_pos = len(tokens) - 1
    return tokens, v_pos, enc.encode(val_q)[0]


def build_multi_eval(enc, seqs, n_fact=3, gap=256, rng=None):
    """n_fact 事实, 提问其中第 ask 个; gap = 提问事实的 MARK_CLOSE 到 QUESTION 的距离."""
    if rng is None:
        rng = random
    names = rng.sample(NAMES, n_fact)
    facts = []
    for nm in names:
        attr = rng.choice(list(ATTRS))
        val = rng.choice(ATTRS[attr])
        facts.append((nm, attr, val))
    ask = rng.randrange(n_fact)
    nm_q, attr_q, val_q = facts[ask]
    q_toks = enc.encode("%s%s是什么" % (nm_q, attr_q))
    ans = enc.encode("答案是%s" % val_q)
    # 事实块: ask 之前放 n_fact-1 个事实+填充, ask 的事实后放 gap 填充, 再提问
    pre = []
    for i, (nm, attr, val) in enumerate(facts):
        if i == ask:
            continue
        f = enc.encode("%s%s是%s" % (nm, attr, val))
        pre += [MARK_OPEN] + f + [MARK_CLOSE]
        pre += seqs[rng.randrange(len(seqs))][:40].tolist()
    fq = enc.encode("%s%s是%s" % (nm_q, attr_q, val_q))
    filler = []
    need = gap
    while need > 0:
        s = seqs[rng.randrange(len(seqs))]
        take = min(s.shape[0], need)
        filler += s[:take].tolist()
        need -= take
    tokens = pre + [MARK_OPEN] + fq + [MARK_CLOSE] + filler + [QUESTION] + q_toks + ans
    v_pos = len(tokens) - 1
    return tokens, v_pos, enc.encode(val_q)[0]
