"""M1 数据: 真实中文文本 + 标记事实 + 提问, 训练/评估样本."""
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


def fact_tokens(enc, name, attr, value):
    q = enc.encode("%s%s是%s" % (name, attr, value))
    qs = enc.encode("%s%s是什么" % (name, attr))
    return q, qs, enc.encode(value)[0] if False else enc.encode(value)[0]


def build_sample(enc, seqs, max_len=256, rng=None):
    """真实文本 + 随机位置事实 + 尾部提问. 返回 (tokens, ans_pos, value_id)."""
    if rng is None:
        rng = random
    name = rng.choice(NAMES)
    attr = rng.choice(list(ATTRS))
    value = rng.choice(ATTRS[attr])
    fact = enc.encode("%s%s是%s" % (name, attr, value))
    q = enc.encode("%s%s是什么" % (name, attr))
    ans = enc.encode("答案是%s" % value)
    tail = q + ans
    seq = seqs[rng.randrange(len(seqs))]
    L = min(seq.shape[0], max_len - len(fact) - 4 - len(tail))
    if L < 10:
        L = 10
    base = seq[:L].tolist()
    pos = rng.randrange(0, max(1, L // 2))
    tokens = base[:pos] + [MARK_OPEN] + fact + [MARK_CLOSE] + base[pos:]
    tokens = tokens + [QUESTION] + tail
    tokens = tokens[:max_len]
    ans_pos = len(tokens) - 2 - 1 if False else None
    # 答案值 token 位置: "答案是" 后
    v_pos = len(tokens) - 1
    value_id = enc.encode(value)[0]
    return tokens, v_pos, value_id


def build_eval_sample(enc, seqs, gap, max_len=None, rng=None):
    """gap = MARK_CLOSE 到 QUESTION 的 token 数. 返回 (tokens, v_pos, value_id)."""
    if rng is None:
        rng = random
    name = rng.choice(NAMES)
    attr = rng.choice(list(ATTRS))
    value = rng.choice(ATTRS[attr])
    fact = enc.encode("%s%s是%s" % (name, attr, value))
    q = enc.encode("%s%s是什么" % (name, attr))
    ans = enc.encode("答案是%s" % value)
    filler = []
    need = gap
    while need > 0:
        s = seqs[rng.randrange(len(seqs))]
        take = min(s.shape[0], need)
        filler += s[:take].tolist()
        need -= take
    tokens = [MARK_OPEN] + fact + [MARK_CLOSE] + filler + [QUESTION] + q + ans
    v_pos = len(tokens) - 1
    return tokens, v_pos, enc.encode(value)[0]
