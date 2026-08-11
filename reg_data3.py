"""M3 数据: 无标记事实 (针) 插入真实文本, 尾部提问."""
import sys, random
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import torch
from open_ash_voc import OpenASHVoc
from openash_reg import QUESTION

CACHE = r"F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt"

NAMES = ["张三", "李四", "王五", "赵六", "孙七", "周八"]
ATTRS = {
    "喜欢的数字": ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"],
    "喜欢的颜色": ["红", "黄", "蓝", "绿", "黑", "白", "紫"],
    "喜欢的动物": ["猫", "狗", "鸟", "鱼", "马", "牛"],
}


def make_encoder():
    return OpenASHVoc(agent_voc_path=r"F:\OpenASH2605\open_ash_voc_agent.json")


def build_needle_sample(enc, seqs, max_len=256, rng=None):
    """一个无标记事实插入真实文本中部, 尾部提问. 返回 (tokens, v_pos, vid)."""
    if rng is None:
        rng = random
    name = rng.choice(NAMES)
    attr = rng.choice(list(ATTRS))
    val = rng.choice(ATTRS[attr])
    fact = enc.encode("%s%s是%s" % (name, attr, val))
    q = enc.encode("%s%s是什么" % (name, attr))
    ans = enc.encode("答案是%s" % val)
    tail = [QUESTION] + q + ans
    seq = seqs[rng.randrange(len(seqs))]
    L = min(seq.shape[0], max_len - len(fact) - 6 - len(tail))
    if L < 10:
        L = 10
    base = seq[:L].tolist()
    pos = rng.randrange(0, max(1, L - len(fact)))
    tokens = base[:pos] + fact + base[pos:]
    tokens = (tokens + tail)[:max_len]
    v_pos = len(tokens) - 1
    return tokens, v_pos, enc.encode(val)[0]


def build_needle_eval(enc, seqs, gap, rng=None):
    """针(无标记事实)到提问的 gap 距离."""
    if rng is None:
        rng = random
    name = rng.choice(NAMES)
    attr = rng.choice(list(ATTRS))
    val = rng.choice(ATTRS[attr])
    fact = enc.encode("%s%s是%s" % (name, attr, val))
    q = enc.encode("%s%s是什么" % (name, attr))
    ans = enc.encode("答案是%s" % val)
    pre = []
    for _ in range(rng.randint(1, 3)):
        pre += seqs[rng.randrange(len(seqs))][:64].tolist()
    filler = []
    need = gap
    while need > 0:
        s = seqs[rng.randrange(len(seqs))]
        take = min(s.shape[0], need)
        filler += s[:take].tolist()
        need -= take
    tokens = pre + fact + filler + [QUESTION] + q + ans
    v_pos = len(tokens) - 1
    return tokens, v_pos, enc.encode(val)[0]
