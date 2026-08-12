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
# 模板变体 (语义匹配): 事实/问题的多样化措辞
FACT_TMPL = ["%s%s是%s", "%s%s是%s", "%s最爱的%s是%s", "%s偏好的%s是%s"]
Q_TMPL = ["%s%s是什么", "%s%s是多少", "%s%s是什么%s"]


def make_encoder():
    return OpenASHVoc(agent_voc_path=r"F:\OpenASH2605\open_ash_voc_agent.json")


def fact_str(name, attr, val, rng):
    t = rng.choice(FACT_TMPL)
    if t.count("%s") == 3:
        return t % (name, attr, val)
    return t % (name, attr, val)


def q_str(name, attr, rng):
    t = rng.choice(Q_TMPL)
    if t.count("%s") == 3:
        return t % (name, attr, attr.replace("喜欢的", ""))
    return t % (name, attr)


def build_needle_sample(enc, seqs, max_len=256, n_fact=1, rng=None):
    """n_fact 个无标记事实 (不同人名) 插入真实文本, 尾部提问其中一个.
    返回 (tokens, v_pos, vid, fact_mask): fact_mask 标记事实 span 位置 (监督门)."""
    if rng is None:
        rng = random
    n_fact = rng.randint(1, n_fact)
    names = rng.sample(NAMES, n_fact)
    facts = []
    for nm in names:
        attr = rng.choice(list(ATTRS))
        val = rng.choice(ATTRS[attr])
        facts.append((nm, attr, val))
    ask = rng.randrange(n_fact)
    nm_q, attr_q, val_q = facts[ask]
    fact_toks = [enc.encode(fact_str(nm, a, v, rng)) for nm, a, v in facts]
    q = enc.encode(q_str(nm_q, attr_q, rng))
    ans = enc.encode("答案是%s" % val_q)
    tail = [QUESTION] + q + ans
    seq = seqs[rng.randrange(len(seqs))]
    span = max(6, (max_len - len(tail) - 20) // (n_fact + 1))
    base = seq[:min(seq.shape[0], (n_fact + 1) * span)].tolist()
    tokens = []
    mask = []
    for i, f in enumerate(fact_toks):
        seg = base[i * span:(i + 1) * span]
        tokens += seg
        mask += [0] * len(seg)
        tokens += f
        mask += [1] * len(f)
    seg = base[n_fact * span:]
    tokens += seg
    mask += [0] * len(seg)
    tokens = (tokens + tail)[:max_len]
    mask = (mask + [0] * len(tail))[:max_len]
    v_pos = len(tokens) - 1
    return tokens, v_pos, enc.encode(val_q)[0], mask


def build_needle_eval(enc, seqs, gap, n_fact=1, rng=None):
    """针(无标记事实)到提问的 gap 距离; n_fact>1 时前放其他事实."""
    if rng is None:
        rng = random
    n_fact = rng.randint(1, n_fact)
    names = rng.sample(NAMES, n_fact)
    facts = []
    for nm in names:
        attr = rng.choice(list(ATTRS))
        val = rng.choice(ATTRS[attr])
        facts.append((nm, attr, val))
    ask = rng.randrange(n_fact)
    nm_q, attr_q, val_q = facts[ask]
    pre = []
    for i, (nm, attr, val) in enumerate(facts):
        if i == ask:
            continue
        pre += enc.encode(fact_str(nm, attr, val, rng))
        pre += seqs[rng.randrange(len(seqs))][:48].tolist()
    fact = enc.encode(fact_str(nm_q, attr_q, val_q, rng))
    q = enc.encode(q_str(nm_q, attr_q, rng))
    ans = enc.encode("答案是%s" % val_q)
    filler = []
    need = gap
    while need > 0:
        s = seqs[rng.randrange(len(seqs))]
        take = min(s.shape[0], need)
        filler += s[:take].tolist()
        need -= take
    tokens = pre + fact + filler + [QUESTION] + q + ans
    v_pos = len(tokens) - 1
    return tokens, v_pos, enc.encode(val_q)[0]
