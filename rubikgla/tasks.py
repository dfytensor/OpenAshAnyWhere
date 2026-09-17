# 顺序敏感任务生成器: 括号匹配 / 复制 / 反转 / FSM 状态跟踪
# 统一返回 dict(input, target, mask) — CE 只算 mask=1 位置, acc 同.
import numpy as np
import torch

PAD, SEP = 0, 5


def _pad_stack(x, y, m):
    return dict(input=torch.tensor(x), target=torch.tensor(y), mask=torch.tensor(m))


# ── 括号匹配: 输入括号串 + SEP + 0/1, 只在末位算损失 ──
BR = {0: "(", 1: ")", 2: "[", 3: "]"}
BRID = {"(": 1, ")": 2, "[": 3, "]": 4, "0": 6, "1": 7}
V_BR = 8


def _balanced(n, rng):
    for _ in range(200):
        steps = [1 if rng.random() < 0.5 else -1 for _ in range(n)]
        if sum(steps) != 0:
            continue
        h = 0
        ok = True
        for s in steps:
            h += s
            if h < 0:
                ok = False
                break
        if ok and h == 0:
            out, stack, types = [], [], []
            for s in steps:
                if s == 1:
                    t = rng.integers(0, 2)
                    types.append(t)
                    out.append(BRID["([" [t]])
                    stack.append(t)
                else:
                    t = stack.pop()
                    out.append(BRID[")]" [t]])
            return out
    return None


def _is_balanced(ids):
    st = []
    for i in ids:
        c = BR[i - 1]
        if c in "([":
            st.append(c)
        else:
            if not st or {"(": ")", "[": "]"}[st.pop()] != c:
                return False
    return not st


def gen_bracket(batch, n=64, seed=0, split="train"):
    rng = np.random.default_rng(seed)
    xs, ys, ms = [], [], []
    for _ in range(batch):
        for _try in range(50):
            if rng.random() < 0.5:
                ids = _balanced(n, rng)
                if ids is None:
                    continue
            else:
                ids = _balanced(n, rng)
                if ids is None:
                    continue
                j = int(rng.integers(0, n))
                ids[j] = BRID[")]" [BR[ids[j] - 1] in "(["] ] if False else ids[j]
                alt = {"(": ")", ")": "(", "[": "]", "]": "["}[BR[ids[j] - 1]]
                ids[j] = BRID[alt]
                if _is_balanced(ids):
                    continue
            break
        lab = "1" if _is_balanced(ids) else "0"
        x = ids + [SEP, BRID[lab]]
        y = [0] * n + [0, BRID[lab]]
        m = [0] * n + [0, 1]
        xs.append(x); ys.append(y); ms.append(m)
    return _pad_stack(xs, ys, ms), V_BR


# ── 复制 / 反转: seq + SEP + answer, answer 段算损失 ──
V_CP = 12
SYM = list(range(6, 12))


def gen_copy(batch, n=48, seed=0, reverse=False):
    rng = np.random.default_rng(seed)
    xs, ys, ms = [], [], []
    for _ in range(batch):
        seq = [int(rng.integers(0, len(SYM))) for _ in range(n)]
        ans = seq[::-1] if reverse else seq
        x = seq + [SEP] + ans
        y = [0] * len(seq) + [0] + ans
        m = [0] * len(seq) + [0] + [1] * n
        xs.append(x); ys.append(y); ms.append(m)
    return _pad_stack(xs, ys, ms), V_CP


# ── FSM 状态跟踪: 固定转移表, 随机符号流, 每个位置预测当前状态 (稠密监督) ──
N_STATE, N_SYM = 5, 6
V_FSM = 6 + N_STATE  # 符号 0-5, 状态 token 6-10


def make_fsm(seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(0, N_STATE, (N_STATE, N_SYM))


def gen_fsm(batch, n=64, fsm=None, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys, ms = [], [], []
    for _ in range(batch):
        syms = rng.integers(0, N_SYM, n)
        x, y, m = [], [], []
        st = 0
        for i in range(n):
            st = int(fsm[st][syms[i]])
            x.append(int(syms[i]))
            y.append(6 + st)
            m.append(1)
        xs.append(x); ys.append(y); ms.append(m)
    return _pad_stack(xs, ys, ms), V_FSM
