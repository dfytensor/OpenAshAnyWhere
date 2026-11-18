"""gen_math_cot.py — Math23K 风格合成 CoT 数据（ASH-F1 ± 回看通道对照实验）。
4 族模板（两步加减 / 乘加 / 均分 / 三步混合），中文实体/人名/句式变化，
统一 CoT 格式（分析→算式→答），150K 训练 + 2K 留出（exact-match 评测）。
输出：math_cot_cached.pt（385 窗打包）+ math_cot_eval.jsonl（留出题目+金标）。"""
import json
import random
gen_math = None
import sys

import torch

sys.path.insert(0, r'F:\OpenASH2605')
from open_ash_voc import OpenASHVoc
from config import agent_voc_path

OUT_PT = r'F:\OpenASH2605\minimind_data\math_cot_cached.pt'
OUT_EVAL = r'F:\OpenASH2605\minimind_data\math_cot_eval.jsonl'
N_TRAIN = 150000
N_EVAL = 2000
WIN = 385

random.seed(2024)
voc = OpenASHVoc(agent_voc_path=agent_voc_path)

OBJS = ['苹果', '铅笔', '糖果', '书本', '气球', '橘子', '篮球', '鸡蛋', '饼干', '邮票',
        '纽扣', '贝壳', '弹珠', '贴纸', '橡皮', '尺子', '桃子', '西瓜', '盒子', '瓶子']
NAMES = ['小明', '小红', '小刚', '小丽', '小华', '小强', '小芳', '小军', '小燕', '小伟',
         '张老师', '李阿姨', '王叔叔', '赵爷爷', '刘奶奶', '陈同学']
PLACES = ['商店', '书包里', '篮子里', '桌子上', '教室里', '仓库', '果园', '货架上']


def pick(objs=None):
    return random.choice(objs or OBJS)


def person():
    return random.choice(NAMES)


def place():
    return random.choice(PLACES)


def gen_addsub():
    a = random.randint(20, 500)
    b = random.randint(5, a - 5)
    c = random.randint(5, 300)
    o = pick()
    p = person()
    order = random.random() < 0.5
    if order:
        q = (f'{p}原来有{a}个{o}，第一天用掉了{b}个，第二天又买来{c}个，'
             f'现在{p}有多少个{o}？')
        r1 = a - b
        cot = (f'分析：原有{a}个，用掉{b}个，剩下{a}-{b}={r1}个；'
               f'又买来{c}个，所以{r1}+{c}={r1 + c}。')
        expr = f'{a}-{b}+{c}={r1 + c}'
        ans = r1 + c
    else:
        q = (f'{p}原来有{a}个{o}，第一天又买了{b}个，第二天送人{c}个，'
             f'现在{p}有多少个{o}？')
        r1 = a + b
        cot = (f'分析：原有{a}个，买了{b}个，共{a}+{b}={r1}个；'
               f'送人{c}个，所以{r1}-{c}={r1 - c}。')
        expr = f'{a}+{b}-{c}={r1 - c}'
        ans = r1 - c
    return q, cot, expr, ans, 2


def gen_muladd():
    u = random.randint(3, 30)
    n = random.randint(3, 25)
    e = random.randint(1, 200)
    o = pick()
    p = person()
    pl = place()
    r1 = u * n
    q = (f'{o}每盒有{u}个，{p}买了{n}盒，{pl}里原来还有{e}个{o}，'
         f'现在{pl}里一共有多少个{o}？')
    cot = (f'分析：每盒{u}个，{n}盒共{u}×{n}={r1}个；'
           f'加上原有的{e}个，所以{r1}+{e}={r1 + e}。')
    expr = f'{u}×{n}+{e}={r1 + e}'
    ans = r1 + e
    return q, cot, expr, ans, 2


def gen_div():
    p_ = random.randint(3, 24)
    q_ = random.randint(3, 40)
    t = p_ * q_
    o = pick()
    ppl = random.randint(2, 8)
    q = (f'一共有{t}个{o}，平均分给{ppl}个人，每人能分到多少个{o}？')
    cot = (f'分析：共{t}个，分给{ppl}人，{t}÷{ppl}={t // ppl}，'
           f'每人分到{t // ppl}个。')
    expr = f'{t}÷{ppl}={t // ppl}'
    ans = t // ppl
    return q, cot, expr, ans, 1


def gen_three():
    """三步混合：(a+b)×n - e"""
    a = random.randint(2, 60)
    b = random.randint(2, 60)
    n = random.randint(2, 9)
    e = random.randint(1, 100)
    r1 = a + b
    r2 = r1 * n
    o = pick()
    q = (f'第一天收集{a}个{o}，第二天收集{b}个{o}，第三天收集的数量是前两天总和的{n}倍，'
         f'送走{e}个后还剩多少个{o}？')
    cot = (f'分析：前两天共{a}+{b}={r1}个；第三天是总和的{n}倍，{r1}×{n}={r2}个；'
           f'送走{e}个，{r2}-{e}={r2 - e}。')
    expr = f'{r1}×{n}-{e}={r2 - e}'
    ans = r2 - e
    return q, cot, expr, ans, 3


GENS = [gen_addsub, gen_muladd, gen_div, gen_three,
        gen_addsub, gen_muladd, gen_addsub]

train_texts, eval_rows = [], []
for i in range(N_TRAIN + N_EVAL):
    g = random.choice(GENS)
    q, cot, expr, ans, steps = g()
    text = f'用户：{q}\n助手：{cot}\n算式：{expr}\n答：{ans}。\n'
    if i < N_TRAIN:
        train_texts.append(text)
    else:
        eval_rows.append({'question': q, 'cot': cot, 'expr': expr,
                          'gold': ans, 'steps': steps})
    if (i + 1) % 30000 == 0:
        print(f'  {i + 1}/{N_TRAIN + N_EVAL} generated')

random.shuffle(train_texts)
stream = []
for t in train_texts:
    stream.extend(voc.encode(t))
print(f'train tokens: {len(stream):,}')
t = torch.tensor(stream, dtype=torch.long)
windows = t.unfold(0, WIN, WIN)   # 非重叠打包（43K 窗 ≈ 133MB，替代滑窗 16.5M 窗）
torch.save(windows.unsqueeze(0).chunk(1), OUT_PT)
print(f'windows: {windows.shape[0]:,} -> {OUT_PT}')
with open(OUT_EVAL, 'w', encoding='utf-8') as f:
    for r in eval_rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'eval rows: {len(eval_rows)} -> {OUT_EVAL}')
