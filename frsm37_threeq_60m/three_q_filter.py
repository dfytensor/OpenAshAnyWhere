"""
三问过滤器 (Three-Questions Filter)
================================================================
博客思想: 人类智能的高效在于"信息过滤器"——接触信息前先问三问:
  Q1 这能被验证吗?  (有没有客观真伪标准?)
  Q2 这能被操作吗?  (能否据此行动并看到结果?)
  Q3 若既不能验证也不能操作, 为什么要记住它?

当前 LLM 来者不拒地吞噬所有文本, 把"物理学定律"和"网络谣言"在统计学上等价对待,
这正是幻觉 (Hallucination) 的根因。本模块为每条训练数据打上可验证性/可操作性标签,
把 C 类(不可验证的主观噪音)控制在 20% 以下, 让模型从"清爽"的数据中学习。

分类体系 (对应博客):
  A 类 — 强可验证: 带数字/单位的事实、定义、公式、代码、历史/地理事实、度量
  B 类 — 弱可验证: 可查证但欠精确的常识、解释性知识
  C 类 — 不可验证: 观点、修辞、主观感受、推测、模糊表达 (≤20% 配额)

实现: 简单正则 (确定性、快速、无需训练)。另见 frsm_classifier.py 用小 frsm3.7 复核边界样本。
"""
import re
import json
from collections import Counter

# ============================================================
# Q1 信号 — 可验证性 (客观真伪标准)
# ============================================================
# 带数字+单位 的度量事实: "169厘米" "680千克" "24GB" "595.97"
RE_MEASURE = re.compile(
    r'\d+(?:\.\d+)?\s*(?:厘米|厘米|米|公里|千米|公斤|千克|克|吨|毫米|微米|'
    r'摄氏度|度|华氏|开尔文|焦耳|瓦|千瓦|伏|安|欧姆|赫兹|兆赫|吉赫|'
    r'GB|MB|TB|KB|字节|比特|像素|帧|dpi|fps|'
    r'年|月|日|时|分|秒|世纪|季度|周|%|‰|万分|百万分|'
    r'元|万元|亿元|美元|欧元|日元|英镑|卢布|港币|'
    r'岁|名|人|辆|架|艘|座|栋|间|台|套|件|本|册|篇|首|部|升|毫升|亩|公顷|平方米|平方千米)'
)
# 年份/日期: "1815年" "2024年3月" "公元前221年"
RE_DATE = re.compile(r'(?:公元前|公元)?\d{2,4}年(?:\d{1,2}月)?(?:\d{1,2}日)?|\d{1,2}月\d{1,2}日|\d{2,4}[-/年]\d{1,2}[-/月]\d{1,2}')
# 数学/公式: 含运算符或等式
RE_MATH = re.compile(r'\d+\s*[+\-*/×÷^]\s*\d+|[=\≈≈≈]|公式|方程|定理|公理|定律|常数|系数|概率|比例|函数')
# 代码: 编程语言特征
RE_CODE = re.compile(
    r'(?:def |function |class |import |from |return |print\(|for |while |if |else|elif |'
    r'printf|scanf|cout|System\.|#include|namespace|public |private |void |int |string |'
    r'var |let |const |=>|->|\{[^}]*\}|lambda|async|await|SELECT |INSERT |CREATE |UPDATE |DELETE )',
    re.IGNORECASE,
)
# 定义性表述: "X是指Y" "X定义为" "X指的是"
RE_DEFINE = re.compile(r'是指|定义为|指的是|所谓|简称|缩写|意思是|含义是|定义是|称为|叫做|称作')
# 事实性判断 (可查证): "X是Y的首都/省会" "X位于Y" "X发明了Y"
RE_FACT = re.compile(
    r'首都|省会|位于|发源于|成立于|建立于|发明|发现|出生|逝世|毕业于|'
    r'面积|人口|海拔|长度|宽度|高度|深度|体积|密度|速度|温度|质量|重量|'
    r'作者是|作者是|导演|主演|作曲|作词|出版|上映|播出'
)
# 化学元素/公式: H2O, CO2
RE_CHEM = re.compile(r'(?:[A-Z][a-z]?)(?:\d+(?:[.,]\d+)?)?(?:[A-Z][a-z]?\d*)+|[A-Z]{2,6}\d*')

# ============================================================
# Q2 信号 — 可操作性 (能否据此行动并看到结果)
# ============================================================
# 指令/步骤: "首先...然后..." "第一步" "1. 2. 3."
RE_STEP = re.compile(r'第[一二三四五六七八九十\d]+[步,，:：]|步骤[一二三四五六七八九十\d]|首先|然后|接着|最后|其次|第一步|第二步')
RE_NUM_LIST = re.compile(r'(?:^|\n)\s*(?:\d+[.、)）]|\([一二三四五六七八十]+\)|[一二三四五六七八十]+[、.）)])\s*', re.MULTILINE)
# 方法/操作动词: "如何制作" "怎样安装" "需要配置"
RE_ACTION = re.compile(
    r'如何|怎样|怎么|方法|步骤|流程|教程|指南|手册|'
    r'制作|实现|安装|配置|部署|运行|编译|执行|启动|创建|建立|搭建|编写|开发|设计|'
    r'设置|修改|删除|添加|插入|更新|升级|备份|恢复|重启|关闭|打开|连接|断开|'
    r'种植|养殖|烹饪|炒|煮|蒸|烤|腌制|清洗|修理|维修|操作|使用'
)
# 条件/因果 (可执行逻辑): "如果...就..." "当...时"
RE_COND = re.compile(r'如果|假如|倘若|当.{1,15}时|若.*则|否则|不然|除非|一旦|只有.*才')

# ============================================================
# C 类信号 — 不可验证的主观噪音 (观点/修辞/模糊)
# ============================================================
# 主观表态: "我觉得" "我认为" "感觉"
RE_SUBJECTIVE = re.compile(r'我觉得|我认为|我感觉|个人觉得|依我看|在我看来|说实话|老实说|老实讲')
# 模糊/不确定: "好像" "似乎" "大概" "可能也许"
RE_VAGUE = re.compile(r'好像|似乎|大概|也许|可能|估计|差不多|之类的|什么的|等等|之类的|什么的|或多或少|总之|总的来说|反正')
# 情绪修辞 (无事实): "太美了" "超级棒"
RE_RHETORIC = re.compile(r'太.{0,4}了|超级|非常.{0,6}|简直|居然|竟然|不过|可惜|幸好|幸好|太好了|棒极了|太棒了|无敌|绝了')
# 主观评价形容词 (脱离事实): "好看" "好玩" "无聊"
RE_OPINION_ADJ = re.compile(r'好看|难看|好玩|无聊|有趣|没意思|好吃|难吃|好听|难听|漂亮|丑|帅气|酷|垃圾|神作|烂')


def _count(pattern, text):
    return len(pattern.findall(text))


def classify_three_q(text):
    """
    对单条文本执行三问过滤, 返回评分与分类标签。

    返回 dict:
      verifiable_score : Q1 可验证性得分 (越高越客观)
      operable_score   : Q2 可操作性得分 (越高越可执行)
      noise_score      : C类噪音得分 (越高越主观/模糊)
      category         : 'A' | 'B' | 'C'
      verify_method    : 验证方法描述 (对应博客"验证方法字段")
      signals          : 命中的信号明细
    """
    if not text or not text.strip():
        return {'category': 'C', 'verifiable_score': 0, 'operable_score': 0,
                'noise_score': 0, 'verify_method': '空文本', 'signals': []}

    # --- Q1: 可验证性 ---
    n_meas = _count(RE_MEASURE, text)
    n_date = _count(RE_DATE, text)
    n_math = _count(RE_MATH, text)
    n_code = _count(RE_CODE, text)
    n_def = _count(RE_DEFINE, text)
    n_fact = _count(RE_FACT, text)
    n_chem = _count(RE_CHEM, text)

    verifiable_score = (n_meas * 3 + n_date * 2 + n_math * 2 + n_code * 3
                        + n_def * 2 + n_fact * 2 + n_chem * 1)

    # --- Q2: 可操作性 ---
    n_step = _count(RE_STEP, text)
    n_list = _count(RE_NUM_LIST, text)
    n_act = _count(RE_ACTION, text)
    n_cond = _count(RE_COND, text)
    operable_score = n_step * 3 + min(n_list, 5) * 2 + n_act * 2 + n_cond * 1

    # --- C类: 噪音 ---
    n_subj = _count(RE_SUBJECTIVE, text)
    n_vague = _count(RE_VAGUE, text)
    n_rhet = _count(RE_RHETORIC, text)
    n_opadj = _count(RE_OPINION_ADJ, text)
    noise_score = n_subj * 3 + n_vague * 2 + n_rhet * 1 + n_opadj * 2

    signals = []
    if n_meas: signals.append(f'度量{n_meas}')
    if n_date: signals.append(f'日期{n_date}')
    if n_code: signals.append(f'代码{n_code}')
    if n_math: signals.append(f'公式{n_math}')
    if n_def: signals.append(f'定义{n_def}')
    if n_fact: signals.append(f'事实{n_fact}')
    if n_step or n_list: signals.append(f'步骤')
    if n_act: signals.append(f'操作动词{n_act}')
    if n_subj: signals.append(f'主观{n_subj}')
    if n_vague: signals.append(f'模糊{n_vague}')

    # --- 分类决策 ---
    # A类(强可验证): 高可验证分 且 噪音低
    # B类(弱可验证): 有一定可验证或可操作信号
    # C类(不可验证): 低可验证+低可操作+高噪音, 或几乎无信息
    if verifiable_score >= 6 and noise_score <= 2:
        category = 'A'
        verify_method = '客观可查证(数字/定义/代码/事实)'
    elif verifiable_score >= 3 or operable_score >= 4:
        category = 'B'
        verify_method = '可操作或可查证(指令/常识/解释)'
    elif noise_score >= 4 and verifiable_score <= 2:
        category = 'C'
        verify_method = '低价值(主观/模糊/修辞), 限流'
    elif verifiable_score <= 1 and operable_score <= 1:
        category = 'C'
        verify_method = '信息稀薄, 限流'
    else:
        category = 'B'
        verify_method = '一般常识'

    return {
        'category': category,
        'verifiable_score': verifiable_score,
        'operable_score': operable_score,
        'noise_score': noise_score,
        'verify_method': verify_method,
        'signals': signals,
    }


def filter_corpus(records, c_quota=0.20, min_len=20):
    """
    对一组 records 执行三问过滤 (Q3: 价值判断 + C类配额).

    records: list[dict], 每个含 'text' 字段
    c_quota : C类保留比例上限 (博客要求 ≤20%)
    min_len : 过短文本直接丢弃 (信息量不足)

    返回 (kept_records, stats)
      kept_records: 过滤后的 records, 每个附加 'tqf' 标签
      stats: {'A':..,'B':..,'C':..,'dropped':..,'total':..}
    """
    tagged = []
    for r in records:
        res = classify_three_q(r.get('text', ''))
        r2 = dict(r)
        r2['tqf'] = res
        tagged.append((res['category'], r2))

    a_recs = [r for c, r in tagged if c == 'A']
    b_recs = [r for c, r in tagged if c == 'B']
    c_recs = [r for c, r in tagged if c == 'C']

    # 过滤过短文本
    a_recs = [r for r in a_recs if len(r.get('text', '')) >= min_len]
    b_recs = [r for r in b_recs if len(r.get('text', '')) >= min_len]
    c_recs = [r for r in c_recs if len(r.get('text', '')) >= min_len]

    # Q3: C类配额 — 保留全部 A+B, C类压缩到 (A+B+C) 的 c_quota 以内
    keep_core = len(a_recs) + len(b_recs)
    c_budget = int(keep_core * c_quota / (1 - c_quota)) if c_quota < 1 else len(c_recs)
    c_budget = max(0, min(len(c_recs), c_budget))
    # C类按 verifiable_score 降序保留 (相对更有价值的 C 类优先)
    c_recs.sort(key=lambda r: r['tqf']['verifiable_score'], reverse=True)
    c_kept = c_recs[:c_budget]

    kept = a_recs + b_recs + c_kept
    stats = {
        'A': len(a_recs), 'B': len(b_recs),
        'C_total': len(c_recs), 'C_kept': len(c_kept),
        'dropped_short': sum(1 for _, r in tagged if len(r.get('text', '')) < min_len),
        'kept': len(kept), 'total': len(tagged),
        'c_ratio_kept': (len(c_kept) / len(kept)) if kept else 0,
    }
    return kept, stats


if __name__ == '__main__':
    samples = [
        '拿破仑身高169厘米，体重约90千克，1815年在滑铁卢战役中战败。',
        '拿破仑很矮，长得不好看，我觉得他可能是个无聊的人，好像也没什么了不起。',
        '安装Python的方法：首先访问python.org下载安装包，然后运行安装程序，勾选Add to PATH，最后点击Install Now。',
        'def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)',
        '我觉得这部电影太棒了，超级好看，简直无敌，非常有趣！',
        '水的化学式是H2O，沸点为100摄氏度，密度约为1克每立方厘米。',
        '三国时期，魏蜀吴三分天下，大家可能都觉得曹操很厉害吧。',
    ]
    print('=== 三问过滤器示例 ===\n')
    for s in samples:
        r = classify_three_q(s)
        print(f'[{r["category"]}] V={r["verifiable_score"]} O={r["operable_score"]} N={r["noise_score"]} '
              f'| {r["verify_method"]} | {",".join(r["signals"]) or "-"}')
        print(f'    {s[:50]}{"..." if len(s)>50 else ""}')
        print()
