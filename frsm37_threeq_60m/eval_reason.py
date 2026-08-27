"""
事实推理评测 — 从"推理"角度验证三问过滤
=====================================
不同于困惑度(统计指标), 这里看模型实际回答事实题的准确率.
博客论点: 过滤掉主观噪音后, 模型对可验证事实的"判断更准" -> 事实题答对率应更高.

题库: 带标准答案关键词的可验证事实题(A类为主) + 少量常识题(B类).
判分: 生成答案中是否命中标准答案关键词(大小写/空格不敏感).
"""
import os, sys, re, torch
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONUTF8', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'F:\OpenASH2605')
from open_ash_voc import OpenASHVoc
from model import FRSMASHv37

HERE = os.path.dirname(os.path.abspath(__file__))
VS = 23006

# 事实题库: (问题, [可接受答案关键词], 类别)
QUESTIONS = [
    # A类 强可验证 (地理/首都)
    ("法国的首都是哪里?", ["巴黎"], 'A'),
    ("日本的首都是哪个城市?", ["东京"], 'A'),
    ("美国的首都是哪里?", ["华盛顿"], 'A'),
    ("英国的首都是哪个城市?", ["伦敦"], 'A'),
    ("中国的首都是哪里?", ["北京"], 'A'),
    ("俄罗斯的首都是哪里?", ["莫斯科"], 'A'),
    ("澳大利亚的首都是哪个城市?", ["堪培拉"], 'A'),
    # A类 数学
    ("7乘以8等于多少?", ["56"], 'A'),
    ("15加27等于多少?", ["42"], 'A'),
    ("100减去37等于多少?", ["63"], 'A'),
    ("9的平方是多少?", ["81"], 'A'),
    ("12乘以12等于多少?", ["144"], 'A'),
    # A类 物理/度量
    ("水的沸点是多少摄氏度?", ["100"], 'A'),
    ("水结冰的温度是多少摄氏度?", ["0"], 'A'),
    ("光速大约是每秒多少千米?", ["30万", "三十万", "299", "3亿", "300000"], 'A'),
    ("一标准大气压约等于多少千帕?", ["101", "100"], 'A'),
    # A类 定义/常识
    ("一年有多少个月?", ["12", "十二"], 'A'),
    ("一周有多少天?", ["7", "七"], 'A'),
    ("一小时有多少分钟?", ["60", "六十"], 'A'),
    ("三角形内角和是多少度?", ["180", "一百八十"], 'A'),
    ("太阳从哪个方向升起?", ["东", "东方"], 'A'),
    # A类 历史
    ("中国的首都是北京, 那么哪个朝代建都于长安?", ["唐", "汉", "长安"], 'A'),
    # B类 弱可验证 (常识/解释)
    ("什么是人工智能?", ["智能", "机器", "计算机", "模拟"], 'B'),
    ("为什么要多喝水?", ["健康", "代谢", "身体", "水分"], 'B'),
    ("阅读有什么好处?", ["知识", "学习", "理解", "增长"], 'B'),
    # A类 编程
    ("Python中输出文本用什么函数?", ["print"], 'A'),
    ("HTML的全文是什么?", ["超文本", "标记", "markup", "hypertext"], 'A'),
]


def load_model(ckpt, dev):
    m = FRSMASHv37(VS, 448, 8, 7).to(dev)
    c = torch.load(ckpt, map_location=dev, weights_only=False)
    m.load_state_dict(c['model'])
    m.eval()
    return m


def generate(m, voc, q, dev, max_new=120):
    is_ = voc.token_to_id['<|im_start|>']
    ie_ = voc.token_to_id['<|im_end|>']
    uid_ = voc.token_to_id['博士']
    aid_ = voc.token_to_id['<|agent|>']
    ids = [is_, uid_] + voc.encode(q) + [ie_, is_, aid_]
    states = [None] * m.num_ssm
    h = torch.zeros(1, 448, device=dev)
    rs = None
    pos = 0
    for tid in ids:
        lg, states, h, rs, pos = m.generate_step(torch.tensor([[tid]], device=dev), states, h, rs, pos)
    out_ids = []
    for _ in range(max_new):
        nid = lg[0].argmax().item()
        out_ids.append(nid)
        if nid == ie_:
            break
        lg, states, h, rs, pos = m.generate_step(torch.tensor([[nid]], device=dev), states, h, rs, pos)
    return voc.decode(out_ids)


def score(answer, keywords):
    a = answer.replace(' ', '').lower()
    for kw in keywords:
        if kw.replace(' ', '').lower() in a:
            return True
    return False


def evaluate(ckpt, tag, dev):
    voc = OpenASHVoc(agent_voc_path=os.path.join(r'F:\OpenASH2605', 'open_ash_voc_agent.json'))
    m = load_model(ckpt, dev)
    print(f'\n=== 推理评测 [{tag}] ===', flush=True)
    correct = {'A': 0, 'B': 0}
    total = {'A': 0, 'B': 0}
    details = []
    for q, kws, cat in QUESTIONS:
        ans = generate(m, voc, q, dev)
        ok = score(ans, kws)
        correct[cat] += int(ok)
        total[cat] += 1
        details.append((cat, ok, q, ans[:60]))
        mark = '✓' if ok else '✗'
        print(f'  {mark} [{cat}] {q} -> {ans[:50].strip()}', flush=True)
    res = {}
    for cat in 'AB':
        res[cat] = {'correct': correct[cat], 'total': total[cat],
                    'acc': round(correct[cat] / max(total[cat], 1), 4)}
    res['all'] = {'acc': round(sum(correct.values()) / max(sum(total.values()), 1), 4),
                  'correct': sum(correct.values()), 'total': sum(total.values())}
    print(f'\n  A类准确率: {res["A"]["correct"]}/{res["A"]["total"]} = {res["A"]["acc"]:.1%}', flush=True)
    print(f'  B类准确率: {res["B"]["correct"]}/{res["B"]["total"]} = {res["B"]["acc"]:.1%}', flush=True)
    print(f'  总准确率:  {res["all"]["correct"]}/{res["all"]["total"]} = {res["all"]["acc"]:.1%}', flush=True)
    return res, details


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--filtered', default=os.path.join(HERE, 'checkpoints', 'frsm37_60m_sft_final.pth'))
    ap.add_argument('--baseline', default=os.path.join(HERE, 'checkpoints', 'frsm37_60m_baseline_sft_final.pth'))
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--tag', default='model')
    args = ap.parse_args()
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.compare:
        rf, df = evaluate(args.filtered, '三问过滤', dev)
        if os.path.exists(args.baseline):
            rb, db = evaluate(args.baseline, '未过滤基线', dev)
            print('\n' + '=' * 50)
            print(f'{"指标":>8} | {"三问过滤":>8} | {"未过滤":>8} | {"差异":>8}')
            print('-' * 50)
            for k in ['A', 'B', 'all']:
                print(f'{k:>8} | {rf[k]["acc"]:>7.1%} | {rb[k]["acc"]:>7.1%} | '
                      f'{(rf[k]["acc"]-rb[k]["acc"])*100:+.1f}pp')
            print('=' * 50)
        else:
            print('\n(基线 SFT 模型不存在, 仅评估过滤版. 用 train.py --baseline --skip_pretrain --sft_epochs 1 生成)')
    elif args.ckpt:
        evaluate(args.ckpt, args.tag, dev)
    else:
        ap.error('需要 --ckpt 或 --compare')


if __name__ == '__main__':
    main()
