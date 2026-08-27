"""
事实置信度评测 — 模型对"正确事实答案"的对数概率
================================================
比贪心准确率更敏感: 不看模型"碰巧生成"什么, 而看它给"已知正确答案"分配多少概率.
博客论点的直接体现: 过滤掉噪音后, 模型对可验证事实应更"确信" -> 正确答案的 NLL 更低.

方法: 对每题, 给模型 prompt(问题+答:), 测它在正确答案 token 上的平均 NLL.
NLL 越低 = 模型越确信这个事实 = 推理判断越准.
"""
import os, sys, torch, torch.nn.functional as F
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONUTF8', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'F:\OpenASH2605')
from open_ash_voc import OpenASHVoc
from model import FRSMASHv37

HERE = os.path.dirname(os.path.abspath(__file__))
VS = 23006
SEQ_MAX = 320

# (问题, 正确参考答案) — A类可验证事实
FACT_PAIRS = [
    ("法国的首都是", "巴黎"),
    ("日本的首都是", "东京"),
    ("美国的首都是", "华盛顿"),
    ("英国的首都是", "伦敦"),
    ("中国的首都是", "北京"),
    ("俄罗斯的首都是", "莫斯科"),
    ("7乘以8等于", "56"),
    ("15加27等于", "42"),
    ("100减去37等于", "63"),
    ("9的平方等于", "81"),
    ("12乘12等于", "144"),
    ("水的沸点是", "100摄氏度"),
    ("水结冰的温度是", "0摄氏度"),
    ("一年有", "12个月"),
    ("一周有", "7天"),
    ("一小时有", "60分钟"),
    ("三角形内角和是", "180度"),
    ("太阳从", "东方升起"),
    ("圆周率约为", "3.14"),
    ("光速约为每秒", "30万千米"),
    ("中国的最长河流是", "长江"),
    ("世界上最大的洋是", "太平洋"),
    ("人体正常体温约为", "37摄氏度"),
    ("一年有", "365天"),
]


def load_model(ckpt, dev):
    m = FRSMASHv37(VS, 448, 8, 7).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)['model'])
    m.eval()
    return m


@torch.no_grad()
def answer_nll(m, voc, prompt, answer, dev):
    """模型在 answer token 上的平均 NLL (给定 prompt). 用 teacher-forcing."""
    pids = voc.encode(prompt)
    aids = voc.encode(answer)
    ids = pids + aids
    if len(ids) > SEQ_MAX:
        ids = ids[:SEQ_MAX]
    x = torch.tensor([ids], device=dev)
    logits = m(x.clamp(0, VS - 1))
    # answer 部分的 NLL
    n_ans = len(aids)
    start = len(pids) - 1
    ans_logits = logits[0, start:start + n_ans]
    ans_tgt = torch.tensor(aids, device=dev).clamp(0, VS - 1)
    nll = F.cross_entropy(ans_logits.float(), ans_tgt, reduction='mean').item()
    return nll, float(torch.exp(torch.tensor(nll)))


def evaluate(ckpt, tag, dev):
    voc = OpenASHVoc(agent_voc_path=os.path.join(r'F:\OpenASH2605', 'open_ash_voc_agent.json'))
    m = load_model(ckpt, dev)
    nlls = []
    ppls = []
    print(f'\n=== 事实置信度 [{tag}] ===', flush=True)
    for prompt, ans in FACT_PAIRS:
        nll, ppl = answer_nll(m, voc, prompt, ans, dev)
        nlls.append(nll)
        ppls.append(ppl)
        print(f'  {prompt}→{ans}: nll={nll:.3f} ppl={ppl:.2f}', flush=True)
    avg_nll = sum(nlls) / len(nlls)
    avg_ppl = sum(ppls) / len(ppls)
    print(f'  平均: nll={avg_nll:.4f}  ppl={avg_ppl:.3f}', flush=True)
    return {'avg_nll': avg_nll, 'avg_ppl': avg_ppl, 'n': len(nlls)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--filtered', default=os.path.join(HERE, 'checkpoints', 'frsm37_60m_pretrain_final.pth'))
    ap.add_argument('--baseline', default=os.path.join(HERE, 'checkpoints', 'frsm37_60m_baseline_pretrain_final.pth'))
    ap.add_argument('--compare', action='store_true')
    args = ap.parse_args()
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.compare:
        rf = evaluate(args.filtered, '三问过滤', dev)
        rb = evaluate(args.baseline, '未过滤基线', dev)
        dn = rf['avg_nll'] - rb['avg_nll']
        dp = (rf['avg_ppl'] - rb['avg_ppl']) / rb['avg_ppl'] * 100
        print('\n' + '=' * 52)
        print(f'{"指标":>10} | {"三问过滤":>10} | {"未过滤":>10} | {"变化":>8}')
        print('-' * 52)
        print(f'{"答案NLL":>10} | {rf["avg_nll"]:>10.4f} | {rb["avg_nll"]:>10.4f} | {dn:+.4f} {"↓更确信" if dn<0 else ""}')
        print(f'{"答案PPL":>10} | {rf["avg_ppl"]:>10.3f} | {rb["avg_ppl"]:>10.3f} | {dp:+.1f}% {"↓更好" if dp<0 else ""}')
        print('=' * 52)
        print('解读: 答案 NLL 越低 = 模型对正确事实越确信 = 推理判断越准。')
        print('      若过滤版 NLL 更低, 即"知道得更对"的推理层面证据。')


if __name__ == '__main__':
    main()
