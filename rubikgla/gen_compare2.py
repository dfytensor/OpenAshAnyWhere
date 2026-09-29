#!/usr/bin/env python3
"""生成对比 2: v13 (三段PT) / v13S (三段SFT) / v10 (单段PT 参照). 同提示同采样."""
import sys, os, torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"F:\OpenASH2605")
from open_ash_voc import OpenASHVoc
from deqlm_v5_30m import DEQLM30
from deqlm_v13_triloop import TriLoopLM, V, DEV

HERE = os.path.dirname(os.path.abspath(__file__))
PT = r"F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt"
PROMPT_CTX = 200


@torch.no_grad()
def gen_causal(m, prompt_ids, n_new=48, top_k=50, top_p=0.9, rep_pen=1.2, temperature=1.0):
    p = prompt_ids[-PROMPT_CTX:]
    x = torch.tensor([p], device=DEV)
    generated = []
    for step in range(n_new):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(x)
            logits = out[0] if isinstance(out, tuple) else out
        logits = logits[0, -1].float()
        for tok in set(generated + p[-20:]):
            logits[tok] /= rep_pen
        logits = logits / temperature
        if top_k > 0:
            vk, ik = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits = torch.full_like(logits, float("-inf"))
            logits[ik] = vk
        if top_p < 1.0:
            sp, si = torch.sort(logits, descending=True)
            cum = torch.cumsum(torch.softmax(sp, -1), -1)
            mask = cum - torch.softmax(sp, -1) >= top_p
            sp[mask] = float("-inf")
            logits = torch.full_like(logits, float("-inf"))
            logits[si] = sp
        prob = torch.softmax(logits, -1)
        nxt = torch.multinomial(prob, 1)
        generated.append(nxt.item())
        x = torch.cat([x, nxt.view(1, 1)], 1)
    return generated


def main():
    torch.manual_seed(42)
    voc = OpenASHVoc(agent_voc_path=r"F:\OpenASH2605\open_ash_voc_agent.json")
    seqs = torch.load(PT, map_location="cpu", weights_only=True)

    models = []
    m13 = TriLoopLM().to(DEV)
    m13.load_state_dict(torch.load(os.path.join(HERE, "deqlm13_pt_full.pth"), map_location=DEV, weights_only=True))
    m13.eval(); models.append(("v13 三段PT (2.4922)", m13))
    m13s = TriLoopLM().to(DEV)
    m13s.load_state_dict(torch.load(os.path.join(HERE, "deqlm13_sft_full.pth"), map_location=DEV, weights_only=True))
    m13s.eval(); models.append(("v13S 三段SFT", m13s))
    m10 = DEQLM30().to(DEV)
    m10.load_state_dict(torch.load(os.path.join(HERE, "deqlm10_pt_full.pth"), map_location=DEV, weights_only=True))
    m10.eval(); models.append(("v10 单段PT (2.508) 参照", m10))

    for si in (100, 5000):
        prompt = seqs[si][:PROMPT_CTX].tolist()
        print("=" * 60)
        print("样本 %d" % si)
        print("[前缀尾部]", voc.decode(prompt[-40:])[-80:])
        for name, m in models:
            torch.manual_seed(42)
            g = gen_causal(m, prompt, n_new=48)
            print("[%s]" % name, voc.decode(g)[:220])
        print()

    print("=" * 60)
    print("[短提示] 人工智能是")
    short = voc.encode("人工智能是")
    for name, m in models:
        torch.manual_seed(42)
        g = gen_causal(m, short, n_new=48)
        print("[%s]" % name, voc.decode(g)[:220])


if __name__ == "__main__":
    main()
