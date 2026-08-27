"""ConvASH30 SFT 模型生成样例: 采样三件套对比 (含/不含 top_p+重复抑制)."""
import sys, torch
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
from convash30 import ConvASH30
from open_ash_voc import OpenASHVoc

DEV = "cuda"
CKPT = r"F:\OpenASH2605\copyfirst_redesign\convash30_sft_full.pth"
PROMPTS = ["你好", "人工智能是什么", "给我讲一个关于环保的小故事"]

m = ConvASH30().to(DEV)
m.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
voc = OpenASHVoc(agent_voc_path=r"F:\OpenASH2605\open_ash_voc_agent.json")

configs = [
    dict(temperature=0.8, top_k=30, top_p=0.0, rep_penalty=1.0),
    dict(temperature=0.8, top_k=30, top_p=0.9, rep_penalty=1.15),
    dict(temperature=0.7, top_k=40, top_p=0.95, rep_penalty=1.2),
]
for cfg in configs:
    print("\n===== %s =====" % cfg, flush=True)
    for q in PROMPTS:
        ids = [1, 5, 67] + voc.encode(q) + [2, 67, 1, 6, 67]
        torch.manual_seed(0)
        gen = m.generate(ids, steps=60, **cfg)
        text = voc.decode(gen[0].tolist())
        ans = text[text.find(q) + len(q):][:160].replace("\n", " ")
        print("  [%s] %s" % (q, ans), flush=True)
