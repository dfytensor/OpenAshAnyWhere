import sys, os, math, torch
ROOT = r"F:\OpenASH2605"
sys.path.insert(0, ROOT); os.chdir(ROOT)
from frsmash_cd import FRSMASH_CD
from open_ash_voc import OpenASHVoc
from config import agent_voc_path

voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1
dev = torch.device("cuda:0")
torch.manual_seed(42)

configs = [
    (None, None, "no cd"),
    (150, 0.97, "cd=150/0.97"),
    (300, 0.99, "cd=300/0.99"),
]

print("=" * 80)
print("  FRSMASH-CD State Stability (H=384 L=4)")
print("=" * 80)

for cap, decay, label in configs:
    model = FRSMASH_CD(vs, 384, 8, 4, K=8, state_cap=cap, state_decay=decay).to(dev).eval()
    print()
    print("--- {} ---".format(label))
    print("  {:>8}  {:>12}  {:>12}  {:>12}".format("Seq", "logit_mean", "logit_std", "logit_max"))

    for sl in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
        x = torch.randint(1, 23000, (1, sl), device=dev)
        with torch.no_grad():
            logits = model(x)
            m = logits.mean().item()
            s = logits.std().item()
            mx = logits.max().item()
        label_sl = "{}K".format(sl // 1024) if sl >= 1024 else str(sl)
        status = " STABLE" if abs(m) < 10 and mx < 50 else " BURST!" if mx > 100 else " OK"
        print("  {:>8}  {:>12.4f}  {:>12.4f}  {:>12.4f}{}".format(label_sl, m, s, mx, status))
        sys.stdout.flush()

    del model; torch.cuda.empty_cache()

print()
print("Done")
