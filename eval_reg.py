"""M1 长距离评估: gap 16 -> 4096, 事实回忆准确率."""
import os, sys, random
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import torch
from openash_reg import build_reg_model
from reg_data import build_eval_sample, make_encoder, CACHE

DEV = "cuda"
OUT = r"F:\OpenASH2605\copyfirst_redesign"


def main():
    torch.manual_seed(0)
    random.seed(0)
    enc = make_encoder()
    seqs = torch.load(CACHE, map_location="cpu", weights_only=True)[:50000]

    ckpt = torch.load(os.path.join(OUT, "openash_reg_sft.pth"), map_location="cpu", weights_only=True)
    model = build_reg_model(r"F:\OpenASH2605\models\full_sft_768_12.pth")
    model.load_state_dict(ckpt["model"])
    model.to(DEV).eval()

    gaps = [16, 64, 256, 512, 1024, 2048, 4096]
    n = 32 if max(gaps) <= 1024 else 16
    print("gap | 准确率")
    for g in gaps:
        c = 0
        for _ in range(n):
            tokens, v_pos, vid = build_eval_sample(enc, seqs, g)
            t = torch.tensor(tokens, dtype=torch.long, device=DEV).unsqueeze(0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out, _ = model(t)
            c += (out[0, v_pos - 1].argmax().item() == vid)
        print("%5d | %.1f%%" % (g, c / n * 100), flush=True)


if __name__ == "__main__":
    main()
