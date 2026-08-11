"""M3 完整评估: 无标记针, 长 gap."""
import os, sys, random
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import torch
from openash_reg3 import build_reg3_model
from reg_data3 import build_needle_eval, make_encoder, CACHE

DEV = "cuda"
OUT = r"F:\OpenASH2605\copyfirst_redesign"


def main():
    torch.manual_seed(0)
    random.seed(0)
    enc = make_encoder()
    seqs = torch.load(CACHE, map_location="cpu", weights_only=True)[:50000]

    ckpt = torch.load(os.path.join(OUT, "openash_reg3_sft.pth"), map_location="cpu", weights_only=True)
    model = build_reg3_model(r"F:\OpenASH2605\models\full_sft_768_12.pth", n_slots=8)
    model.load_state_dict(ckpt["model"])
    model.to(DEV).eval()

    print("gap | 准确率")
    for g in [32, 64, 256, 512, 1024, 2048, 4096]:
        n = 32 if g <= 1024 else 16
        c = 0
        for _ in range(n):
            tokens, v_pos, vid = build_needle_eval(enc, seqs, g)
            t = torch.tensor(tokens, dtype=torch.long, device=DEV).unsqueeze(0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out, _ = model(t)
            c += (out[0, v_pos - 1].argmax().item() == vid)
        print("%5d | %.1f%%" % (g, c / n * 100), flush=True)


if __name__ == "__main__":
    main()
