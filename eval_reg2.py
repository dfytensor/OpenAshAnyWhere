"""M2 完整评估: 多事实 × 长 gap 内容寻址回忆."""
import os, sys, random
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import torch
from openash_reg2 import build_reg2_model
from reg_data2 import build_multi_eval, make_encoder, CACHE

DEV = "cuda"
OUT = r"F:\OpenASH2605\copyfirst_redesign"


def main():
    torch.manual_seed(0)
    random.seed(0)
    enc = make_encoder()
    seqs = torch.load(CACHE, map_location="cpu", weights_only=True)[:50000]

    ckpt = torch.load(os.path.join(OUT, "openash_reg2_sft.pth"), map_location="cpu", weights_only=True)
    model = build_reg2_model(r"F:\OpenASH2605\models\full_sft_768_12.pth", stable=True, R=10.0,
                             n_slots=8)
    model.load_state_dict(ckpt["model"])
    model.to(DEV).eval()

    print("事实数 | gap 64 | gap 512 | gap 1024 | gap 2048")
    for nf in [2, 3, 4]:
        row = [nf]
        for g in [64, 512, 1024, 2048]:
            n = 32 if g <= 1024 else 16
            c = 0
            for _ in range(n):
                tokens, v_pos, vid = build_multi_eval(enc, seqs, n_fact=nf, gap=g)
                t = torch.tensor(tokens, dtype=torch.long, device=DEV).unsqueeze(0)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out, _ = model(t)
                c += (out[0, v_pos - 1].argmax().item() == vid)
            row.append(c / n * 100)
        print("%5d | %.1f%% | %.1f%% | %.1f%% | %.1f%%" % tuple(row), flush=True)


if __name__ == "__main__":
    main()
