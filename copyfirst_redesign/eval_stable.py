"""稳定版验证: 长序列 loss + M1 事实回忆 (2048/4096)."""
import os, sys, random
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import torch
import torch.nn.functional as F
from openash_reg import build_reg_model
from reg_data import build_eval_sample, make_encoder, CACHE

DEV = "cuda"
OUT = r"F:\OpenASH2605\copyfirst_redesign"


def main():
    torch.manual_seed(0)
    random.seed(0)
    enc = make_encoder()
    seqs = torch.load(CACHE, map_location="cpu", weights_only=True)[:50000]

    ckpt = torch.load(os.path.join(OUT, "openash_reg_stable.pth"), map_location="cpu", weights_only=True)
    model = build_reg_model(r"F:\OpenASH2605\models\full_sft_768_12.pth", stable=True, R=10.0)
    model.load_state_dict(ckpt["model"])
    model.to(DEV).eval()

    print("=== 1) 长序列 loss (稳定版) ===")
    for L in [256, 1024, 2048, 4096]:
        tot = 0; cnt = 0
        with torch.no_grad():
            for _ in range(8):
                filler = []
                while len(filler) < L:
                    s = seqs[random.randrange(len(seqs))]
                    filler += s[:min(s.shape[0], L - len(filler))].tolist()
                t = torch.tensor(filler[:L], dtype=torch.long, device=DEV).unsqueeze(0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out, _ = model(t)
                loss = F.cross_entropy(out[0, :-1].reshape(-1, out.shape[-1]), t[0, 1:], reduction="sum")
                tot += loss.item(); cnt += (L - 1)
        print("  L=%5d  loss=%.3f" % (L, tot / cnt), flush=True)

    print("=== 2) 事实回忆 (稳定版) ===")
    for g in [32, 512, 1024, 2048, 4096]:
        n = 32 if g <= 1024 else 16
        c = 0
        for _ in range(n):
            tokens, v_pos, vid = build_eval_sample(enc, seqs, g)
            t = torch.tensor(tokens, dtype=torch.long, device=DEV).unsqueeze(0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out, _ = model(t)
            c += (out[0, v_pos - 1].argmax().item() == vid)
        print("  gap=%5d  %.1f%%" % (g, c / n * 100), flush=True)


if __name__ == "__main__":
    main()
