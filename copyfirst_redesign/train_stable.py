"""稳定性微调: OpenASHStable(cap R=10) + FactRegister, 长序列适配."""
import os, sys, time, random
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import torch
import torch.nn.functional as F
from openash_reg import build_reg_model, OpenASHReg
from reg_data import build_sample, build_eval_sample, make_encoder, CACHE

DEV = "cuda"
OUT = r"F:\OpenASH2605\copyfirst_redesign"


def main():
    torch.manual_seed(0)
    random.seed(0)
    enc = make_encoder()
    seqs = torch.load(CACHE, map_location="cpu", weights_only=True)[:200000]

    model = build_reg_model(r"F:\OpenASH2605\models\full_sft_768_12.pth", stable=True, R=10.0).to(DEV)
    # 载入 M1 的寄存器权重
    m1 = torch.load(os.path.join(OUT, "openash_reg_sft.pth"), map_location="cpu", weights_only=True)
    reg_sd = {k: v for k, v in m1["model"].items() if k.startswith("register")}
    model.load_state_dict(reg_sd, strict=False)
    print("寄存器权重已载入 M1, 可训练参数: %.1fM" % (
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    BATCH, STEPS = 8, 1000
    t0 = time.time()
    for step in range(STEPS):
        model.train()
        xs, ys, ws = [], [], []
        for _ in range(BATCH):
            tokens, v_pos, vid = build_sample(enc, seqs, max_len=256)
            t = torch.tensor(tokens, dtype=torch.long)
            xs.append(t)
            y = t.clone(); y[:-1] = t[1:]; y[-1] = 0
            w = torch.zeros_like(y); w[v_pos - 1] = 5.0
            ys.append(y); ws.append(w)
        L = max(x.shape[0] for x in xs)
        X = torch.zeros(BATCH, L, dtype=torch.long, device=DEV)
        Y = torch.zeros(BATCH, L, dtype=torch.long, device=DEV)
        W = torch.zeros(BATCH, L, device=DEV)
        for i, (x, y, w) in enumerate(zip(xs, ys, ws)):
            X[i, :x.shape[0]] = x; Y[i, :y.shape[0]] = y; W[i, :w.shape[0]] = w
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, state = model(X)
            loss = (F.cross_entropy(out.reshape(-1, out.shape[-1]), Y.reshape(-1),
                                    reduction="none").view(BATCH, L) * W).sum() / W.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            acc = eval_acc(model, enc, seqs, [32, 512])
            print("step %d loss=%.3f acc(32/512)=%s (%.0fs)" % (
                step, loss.item(), "/".join("%.0f" % a for a in acc), time.time() - t0), flush=True)
    torch.save({"model": model.state_dict()}, os.path.join(OUT, "openash_reg_stable.pth"))
    print("完成 %.0fs" % (time.time() - t0), flush=True)


@torch.no_grad()
def eval_acc(model, enc, seqs, gaps, n=16):
    model.eval()
    accs = []
    for g in gaps:
        c = 0
        for _ in range(n):
            tokens, v_pos, vid = build_eval_sample(enc, seqs, g)
            t = torch.tensor(tokens, dtype=torch.long, device=DEV).unsqueeze(0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out, _ = model(t)
            c += (out[0, v_pos - 1].argmax().item() == vid)
        accs.append(c / n * 100)
    return accs


if __name__ == "__main__":
    main()
