"""M2 训练 + 评估: 多槽寄存器内容寻址."""
import os, sys, time, random
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import torch
import torch.nn.functional as F
from openash_reg2 import build_reg2_model
from reg_data2 import build_multi_sample, build_multi_eval, make_encoder, CACHE

DEV = "cuda"
OUT = r"F:\OpenASH2605\copyfirst_redesign"


def main():
    torch.manual_seed(0)
    random.seed(0)
    enc = make_encoder()
    seqs = torch.load(CACHE, map_location="cpu", weights_only=True)[:200000]

    model = build_reg2_model(r"F:\OpenASH2605\models\full_sft_768_12.pth", stable=True, R=10.0,
                             n_slots=8).to(DEV)
    print("可训练参数: %.1fM" % (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6),
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    BATCH, STEPS = 8, 2000
    t0 = time.time()
    for step in range(STEPS):
        model.train()
        n_fact = random.randint(2, 4)
        xs, ys, ws = [], [], []
        for _ in range(BATCH):
            tokens, v_pos, vid = build_multi_sample(enc, seqs, n_fact=n_fact, max_len=256)
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
        if step % 250 == 0:
            acc2, acc3 = eval_acc(model, enc, seqs)
            print("step %d loss=%.3f acc(2事实/3事实)=%.0f/%.0f (%.0fs)" % (
                step, loss.item(), acc2, acc3, time.time() - t0), flush=True)
    torch.save({"model": model.state_dict()}, os.path.join(OUT, "openash_reg2_sft.pth"))
    print("完成 %.0fs" % (time.time() - t0), flush=True)


@torch.no_grad()
def eval_acc(model, enc, seqs, n=24):
    model.eval()
    outs = []
    for nf in [2, 3]:
        c = 0
        for _ in range(n):
            tokens, v_pos, vid = build_multi_eval(enc, seqs, n_fact=nf, gap=64)
            t = torch.tensor(tokens, dtype=torch.long, device=DEV).unsqueeze(0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out, _ = model(t)
            c += (out[0, v_pos - 1].argmax().item() == vid)
        outs.append(c / n * 100)
    return outs


if __name__ == "__main__":
    main()
