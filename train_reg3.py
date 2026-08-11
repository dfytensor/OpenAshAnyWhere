"""M3 训练 + 评估: 无标记针 (学习式写入门)."""
import os, sys, time, random
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import torch
import torch.nn.functional as F
from openash_reg3 import build_reg3_model
from reg_data3 import build_needle_sample, build_needle_eval, make_encoder, CACHE

DEV = "cuda"
OUT = r"F:\OpenASH2605\copyfirst_redesign"


def main():
    torch.manual_seed(0)
    random.seed(0)
    enc = make_encoder()
    seqs = torch.load(CACHE, map_location="cpu", weights_only=True)[:200000]

    model = build_reg3_model(r"F:\OpenASH2605\models\full_sft_768_12.pth", n_slots=8).to(DEV)
    print("可训练参数: %.1fM" % (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6),
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    BATCH, STEPS = 8, 2500
    t0 = time.time()
    for step in range(STEPS):
        model.train()
        xs, ys, ws = [], [], []
        for _ in range(BATCH):
            tokens, v_pos, vid = build_needle_sample(enc, seqs, max_len=256)
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
            ce = (F.cross_entropy(out.reshape(-1, out.shape[-1]), Y.reshape(-1),
                                  reduction="none").view(BATCH, L) * W).sum() / W.sum()
            g_raw = model.register.Wg(model.base.em(X)).squeeze(-1)
            sparsity = torch.sigmoid(g_raw).mean()          # 门稀疏正则
            loss = ce + 0.02 * sparsity
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0:
            acc64, acc512 = eval_acc(model, enc, seqs)
            print("step %d loss=%.3f acc(gap64/512)=%.0f/%.0f (%.0fs)" % (
                step, loss.item(), acc64, acc512, time.time() - t0), flush=True)
    torch.save({"model": model.state_dict()}, os.path.join(OUT, "openash_reg3_sft.pth"))
    print("完成 %.0fs" % (time.time() - t0), flush=True)


@torch.no_grad()
def eval_acc(model, enc, seqs, n=24):
    model.eval()
    outs = []
    for g in [64, 512]:
        c = 0
        for _ in range(n):
            tokens, v_pos, vid = build_needle_eval(enc, seqs, g)
            t = torch.tensor(tokens, dtype=torch.long, device=DEV).unsqueeze(0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out, _ = model(t)
            c += (out[0, v_pos - 1].argmax().item() == vid)
        outs.append(c / n * 100)
    return outs


if __name__ == "__main__":
    main()
