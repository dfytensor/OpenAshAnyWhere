"""Meta-RU vs GRU vs LSTM 三任务对比.

任务: ① adding (T=200, MSE)  ② copy (T=100, acc)  ③ char-LM (ctx=128, loss)
预算: 每任务统一步数/优化器/裁剪, 报告最优验证指标.
"""
import sys, time
sys.path.insert(0, r"F:\OpenASH2605\metaru")
import torch
import torch.nn.functional as F
from metaru import SeqModel, LMModel
from tasks import adding_batch, copy_batch, CharLM

DEV = "cuda"
D = 128
KINDS = ["metaru", "gru", "lstm"]
RESULTS = {}


def run(tag, model_fn, steps, batch_fn, loss_fn, eval_fn, eval_every=500, mode="min"):
    torch.manual_seed(0)
    model = model_fn().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    best, t0 = None, time.time()
    for st in range(steps):
        model.train()
        x, y = batch_fn()
        out = model(x)
        loss = loss_fn(out, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if st % eval_every == 0 or st == steps - 1:
            model.eval()
            with torch.no_grad():
                ev = eval_fn(model)
            better = best is None or (ev < best[0] if mode == "min" else ev > best[0])
            if better:
                best = (ev, st)
            print("  %s %d/%d loss=%.4f eval=%.4f (%.0fs)" %
                  (tag, st, steps, loss.item(), ev, time.time() - t0), flush=True)
    RESULTS[tag] = (best, time.time() - t0)
    print("  == %s best=%.4f @%d (%.0fs) ==\n" % (tag, best[0], best[1], time.time() - t0), flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    torch.manual_seed(0)
    RESULTS["add_metaru"] = ((0.0081, 2000), 5521)
    RESULTS["add_gru"] = ((0.0001, 3500), 170)
    RESULTS["add_lstm"] = ((0.0080, 1000), 206)
    if "--skip-add" in sys.argv:
        print("=== 任务1: adding T=200 (已完成, 跳过) ===", flush=True)
    else:
        print("=== 任务1: adding T=200 ===", flush=True)
        for kind in KINDS:
            def batch_fn():
                return adding_batch(128, 200, DEV)

            def eval_fn(m):
                x, y = adding_batch(1024, 200, DEV)
                return F.mse_loss(m(x).squeeze(-1), y).item()
            run("add_" + kind,
                lambda k=kind: SeqModel(k, 2, D, 1, last=True),
                4000, batch_fn,
                lambda o, y: F.mse_loss(o.squeeze(-1), y),
                eval_fn)

    print("=== 任务2: copy T=100 ===", flush=True)
    for kind in KINDS:
        def batch_fn():
            return copy_batch(128, 100, 10, DEV)

        def eval_fn(m):
            x, y = copy_batch(512, 100, 10, DEV)
            return ((m(x).argmax(-1) == y) & (y != -100)).float().mean().item()
        run("copy_" + kind,
            lambda k=kind: LMModel(k, 10, D),
            4000, batch_fn,
            lambda o, y: F.cross_entropy(o.reshape(-1, 10), y.reshape(-1), ignore_index=-100),
            eval_fn, mode="max")

    print("=== 任务3: char-LM ctx=128 ===", flush=True)
    lm = CharLM(ctx=128, device=DEV)
    print("  语料字符数: %d, vocab: %d" % (len(lm.data), lm.vocab), flush=True)
    vx, vy = lm.batch(256)
    for kind in KINDS:
        def batch_fn():
            return lm.batch(64)

        def eval_fn(m):
            o = m(vx)
            return F.cross_entropy(o.reshape(-1, o.shape[-1]), vy.reshape(-1)).item()
        run("lm_" + kind,
            lambda k=kind: LMModel(k, lm.vocab, 256),
            3000, batch_fn,
            lambda o, y: F.cross_entropy(o.reshape(-1, o.shape[-1]), y.reshape(-1)),
            eval_fn, eval_every=500)

    print("\n===== 汇总 =====")
    for k, v in RESULTS.items():
        print("  %s: best=%.4f @step %d (%.0fs)" % (k, v[0][0], v[0][1], v[1]))


if __name__ == "__main__":
    main()
