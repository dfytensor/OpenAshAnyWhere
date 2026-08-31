"""三模型各自调优 (adding T=200): Meta-RU 变体扫描 + GRU/LSTM 超参扫描."""
import sys, time, itertools
sys.path.insert(0, r"F:\OpenASH2605\metaru")
import torch
import torch.nn.functional as F
from metaru import SeqModel
from tasks import adding_batch

DEV = "cuda"
STEPS = 2500
RESULTS = []


def run(tag, model_fn, lr):
    torch.manual_seed(0)
    model = model_fn().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best, t0 = None, time.time()
    for st in range(STEPS):
        model.train()
        x, y = adding_batch(128, 200, DEV)
        loss = F.mse_loss(model(x).squeeze(-1), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if st % 500 == 0 or st == STEPS - 1:
            model.eval()
            with torch.no_grad():
                xe, ye = adding_batch(1024, 200, DEV)
                ev = F.mse_loss(model(xe).squeeze(-1), ye).item()
            if best is None or ev < best:
                best = ev
            print("  %s %d/%d loss=%.4f eval=%.4f (%.0fs)" %
                  (tag, st, STEPS, loss.item(), ev, time.time() - t0), flush=True)
    RESULTS.append((tag, best, time.time() - t0))
    print("  == %s best=%.5f (%.0fs) ==\n" % (tag, best, time.time() - t0), flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    torch.manual_seed(0)

    print("=== Meta-RU 变体扫描 ===", flush=True)
    variants = [
        ("metaru_v0_base", dict()),
        ("metaru_v1_eta0", dict(eta=0.0)),
        ("metaru_v2_eta0_ga", dict(eta=0.0, gated_a=True)),
        ("metaru_v3_eta0_ga_noclamp", dict(eta=0.0, gated_a=True, clamp_h=False)),
        ("metaru_v4_eta005_ga_noclamp", dict(eta=0.005, gated_a=True, clamp_h=False)),
        ("metaru_v5_eta0_ga_rinit05", dict(eta=0.0, gated_a=True, r_init=0.5)),
    ]
    for tag, kw in variants:
        for lr in [1e-3, 3e-4]:
            run("%s_lr%g" % (tag, lr), lambda kw=kw: SeqModel("metaru", 2, 128, 1, **kw), lr)

    print("=== GRU / LSTM 超参扫描 ===", flush=True)
    for kind in ["gru", "lstm"]:
        for lr, layers, d in itertools.product([1e-3, 3e-4], [1, 2], [128, 256]):
            run("%s_lr%g_L%d_d%d" % (kind, lr, layers, d),
                lambda kind=kind, layers=layers, d=d: SeqModel(kind, 2, d, 1, layers=layers), lr)

    print("\n===== 汇总 (adding T=200, 按 best 排序) =====")
    for tag, best, el in sorted(RESULTS, key=lambda r: r[1]):
        print("  %-32s best=%.5f (%.0fs)" % (tag, best, el))


if __name__ == "__main__":
    main()
