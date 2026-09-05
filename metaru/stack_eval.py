"""堆叠模型评估: 加载 ckpt 测召回外推 + LM/生成多样性."""
import sys, os, json, time
sys.path.insert(0, r"F:\OpenASH2605\metaru")
import torch
import torch.nn as nn
import torch.nn.functional as F
from tasks import CharLM
from stack_test import StackRecall, make_lm, gen_distinct
from needle_trained import eval_at

DEV = "cuda"
RJ = r"F:\OpenASH2605\metaru\mx_results.json"
OUT = json.load(open(RJ)) if os.path.exists(RJ) else {}
ORDER = sys.argv[1] if len(sys.argv) > 1 else "x-m"


def main():
    lm = CharLM(ctx=128, device=DEV)

    print("=== 召回外推 (MSE) ===", flush=True)
    ck = r"F:\OpenASH2605\metaru\recall_stack_%s.pth" % ORDER
    if os.path.exists(ck):
        for L in [512, 1024, 2048, 4096]:
            me = StackRecall(ORDER).to(DEV)
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            me.load_state_dict(sd["model"])
            row = dict(OUT.get("recall_%d" % L, {}))
            row["stack_" + ORDER] = eval_at(me, L)
            OUT["recall_%d" % L] = row
            json.dump(OUT, open(RJ, "w"))
            print("  L=%4d: %s" % (L, "  ".join("%s=%.4f" % (k, v)
                                                for k, v in row.items())), flush=True)

    print("=== LM 评估 (500 步训练后) ===", flush=True)
    ckl = r"F:\OpenASH2605\metaru\lm_stack_%s.pth" % ORDER
    if not os.path.exists(ckl):
        torch.manual_seed(0)
        m = make_lm(ORDER, lm.vocab).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
        x0, y0 = lm.batch(256)
        gm = torch.cuda.make_graphed_callables(m, (x0,))
        t0 = time.time()
        for st in range(500):
            x, y = lm.batch(256)
            x0.copy_(x)
            loss = F.cross_entropy(gm(x0).reshape(-1, lm.vocab), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if st % 100 == 0:
                print("  %s-lm %d loss=%.4f (%.0fs)" % (ORDER, st, loss.item(),
                                                        time.time() - t0), flush=True)
            if (st + 1) % 250 == 0:
                torch.save(m.state_dict(), ckl)
                print("  lm ckpt @%d 已存" % (st + 1), flush=True)
        torch.save(m.state_dict(), ckl)
    m = make_lm(ORDER, lm.vocab).to(DEV)
    m.load_state_dict(torch.load(ckl, map_location="cpu", weights_only=True))
    m.eval()
    vx, vy = lm.batch(512)
    with torch.no_grad():
        ce = F.cross_entropy(m(vx).reshape(-1, lm.vocab), vy.reshape(-1)).item()
    d2, run = gen_distinct(m, lm)
    OUT["lm_stack_%s" % ORDER] = {"ce": ce, "distinct2": d2, "max_run": run}
    json.dump(OUT, open(RJ, "w"))
    print("  %s: CE=%.4f distinct2=%.3f" % (ORDER, ce, d2), flush=True)
    print("\n完成")


if __name__ == "__main__":
    main()
