"""推理能力专项: 训练后模型的 ①长上下文外推 ②流内寻针 ③生成重复度.

与训练 loss (teacher forcing) 不同, 这里测真实推理: 长状态外推、精确回忆、生成退化.
"""
import sys, os, json, time
sys.path.insert(0, r"F:\OpenASH2605\metaru")
import torch
import torch.nn.functional as F
from bench_xgru import build_lm, B
from tasks import CharLM

DEV = "cuda"
KINDS = ["gru", "gru2", "lstm", "metagru", "xgru"]
OUT = {}
RJ = r"F:\OpenASH2605\metaru\xgru_infer.json"
if os.path.exists(RJ):
    OUT = json.load(open(RJ))


def train_lm(kind, lm, steps=500):
    ckpt = r"F:\OpenASH2605\metaru\infer_ckpt_%s.pth" % kind
    if os.path.exists(ckpt):
        model = build_lm(kind, lm.vocab, 256).to(DEV)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        return model
    torch.manual_seed(0)
    model = build_lm(kind, lm.vocab, 256).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    for st in range(steps):
        x, y = lm.batch(256)
        loss = F.cross_entropy(model(x).reshape(-1, model.head.out_features), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    torch.save(model.state_dict(), ckpt)
    return model


@torch.no_grad()
def ce_at_ctx(model, lm, ctx, n=8, bs=32):
    """长上下文 CE: 取 ctx 长窗口, 只统计后半段 (状态充分建立后的预测质量)."""
    tot, n_tok = 0.0, 0
    for _ in range(n):
        i = torch.randint(0, len(lm.data) - ctx - 1, (bs,), device=DEV)
        x = torch.stack([lm.data[j:j + ctx] for j in i])
        y = torch.stack([lm.data[j + 1:j + ctx + 1] for j in i])
        logits = model(x)
        half = ctx // 2
        l = F.cross_entropy(logits[:, half:].reshape(-1, logits.shape[-1]),
                            y[:, half:].reshape(-1))
        tot += l.item() * bs * half
        n_tok += bs * half
    return tot / n_tok


@torch.no_grad()
def needle_recall(model, lm, stream_len, trials=30):
    """流内寻针: 长流 70% 处埋 '密码:XXXX', 流尾问 '密码:' 贪婪解码 4 位, 测准确率."""
    marker = "密码:"
    mids = torch.tensor([lm.stoi[c] for c in marker], device=DEV)
    acc = 0
    for _ in range(trials):
        key = torch.randint(0, lm.vocab, (4,), device=DEV)
        seg = torch.randint(0, lm.vocab, (stream_len,), device=DEV)
        pos = int(stream_len * 0.7)
        seg[pos:pos + 4] = key
        ctx_ids = torch.cat([seg, mids])
        model.eval()
        logits = model(ctx_ids.unsqueeze(0))
        pred = []
        cur = ctx_ids.unsqueeze(0)
        h = None
        for _ in range(4):
            logits = model(cur)
            nxt = logits[0, -1].argmax().view(1, 1)
            pred.append(nxt.item())
            cur = torch.cat([cur, nxt], 1)
        acc += int(torch.tensor(pred, device=DEV) == key).item() if \
            (torch.tensor(pred, device=DEV) == key).all().item() else 0
    return acc / trials


@torch.no_grad()
def gen_metrics(model, lm, prompt="第一", n=400):
    """贪婪生成 400 字: distinct-2 与最长重复游程."""
    ids = torch.tensor([[lm.stoi.get(c, 0) for c in prompt]], device=DEV)
    cur = ids
    for _ in range(n):
        logits = model(cur)
        nxt = logits[0, -1].argmax().view(1, 1)
        cur = torch.cat([cur, nxt], 1)
        if cur.shape[1] > 512:
            cur = cur[:, -512:]
    g = cur[0, ids.shape[1]:].tolist()
    grams2 = set(zip(g[:-1], g[1:]))
    d2 = len(grams2) / max(len(g) - 1, 1)
    # 最长重复游程 (连续重复同一 token 的最大长度)
    run = best = 1
    for a, b2 in zip(g[:-1], g[1:]):
        run = run + 1 if a == b2 else 1
        best = max(best, run)
    return d2, best


def main():
    torch.manual_seed(0)
    lm = CharLM(ctx=128, device=DEV)
    print("语料 %d 字符 vocab %d" % (len(lm.data), lm.vocab), flush=True)

    models = {}
    for kind in KINDS:
        t0 = time.time()
        models[kind] = train_lm(kind, lm)
        print("%s 就绪 (%.0fs)" % (kind, time.time() - t0), flush=True)

    print("\n=== ① 长上下文外推 (CE, 训练 ctx=128) ===", flush=True)
    for ctx in [128, 256, 512, 1024]:
        row = {}
        for kind in KINDS:
            v = ce_at_ctx(models[kind], lm, ctx)
            row[kind] = v
        OUT["ce_ctx%d" % ctx] = row
        print("  ctx=%4d: %s" % (ctx, "  ".join("%s=%.3f" % (k, v) for k, v in row.items())),
              flush=True)
        json.dump(OUT, open(RJ, "w"))

    print("\n=== ② 流内寻针 (4 位密钥回忆 ACC) ===", flush=True)
    for sl in [256, 1024, 2048]:
        row = {}
        for kind in KINDS:
            v = needle_recall(models[kind], lm, sl)
            row[kind] = v
        OUT["needle_%d" % sl] = row
        print("  流长=%4d: %s" % (sl, "  ".join("%s=%.0f%%" % (k, v * 100)
                                               for k, v in row.items())), flush=True)
        json.dump(OUT, open(RJ, "w"))

    print("\n=== ③ 生成质量 (贪婪 400 字) ===", flush=True)
    for kind in KINDS:
        d2, run = gen_metrics(models[kind], lm)
        OUT["gen_%s" % kind] = {"distinct2": d2, "max_run": run}
        print("  %s: distinct2=%.3f 最长重复游程=%d" % (kind, d2, run), flush=True)
        json.dump(OUT, open(RJ, "w"))

    print("\n完成")


if __name__ == "__main__":
    main()
