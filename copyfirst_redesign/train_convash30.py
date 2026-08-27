"""ConvASH30 全量训练: MiniMind 1.27M 全量预训练 1 epoch (B=32) + SFT 905k 1 epoch + 生成样例."""
import sys, time, random
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import torch
import torch.nn.functional as F
from convash30 import ConvASH30, VOCAB
from open_ash_voc import OpenASHVoc

DEV = "cuda"
PT_CACHE = r"F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt"
SFT_CACHE = r"F:\rowcol_llm\sft_cached_256.pt"
OUT = r"F:\OpenASH2605\copyfirst_redesign"
B, S = 32, 256


def main():
    torch.manual_seed(0)
    random.seed(0)
    m = ConvASH30().to(DEV)
    n = sum(p.numel() for p in m.parameters())
    print("ConvASH30 参数量: %.2fM" % (n / 1e6), flush=True)

    # ============ 预训练 (全量 1 epoch) ============
    print("加载预训练缓存...", flush=True)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    N = len(seqs)
    STEPS = (N + B - 1) // B
    print("样本数: %d, batch=%d, steps=%d" % (N, B, STEPS), flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    t0 = time.time()
    for st in range(STEPS):
        m.train()
        xs = []
        for _ in range(B):
            s = seqs[random.randrange(N)][:S]
            xs.append(F.pad(s, (0, S - s.numel())))
        x = torch.stack(xs).to(DEV)
        y = x.clone()
        y[:, :-1] = x[:, 1:]
        y[:, -1] = 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = m(x)
            loss = F.cross_entropy(out[:, :-1].reshape(-1, VOCAB),
                                   y[:, :-1].reshape(-1), ignore_index=0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if st % 250 == 0:
            el = time.time() - t0
            print("PT %d/%d loss=%.3f (%.0fs, %.0fms/step)" %
                  (st, STEPS, loss.item(), el, el / (st + 1) * 1000), flush=True)
    torch.save(m.state_dict(), OUT + r"\convash30_pt_full.pth")
    print("预训练完成 %.0fs" % (time.time() - t0), flush=True)

    # ============ SFT (全量 1 epoch) ============
    print("加载 SFT 缓存...", flush=True)
    cache = torch.load(SFT_CACHE, map_location="cpu", weights_only=True)
    G, T = cache["grids"], cache["tgts"]
    n_sft = G.shape[0]
    STEPS_S = (n_sft + B - 1) // B
    print("SFT 样本数: %d, steps=%d" % (n_sft, STEPS_S), flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=0.01)
    t0 = time.time()
    for st in range(STEPS_S):
        m.train()
        idx = torch.randint(0, n_sft, (B,))
        g = (G[idx] - 34).clamp(min=0).to(DEV).reshape(B, 256)
        t = (T[idx] - 34).clamp(min=0).to(DEV).reshape(B, 256)
        mask = (T[idx] != 0).reshape(B, 256).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = m(g)
            loss = (F.cross_entropy(out[:, :-1].reshape(-1, VOCAB), t[:, 1:].reshape(-1),
                                    reduction="none").view(B, 255) * mask[:, 1:]).sum() / mask[:, 1:].sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if st % 250 == 0:
            el = time.time() - t0
            print("SFT %d/%d loss=%.3f (%.0fs, %.0fms/step)" %
                  (st, STEPS_S, loss.item(), el, el / (st + 1) * 1000), flush=True)
    torch.save(m.state_dict(), OUT + r"\convash30_sft_full.pth")
    print("SFT 完成 %.0fs" % (time.time() - t0), flush=True)

    # ============ 生成样例 ============
    voc = OpenASHVoc(agent_voc_path=r"F:\OpenASH2605\open_ash_voc_agent.json")
    for q in ["你好", "人工智能是什么", "给我讲一个关于环保的小故事"]:
        ids = [1, 5, 67] + voc.encode(q) + [2, 67, 1, 6, 67]
        torch.manual_seed(0)
        gen = m.generate(ids, steps=50, temperature=0.8, top_k=30)
        text = voc.decode(gen[0].tolist())
        print("\n[问] %s\n[答] %s" % (q, text[text.find(q) + len(q):][:150]), flush=True)


if __name__ == "__main__":
    main()
