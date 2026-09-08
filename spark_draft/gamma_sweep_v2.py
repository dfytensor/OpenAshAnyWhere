#!/usr/bin/env python3
"""γ 扫描: 草稿模型蒸馏 + 推测解码 (不依赖 PEFT, 手动 adapter)."""
import sys, os, json, time, math
import numpy as np
sys.path.insert(0, r"F:\Spark-X2.5-1.7B")
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
MDIR = r"F:\Spark-X2.5-1.7B"
N_DRAFT = 4
N_GEN = 128


class Adapter(nn.Module):
    """LoRA 式瓶颈适配器 (零初始化输出 = 初始恒等)."""
    def __init__(self, d, r=16):
        super().__init__()
        self.down = nn.Linear(d, r, bias=False, dtype=torch.bfloat16)
        self.up = nn.Linear(r, d, bias=False, dtype=torch.bfloat16)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(self.down(x))

    def forward(self, x):
        return x + self.up(self.down(x))


class DraftModel(nn.Module):
    """草稿 = em(共享) + 前4层(共享+Adapter) + head(共享)."""
    def __init__(self, target, n_layers=N_DRAFT):
        super().__init__()
        self.embedding = target.model.embedding
        self.layers = nn.ModuleList([target.model.layers[i] for i in range(n_layers)])
        self.norm = target.model.norm
        self.head = target.lm_head
        self.head_dim = target.config.head_dim
        # 可训练 adapter
        self.adapters = nn.ModuleList([Adapter(2048, 16) for _ in range(n_layers)])
        self.trainable = [p for p in self.parameters() if p.requires_grad]

    def _rope(self, L):
        inv = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2,
                      device=next(self.parameters()).device, dtype=torch.float32) / self.head_dim))
        pos = torch.arange(L, device=inv.device, dtype=torch.float32)
        freqs = torch.outer(pos, inv)
        emb = torch.cat([freqs, freqs], -1)
        return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)

    def forward(self, ids):
        x = self.embedding(ids)
        cos, sin = self._rope(ids.shape[1])
        pe = (cos, sin)
        for i, layer in enumerate(self.layers):
            out = layer(x, position_embeddings=pe)
            x = out[0] if isinstance(out, tuple) else x
            x = x + self.adapters[i](x)
        x = self.norm(x)
        return self.head(x)


def main():
    t00 = time.time()
    tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        MDIR, dtype=torch.bfloat16, device_map=DEV, trust_remote_code=True)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    # 数据
    texts = []
    for p in [r"F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl"]:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    t = json.loads(line).get("text", "").replace("\n", "")
                    if len(t) > 500:
                        texts.append(t)
                except Exception:
                    pass
                if len(texts) >= 400:
                    break

    prompt_ids = tok.encode(texts[0], add_special_tokens=False)[:256]

    # ① 蒸馏
    draft = DraftModel(target).to(DEV)
    n_train = sum(p.numel() for p in draft.trainable)
    print("可训练: %.2fM" % (n_train / 1e6), flush=True)
    opt = torch.optim.AdamW(draft.trainable, lr=5e-5, weight_decay=0.01)

    print("蒸馏 500 步...", flush=True)
    draft.train()
    t0 = time.time()
    for st in range(500):
        batch = []
        for _ in range(2):
            txt = texts[np.random.randint(len(texts))]
            ids = tok.encode(txt, add_special_tokens=False)[:64]
            if len(ids) < 64:
                ids = ids + [0] * (64 - len(ids))
            batch.append([tok.bos_token_id or 0] + ids)
        t_ids = torch.tensor(batch, device=DEV)
        with torch.no_grad():
            t_logits = target(t_ids).logits.float()
        d_logits = draft(t_ids).float()
        kl = F.kl_div(F.log_softmax(d_logits, -1),
                      F.log_softmax(t_logits, -1), log_target=True, reduction="batchmean")
        opt.zero_grad(set_to_none=True)
        kl.backward()
        torch.nn.utils.clip_grad_norm_(draft.trainable, 1.0)
        opt.step()
        if st % 100 == 0 or st == 499:
            print("  step %d KL=%.4f (%.0fs)" % (st, kl.item(), time.time() - t0), flush=True)
    draft.eval()

    # ② 初始/蒸馏后接受率
    @torch.no_grad()
    def acceptance():
        total_match, total = 0, 0
        for i in range(min(10, len(texts))):
            ids = tok.encode(texts[i], add_special_tokens=False)[:256]
            if len(ids) < 32:
                continue
            ids = [tok.bos_token_id or 0] + ids
            t_ids = torch.tensor([ids], device=DEV)
            t_lg = F.log_softmax(target(t_ids).logits[0].float(), -1)
            d_lg = F.log_softmax(draft(t_ids)[0].float(), -1)
            match = (t_lg.argmax(-1) == d_lg.argmax(-1)).sum().item()
            total_match += match
            total += t_ids.shape[1] - 1
        return total_match / max(total, 1)

    print("接受率: %.3f" % acceptance(), flush=True)

    # ③ γ 扫描
    print("\n===== γ 扫描 (128 tok 生成) =====", flush=True)

    @torch.no_grad()
    def spec_decode(gamma):
        ids = list(prompt_ids)
        n_prompt = len(ids)
        total_accepted = 0
        rounds = 0
        t0 = time.time()
        while len(ids) - n_prompt < n_gen:
            rounds += 1
            n_orig = len(ids)
            remaining = n_gen - (n_orig - n_prompt)
            g = min(gamma, remaining)
            cur = torch.tensor([ids], device=DEV)
            draft_toks = []
            for _ in range(g):
                d_out = draft(cur)[0, -1]
                nxt = d_out.argmax(-1).view(1, 1)
                draft_toks.append(nxt.item())
                cur = torch.cat([cur, nxt], dim=1)
            verify = torch.tensor([ids + draft_toks], device=DEV)
            t_logits = target(verify).logits[0]
            n_accept = 0
            for j in range(g):
                if draft_toks[j] == t_logits[n_orig - 1 + j].argmax().item():
                    n_accept += 1
                else:
                    break
            for j in range(n_accept):
                ids.append(draft_toks[j])
            bonus = n_orig - 1 + n_accept
            if bonus < t_logits.shape[0]:
                ids.append(t_logits[bonus].argmax().item())
            total_accepted += n_accept
        dt = time.time() - t0
        acc = total_accepted / (rounds * gamma) if rounds else 0
        return dt, n_gen / dt, acc, rounds

    @torch.no_grad()
    def ar_decode():
        cur = torch.tensor([prompt_ids], device=DEV)
        t0 = time.time()
        for _ in range(n_gen):
            logits = target(cur).logits[0, -1]
            nxt = logits.argmax(-1).view(1, 1)
            cur = torch.cat([cur, nxt], dim=1)
        return time.time() - t0

    t_ar = ar_decode()
    tps_ar = n_gen / t_ar
    print("AR 基线: %.2fs %.1f tok/s\n" % (t_ar, tps_ar), flush=True)
    print("%-6s %8s %8s %8s %6s %8s" % ("γ", "时间", "tok/s", "接受率", "轮数", "加速比"), flush=True)
    print("-" * 50, flush=True)
    for gamma in [4, 6, 10, 16, 20]:
        dt, tps, acc, rounds = spec_decode(gamma)
        print("%-6d %8.2f %8.1f %7.1f%% %6d %8.2fx" %
              (gamma, dt, tps, acc * 100, rounds, tps / tps_ar), flush=True)
    print("\n基线: %.2fs (%.1f tok/s)" % (t_ar, tps_ar), flush=True)
    print("总用时 %.0fs" % (time.time() - t00), flush=True)


if __name__ == "__main__":
    main()
