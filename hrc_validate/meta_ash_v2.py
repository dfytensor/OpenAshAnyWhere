#!/usr/bin/env python3
"""
Meta-ASH: ConvASH30 + MetaGRU 融合
保留 ConvMaxStateSuper 多分支+cummax+gen_model
新增: 内稳态R + 繁衍项 + 有界clamp + GRU门控
"""
import sys, os, json, time, math
import numpy as np
sys.path.insert(0, r"F:\OpenASH2605")
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda"
ETA = 0.02
RHO = 0.5


class MetaMaxState(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.br = nn.ModuleList([nn.Linear(d, d) for _ in range(4)])
        self.gen_model = nn.Linear(d, d, bias=False)
        self.meta_r = nn.Linear(d, d)
        self.meta_z = nn.Linear(d, d)
        self.b_meta = nn.Parameter(torch.zeros(d))
        self.register_buffer("R", torch.full((1, 1, d), 2.5))
        self.eta = ETA
        self.rho = RHO

    def forward(self, x, state=None):
        b, s, d = x.shape
        o0 = self.br[0](x)
        o1 = self.br[1](x)
        o2 = self.br[2](x)
        o3 = self.br[3](x)
        o2_cm, _ = torch.cummax(o2, dim=1)
        mr = torch.sigmoid(self.meta_r(x) + self.b_meta)
        mz = torch.sigmoid(self.meta_z(x) + self.b_meta)
        # R 用 .data 避免版本冲突
        repro = self.R.data * torch.sigmoid(o1) * (1 - torch.sigmoid(o1))
        o1 = o1 + repro * 0.1
        combined = o0 + o1 + o2 + o3 + o2_cm
        gen_out = self.gen_model(combined.view(-1, d)).view(b, s, d)
        out = mz * gen_out + (1 - mz) * mr * x
        # 内稳态
        with torch.no_grad():
            self.R.data += self.eta * self.R.data * (self.rho - o1.mean().item())
            self.R.data.clamp_(0.1, 4.0)
        return out, None


class MetaASHLayer(nn.Module):
    def __init__(self, d, ffn_mult=4):
        super().__init__()
        self.attn = MetaMaxState(d)
        self.ln1 = nn.LayerNorm(d)
        ffn_d = d * ffn_mult
        self.ffn_gate = nn.Linear(d, ffn_d)
        self.ffn_up = nn.Linear(d, ffn_d)
        self.ffn_down = nn.Linear(ffn_d, d)
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x, state=None):
        attn_out, _ = self.attn(x, state)
        x = self.ln1(x + attn_out)
        ffn = self.ffn_down(F.silu(self.ffn_gate(x)) * self.ffn_up(x))
        return self.ln2(x + ffn), None


class MetaASHLM(nn.Module):
    def __init__(self, vocab, d=640, n_layers=16, ffn_mult=4):
        super().__init__()
        self.em = nn.Embedding(vocab, d, padding_idx=0)
        self.layers = nn.ModuleList([MetaASHLayer(d, ffn_mult) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids, targets=None):
        x = self.em(ids)
        for layer in self.layers:
            x, _ = layer(x, None)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]),
                                   targets.view(-1), ignore_index=0)
        return logits, loss


def main():
    t00 = time.time()
    from open_ash_voc import OpenASHVoc
    voc = OpenASHVoc(agent_voc_path=r"F:\OpenASH2605\open_ash_voc_agent.json")
    V = 23005

    model = MetaASHLM(vocab=V, d=640, n_layers=16, ffn_mult=4).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    print("Meta-ASH 参数: %.1fM" % (n_params / 1e6), flush=True)

    texts = []
    for p in [r"F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl",
              r"F:\OpenASH2605\minimind_data\sft_t2t_mini.jsonl"]:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    t = d.get("text", "")
                    if not t and "conversations" in d:
                        t = "".join(x.get("content", "") for x in d["conversations"])
                    if len(t) > 500:
                        texts.append(t.replace("\n", ""))
                except Exception:
                    pass
                if len(texts) >= 3000:
                    break
    print("texts:", len(texts), flush=True)

    seq_len = 256
    docs = []
    for txt in texts[:800]:
        ids = voc.encode(txt)[:seq_len]
        ids = [max(0, min(x, V - 1)) for x in ids]
        if len(ids) < seq_len:
            ids = ids + [0] * (seq_len - len(ids))
        docs.append(torch.tensor([ids], device=DEV))
    print("docs:", len(docs), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000, eta_min=3e-5)
    BS = 8
    model.train()
    t0 = time.time()
    for st in range(3000):
        idx = np.random.randint(0, len(docs), BS)
        x = torch.cat([docs[i] for i in idx])
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               x[:, 1:].reshape(-1), ignore_index=0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if st % 200 == 0 or st == 2999:
            print("  step %d/3000 loss=%.4f (%.0fs)" %
                  (st, loss.item(), time.time() - t0), flush=True)

    # 生成
    model.eval()
    with torch.no_grad():
        test = voc.encode("人工智能是")
        cur = torch.tensor([test], device=DEV)
        for _ in range(50):
            logits, _ = model(cur)
            nxt = logits[0, -1].argmax().view(1, 1)
            cur = torch.cat([cur, nxt], dim=1)
        gen_text = voc.decode(cur[0].tolist())
        print("\n生成: ", gen_text[:200], flush=True)

    torch.save(model.state_dict(), r"F:\夸克\hrc_validate\meta_ash_final.pt")
    print("\n完成 %.0fs" % (time.time() - t00), flush=True)


if __name__ == "__main__":
    main()
