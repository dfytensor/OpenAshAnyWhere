#!/usr/bin/env python3
"""微探针: 不训练, 检查 bank 读出路径是否携带配对 value 信号.
chunk-1 查询 (J=1, w≡1, 最干净路径):
  B: cos(q_t·S₀, 正确value向量) vs 错误value
  A: cos(q_t·S_t, 正确value向量) vs 错误value   (源经 ~256+ 步衰减)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics as st

torch.manual_seed(0)
DEV = "cuda"
B, L, CH, G, DH, NS = 8, 384, 128, 4, 16, 2
N_PAIR = 8
FILL0, KEY0, VAL0 = 1, 65, 73
V_FILL, V_KEY, V_VAL = 64, 8, 8
VOCAB = VAL0 + V_VAL
D = 64

class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D)
        self.ln = nn.LayerNorm(D)
        self.wq = nn.Linear(D, G * NS * DH, bias=False)
        self.wk = nn.Linear(D, G * DH, bias=False)
        self.wv = nn.Linear(D, G * DH, bias=False)
        self.w_rate = nn.Linear(D, G * DH, bias=True)
        self.w_eta = nn.Linear(D, G * DH, bias=True)
        nn.init.normal_(self.emb.weight, std=0.6)
        nn.init.constant_(self.w_rate.bias, -4.0)
        nn.init.constant_(self.w_eta.bias, -2.0)

def make_batch(bs, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(FILL0, FILL0 + V_FILL, (bs, L), generator=g)
    cm = torch.zeros(bs, L)
    queries = []                       # (b, t, key, val)
    for b in range(bs):
        pairs = torch.stack([torch.arange(KEY0, KEY0 + V_KEY),
                             VAL0 + torch.randperm(V_VAL, generator=g)], dim=1)
        pos = torch.randperm(CH - 16, generator=g)[:16]
        for i in range(N_PAIR):
            p, q = sorted([int(pos[2*i].item()), int(pos[2*i+1].item())])
            ids[b, p], ids[b, q] = int(pairs[i][0]), int(pairs[i][1])
            cm[b, p] = cm[b, q] = 1.0
        qpos = torch.randperm(CH - 2, generator=g)[:8] + 1
        for i in range(8):
            t = CH + qpos[i]                       # chunk-1 查询
            ids[b, t] = int(pairs[i][0])
            queries.append((b, int(t), int(pairs[i][0]), int(pairs[i][1])))
    return ids.to(DEV), cm.to(DEV), queries

def scan(k, v, rate, eta):
    S = torch.zeros(k.shape[0], G, DH, DH, device=k.device)
    ys = []
    for t in range(k.shape[1]):
        kt, vt = k[:, t], v[:, t]
        rr = (kt.unsqueeze(-1) * S).sum(-2)
        S = torch.exp(-rate[:, t]).unsqueeze(-1) * (S - eta[:, t].unsqueeze(-1) * kt.unsqueeze(-1) * rr.unsqueeze(-2)) \
            + kt.unsqueeze(-1) * vt.unsqueeze(-2)
        ys.append(S)
    return torch.stack(ys, dim=3), S

tr = Trunk().to(DEV)
with torch.no_grad():                          # 对齐: q头0 = k
    wq = tr.wq.weight.view(G, NS, DH, D)
    wq[:, 0] = tr.wk.weight.view(G, DH, D)
    wq[:, 1] = 0.0

ids, cm, queries = make_batch(8, seed=0)
with torch.no_grad():
    h = tr.ln(tr.emb(ids))
    vw = lambda x, n: x.view(x.shape[0], x.shape[1], *n)
    q = vw(tr.wq(h), (G, NS, DH)).float()
    k = (vw(tr.wk(h), (G, DH)) * cm.unsqueeze(-1).unsqueeze(-1)).float()
    v = (vw(tr.wv(h), (G, DH)) * cm.unsqueeze(-1).unsqueeze(-1)).float()
    rate = F.softplus(vw(tr.w_rate(h), (G, DH))).float()
    eta = torch.sigmoid(vw(tr.w_eta(h), (G, DH))).float()
    ysA, _ = scan(k, v, rate, eta)
    _, S0 = scan(k[:, :CH], v[:, :CH], rate[:, :CH], eta[:, :CH])
    Wv = tr.wv.weight.view(G, DH, D)           # (G,DH,D)

    cosB_c, cosB_w, cosA_c, cosA_w = [], [], [], []
    for (b, t, key, val) in queries:
        correct = torch.cat([Wv[g] @ h[b, val] for g in range(G)]).flatten()      # (G*DH,)
        wrongs = [torch.cat([Wv[g] @ h[b, v2] for g in range(G)]).flatten()
                  for v2 in range(VAL0, VAL0 + V_VAL) if v2 != val]
        yB = torch.cat([q[b, t, g, 0] @ S0[b, g] for g in range(G)]).flatten()    # bank 读出
        yA = torch.cat([q[b, t, g, 0] @ ysA[b, g, :, t, :] for g in range(G)]).flatten()  # 全state读出
        cosB_c.append(F.cosine_similarity(yB, correct, dim=0).item())
        cosA_c.append(F.cosine_similarity(yA, correct, dim=0).item())
        cosB_w.append(max(F.cosine_similarity(yB, w, dim=0).item() for w in wrongs))
        cosA_w.append(max(F.cosine_similarity(yA, w, dim=0).item() for w in wrongs))

n = len(queries)
print(f"探针样本: {n} 个 chunk-1 查询")
print(f"B bank (q·S₀):   cos正确 {st.mean(cosB_c):+.3f} | cos错误 {st.mean(cosB_w):+.3f} | 信号差 {st.mean(cosB_c)-st.mean(cosB_w):+.3f}")
print(f"A 全state (q·S_t): cos正确 {st.mean(cosA_c):+.3f} | cos错误 {st.mean(cosA_w):+.3f} | 信号差 {st.mean(cosA_c)-st.mean(cosA_w):+.3f}")
print("信号差 >0.1 = 路径健康; ≈0 = 路径断")
