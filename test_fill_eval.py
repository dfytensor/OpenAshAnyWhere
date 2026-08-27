"""
快速补全 Transformer/LSTM/GRU 的 CopyFirst 距离准确率
训练 + 评估关键距离点
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random

device = torch.device("cuda")
VOCAB = 32; END = 0; IGNORE = 1; H = 128

class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(VOCAB,H)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(H,4,H*4,0.0,batch_first=True) for _ in range(2)])
        self.head=nn.Linear(H,VOCAB)
    def forward(self,x,hp=None):
        T=x.size(1); m=nn.Transformer.generate_square_subsequent_mask(T,device=device)
        return self.head(self.layers[1](self.layers[0](self.embed(x)*math.sqrt(H),src_mask=m),src_mask=m)),None

class MiniLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(VOCAB,H); self.lstm=nn.LSTM(H,H,2,batch_first=True); self.head=nn.Linear(H,VOCAB)
    def forward(self,x,hp=None): return self.head(self.lstm(self.embed(x))[0]),None

class MiniGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(VOCAB,H); self.gru=nn.GRU(H,H,2,batch_first=True); self.head=nn.Linear(H,VOCAB)
    def forward(self,x,hp=None): return self.head(self.gru(self.embed(x))[0]),None

def make_batch(bs,nl):
    t=torch.randint(2,VOCAB,(bs,))
    n=torch.randint(2,VOCAB,(bs,nl))
    e=torch.full((bs,1),END,dtype=torch.long)
    x=torch.cat([t.unsqueeze(1),n,e],1)
    y=torch.full_like(x,IGNORE); y[:,-1]=t
    return x,y

for name, model_class in [("Transformer", MiniTransformer), ("LSTM", MiniLSTM), ("GRU", MiniGRU)]:
    torch.manual_seed(42)
    m = model_class().to(device)
    n_p = sum(p.numel() for p in m.parameters())
    
    # Train
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 2500)
    best = float('inf')
    for st in range(1, 2501):
        nl = random.randint(4, 64)
        x, y = make_batch(64, nl); x, y = x.to(device), y.to(device)
        logits, _ = m(x)
        loss = F.cross_entropy(logits[:, -1, :], y[:, -1], ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
        if loss.item() < best: best = loss.item()
    print(f"{name:>12} ({n_p:>7,}p): best_loss={best:.5f}", flush=True)
    
    # Eval (Transformer needs smaller batch for long sequences due to O(n²))
    m.eval()
    dists = [4, 64, 512, 2048, 8192, 16384, 32768]
    accs = {}
    for d in dists:
        if name == "Transformer" and d >= 2048:
            eb = 4  # tiny batch for long sequences
        else:
            eb = 64 if d <= 8192 else 32
        c = 0
        for _ in range(4):
            x, y = make_batch(eb, d); x, y = x.to(device), y.to(device)
            logits, _ = m(x)
            c += (logits[:, -1, :].argmax(-1) == y[:, -1]).sum().item()
        accs[d] = c / (4 * eb) * 100
    print(f"           Acc: " + " | ".join([f"{d}:{accs[d]:5.1f}%" for d in dists]))

# 汇总表
print(f"\n  Model          |   4-64 |   2048 |   8192 |    32K |  131K")
print(f"  " + "-" * 60)
print(f"  FRSM 255K      |   100% |   100% |    99% |    95% |   91%")
print(f"  FRSM 14.7M     |   100% |    98% |    38% |    19% |   —")
print(f"  Transformer    |   ~100%|       ? |      ? |      ? | O(n²)")
print(f"  LSTM           |      ? |       ? |      ? |      ? |    ?")
print(f"  GRU            |      ? |       ? |      ? |      ? |    ?")
print(f"  OpenASH        |   ~3.3%|   ~3.3%|   ~3.3%|   ~3.3%| ~3.3%  (未收敛)")
print(f"  WDLM-Neural    |   ~3.3%|   ~3.3%|   ~3.3%|   ~3.3%| ~3.3%  (未收敛)")
print(f"  WDLM-Real      |   ~3.3%|   ~3.3%|   ~3.3%|   ~3.3%| ~3.3%  (未收敛)")
