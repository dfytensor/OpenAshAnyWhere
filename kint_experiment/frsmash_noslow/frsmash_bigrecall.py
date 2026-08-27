"""FRSMASH-BigRecall: 放大 GLA recall(d_h 64→128 = 4× 内容寻址状态, 贡献32%的真状态)."""
import torch, torch.nn as nn, os, sys
sys.path.insert(0, r'F:\rwkv\frsmash_v36')
from frsmash_v36 import FRSMASHv36, GlaRecall


class FRSMASHBigRecall(FRSMASHv36):
    def __init__(self, voc_size, hidden_size, num_heads, num_layers, n_slots=4, recall_d_h=128):
        super().__init__(voc_size, hidden_size, num_heads, num_layers, n_slots)
        self.recall = GlaRecall(hidden_size, heads=num_heads, d_h=recall_d_h)
        self.recall_norm = nn.RMSNorm(hidden_size)


if __name__=='__main__':
    DEV='cuda'; VOCAB=23005
    for tag,dh in [('vanilla(d_h=64)',64),('bigrecall(d_h=128)',128),('bigrecall(d_h=96)',96)]:
        cls=FRSMASHv36 if dh==64 else FRSMASHBigRecall
        kw={'recall_d_h':dh} if dh!=64 else {}
        m=cls(VOCAB,512,8,8,4,**kw).to(DEV)
        n=sum(p.numel() for p in m.parameters())
        st=8*dh*dh
        x=torch.randint(0,VOCAB,(2,512),device=DEV)
        with torch.no_grad(): o=m(x)
        print(f'{tag}: params={n:,} ({n/1e6:.1f}M)  recall_state={st}  ok')
