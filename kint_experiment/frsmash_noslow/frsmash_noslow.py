"""FRSMASH-NoSlow: 砍掉 SlowMemory(贡献0%的死分支), 验证 ppl 不变 => 确认死代码.
forward 跳过 slow_cell/mem_input_proj/mem_proj/fusion_gate, 直接 x_ash + x_emb + x_recall.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, os, sys
sys.path.insert(0, r'F:\rwkv\frsmash_v36')
from frsmash_v36 import FRSMASHv36


class FRSMASHNoSlow(FRSMASHv36):
    """砍掉 SlowMemory: fused = norm(x_ash + x_emb) + x_recall (无 gate/mem)."""
    def forward(self, x, states=None, h_slow=None, recall_state=None, return_state=False, pos_offset=0):
        B,T=x.shape; D=self.D; dt=self.head.weight.dtype
        x_emb=self.em(x).to(dt)+self.pe[pos_offset:pos_offset+T].to(dt)
        if states is None: states=[None]*self.num_ssm
        h=x_emb; new_states=[] if return_state else None
        for i,layer in enumerate(self.layers):
            s_in=states[i] if return_state else None
            h,s=layer(h,s_in)
            if return_state: new_states.append(s)
        x_ash=self.final_norm(h)
        # 跳过 SlowMemory: 无 mem_input_proj / slow_cell / mem_proj / fusion_gate
        if return_state or recall_state is not None:
            recall_out,recall_state=self.recall(x_emb,initial_state=recall_state,return_state=True)
        else:
            recall_out=self.recall(x_emb)
        x_recall=self.recall_norm(recall_out)
        fused=self.fusion_norm(x_ash+x_emb)+x_recall       # 无 gate, 无 x_mem
        logits=self.head(fused)
        if return_state: return logits,new_states,None,recall_state
        return logits


if __name__=='__main__':
    DEV='cuda'; VOCAB=23005
    m=FRSMASHNoSlow(VOCAB,512,8,8,4).to(DEV)
    n=sum(p.numel() for p in m.parameters())
    x=torch.randint(0,VOCAB,(2,512),device=DEV)
    with torch.no_grad(): o=m(x)
    print(f'NoSlow params={n:,} ({n/1e6:.1f}M)  out={o.shape} ok')
