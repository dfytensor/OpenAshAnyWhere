"""FRSMASH-CrossFeed: SlowMemory 从 x_ash(骨干输出)取输入, 不从 x_emb(浅层).
用户洞察: x_mem=x_mem(x_ash) — 让 SlowMemory 处理深度表示, 不再从浅层出发被碾压.
加 mem_norm(RMSNorm) 修复幅度. 看 gate 是否打开、ppl 是否变.
"""
import torch, torch.nn as nn, os, sys
sys.path.insert(0, r'F:\rwkv\frsmash_v36')
from frsmash_v36 import FRSMASHv36


class FRSMASHCrossFeed(FRSMASHv36):
    def __init__(self, voc_size, hidden_size, num_heads, num_layers, n_slots=4):
        super().__init__(voc_size, hidden_size, num_heads, num_layers, n_slots)
        self.mem_norm = nn.RMSNorm(hidden_size)

    def forward(self, x, states=None, h_slow=None, recall_state=None, return_state=False, pos_offset=0):
        B,T=x.shape; D=self.D; dt=self.head.weight.dtype
        x_emb=self.em(x).to(dt)+self.pe[pos_offset:pos_offset+T].to(dt)
        if states is None: states=[None]*self.num_ssm
        if h_slow is None: h_slow=torch.zeros(B,D,device=x.device,dtype=dt)
        h=x_emb; new_states=[] if return_state else None
        for i,layer in enumerate(self.layers):
            s_in=states[i] if return_state else None
            h,s=layer(h,s_in)
            if return_state: new_states.append(s)
        x_ash=self.final_norm(h)
        # ★ 核心改动: SlowMemory 从 x_ash(深度) 取输入, 不从 x_emb(浅层)
        inp_seq=self.mem_input_proj(x_ash)
        H_slow,h_slow=self.slow_cell(inp_seq,h_slow)
        x_mem=self.mem_norm(self.mem_proj(H_slow))
        if return_state or recall_state is not None:
            recall_out,recall_state=self.recall(x_emb,initial_state=recall_state,return_state=True)
        else:
            recall_out=self.recall(x_emb)
        x_recall=self.recall_norm(recall_out)
        cat=torch.cat([x_ash,x_mem],-1); gate=self.fusion_gate(cat)
        fused=self.fusion_norm(gate*x_ash+(1-gate)*x_mem+x_emb)+x_recall
        logits=self.head(fused)
        if return_state: return logits,new_states,h_slow,recall_state
        return logits


if __name__=='__main__':
    DEV='cuda'; VOCAB=23005
    m=FRSMASHCrossFeed(VOCAB,512,8,8,4).to(DEV)
    n=sum(p.numel() for p in m.parameters())
    x=torch.randint(0,VOCAB,(2,512),device=DEV)
    with torch.no_grad(): o=m(x)
    print(f'CrossFeed params={n:,} ({n/1e6:.1f}M) ok')
