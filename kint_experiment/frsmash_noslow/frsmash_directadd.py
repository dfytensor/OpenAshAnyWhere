"""FRSMASH-DirectAdd: 去掉 gate, 直接 x_ash + x_mem + x_emb + x_recall.
gate 永远=1 是因为模型学会了砍 SlowMemory; 去掉 gate 强制所有路贡献.
"""
import torch, torch.nn as nn, os, sys
sys.path.insert(0, r'F:\rwkv\frsmash_v36')
from frsmash_v36 import FRSMASHv36


class FRSMASHDirectAdd(FRSMASHv36):
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
        inp_seq=self.mem_input_proj(x_emb)
        H_slow,h_slow=self.slow_cell(inp_seq,h_slow)
        x_mem=self.mem_norm(self.mem_proj(H_slow))
        if return_state or recall_state is not None:
            recall_out,recall_state=self.recall(x_emb,initial_state=recall_state,return_state=True)
        else:
            recall_out=self.recall(x_emb)
        x_recall=self.recall_norm(recall_out)
        # ★ 去掉 gate, 直接 +
        fused=self.fusion_norm(x_ash+x_mem+x_emb)+x_recall
        logits=self.head(fused)
        if return_state: return logits,new_states,h_slow,recall_state
        return logits
