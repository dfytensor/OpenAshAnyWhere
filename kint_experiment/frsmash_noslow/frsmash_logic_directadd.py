"""FRSMASH-LogicDirectAdd: DirectAdd + GRU 逻辑状态(替代线性 SlowMemory).
去掉 gate(防坍缩) + GRU 替代线性递归(把逻辑塞进 state).
fused = norm(x_ash + x_mem + x_emb) + x_recall
"""
import torch, torch.nn as nn, os, sys
sys.path.insert(0, r'F:\rwkv\frsmash_v36')
from frsmash_v36 import FRSMASHv36


class LogicState(nn.Module):
    """GRU(K步) deep state transition: 条件门控=逻辑, 不是线性累加."""
    def __init__(self, d, K=2):
        super().__init__(); self.K = K
        self.gru = nn.GRU(d, d, num_layers=K, batch_first=True)
        self.proj = nn.Linear(d, d, bias=False)
    def forward(self, x_seq, h0):
        h0e = h0.unsqueeze(0).expand(self.K, -1, -1).contiguous() if h0.dim()==2 else h0
        out, h_final = self.gru(x_seq, h0e)
        return self.proj(out), h_final[-1]


class FRSMASHLogicDirectAdd(FRSMASHv36):
    """SSM骨干 + GRU逻辑状态(DirectAdd无gate) + GLA recall."""
    def __init__(self, voc_size, hidden_size, num_heads, num_layers, n_slots=4, logic_K=2):
        super().__init__(voc_size, hidden_size, num_heads, num_layers, n_slots)
        self.slow_cell = LogicState(hidden_size, K=logic_K)
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
        fused=self.fusion_norm(x_ash+x_mem+x_emb)+x_recall
        logits=self.head(fused)
        if return_state: return logits,new_states,h_slow,recall_state
        return logits
