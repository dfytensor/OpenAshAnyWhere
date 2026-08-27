"""FRSMASH-LogicState: 用 GRU(K步逻辑推理/token) 替代死的线性 SlowMemory.
核心洞察: state 不该只存记忆(线性累加), 要存逻辑(条件推理).
GRU 的 reset/update 门 = IF-THEN 逻辑: "重要→更新, 不重要→保留". 这是 SSM 给不了的.
nn.GRU CUDA 优化, 无 Python 循环. K=2 层 = 每 token 2 步逻辑推理.
"""
import torch, torch.nn as nn, os, sys
sys.path.insert(0, r'F:\rwkv\frsmash_v36')
from frsmash_v36 import FRSMASHv36


class LogicState(nn.Module):
    """GRU-based deep state transition: h_t = GRU(h_{t-1}, x_t), K 层 = K 步逻辑推理.
    替代线性 SlowMemory(h_t = A*h + B*x, 无逻辑)."""
    def __init__(self, d, K=2):
        super().__init__()
        self.K = K
        self.gru = nn.GRU(d, d, num_layers=K, batch_first=True)
        self.proj = nn.Linear(d, d, bias=False)
    def forward(self, x_seq, h0):
        # h0: (B, d) → expand to (K, B, d)
        if h0.dim() == 2:
            h0e = h0.unsqueeze(0).expand(self.K, -1, -1).contiguous()
        else:
            h0e = h0
        out, h_final = self.gru(x_seq, h0e)
        return self.proj(out), h_final[-1]


class FRSMASHLogicState(FRSMASHv36):
    """SSM 骨干 + GLA recall + LogicState(GRU, 替代 SlowMemory)."""
    def __init__(self, voc_size, hidden_size, num_heads, num_layers, n_slots=4, logic_K=2):
        super().__init__(voc_size, hidden_size, num_heads, num_layers, n_slots)
        self.slow_cell = LogicState(hidden_size, K=logic_K)
        self.K_logic = logic_K

    def forward(self, x, states=None, h_slow=None, recall_state=None, return_state=False, pos_offset=0):
        B,T=x.shape; D=self.D; dt=self.head.weight.dtype
        x_emb=self.em(x).to(dt)+self.pe[pos_offset:pos_offset+T].to(dt)
        if states is None: states=[None]*self.num_ssm
        if h_slow is None: h_slow=torch.zeros(B, D, device=x.device, dtype=dt)
        h=x_emb; new_states=[] if return_state else None
        for i,layer in enumerate(self.layers):
            s_in=states[i] if return_state else None
            h,s=layer(h,s_in)
            if return_state: new_states.append(s)
        x_ash=self.final_norm(h)
        # LogicState (GRU) 替代 SlowMemory
        inp_seq=self.mem_input_proj(x_emb)
        H_slow, h_slow = self.slow_cell(inp_seq, h_slow)
        x_mem=self.mem_proj(H_slow)
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
    m=FRSMASHLogicState(VOCAB,512,8,8,4,logic_K=2).to(DEV)
    n=sum(p.numel() for p in m.parameters())
    x=torch.randint(0,VOCAB,(2,512),device=DEV)
    with torch.no_grad(): o=m(x)
    print(f'LogicState(K=2) params={n:,} ({n/1e6:.1f}M)  out={o.shape} ok')
