"""FRSMASH-DenseStateMoE: 每专家看全序列(全强度状态, 非 masked) + 学习门控混合.
诊断(loop10)指出 masked 版因输入饥饿失效(状态 1/19 强度); 本版 dense, 每专家全强度.
10× 递归算力换真 10× 状态信号. 假设: 应能胜 vanilla.
"""
import torch, torch.nn as nn, torch.nn.functional as F, os, sys
sys.path.insert(0, r'F:\rwkv\frsmash_v36')
from frsmash_v36 import FRSMASHv36, LinearSlowMemory


class DenseStateMoESlow(nn.Module):
    def __init__(self, d, n_experts=10, rank=None):
        super().__init__(); self.N=n_experts
        self.experts=nn.ModuleList([LinearSlowMemory(d, rank) for _ in range(n_experts)])
        self.gate=nn.Linear(d, n_experts, bias=False)
        self.last_aux=0.0; self.diag={}
    def forward(self, x_seq, h_states):    # x_seq:(B,T,d), h_states:(N,B,d)
        B,T,d=x_seq.shape; N=self.N
        w=F.softmax(self.gate(x_seq), dim=-1)               # (B,T,N) 每 token 门控混合权重
        outs=[]; new_states=[]
        for e in range(N):
            y_e, h_e = self.experts[e](x_seq, h_states[e])  # 全序列, 全强度状态
            outs.append(y_e); new_states.append(h_e)
        out=sum(w[...,e:e+1]*outs[e] for e in range(N))     # 加权混合
        h_states=torch.stack(new_states, dim=0)
        self.last_aux=0.0
        with torch.no_grad():
            st=torch.stack(new_states,0).flatten(1).norm(dim=1)
            self.diag={'state_norm':st.cpu(), 'gate_ent':float(-(w.clamp(1e-9).log()*w).sum(-1).mean())}
        return out, h_states


class FRSMASHDenseStateMoE(FRSMASHv36):
    def __init__(self, voc_size, hidden_size, num_heads, num_layers, n_slots=4, n_state_experts=10):
        super().__init__(voc_size, hidden_size, num_heads, num_layers, n_slots)
        self.slow_cell=DenseStateMoESlow(hidden_size, n_state_experts); self.N=n_state_experts
    def forward(self, x, states=None, h_slow=None, recall_state=None, return_state=False, pos_offset=0):
        B,T=x.shape; D=self.D; dt=self.head.weight.dtype
        x_emb=self.em(x).to(dt)+self.pe[pos_offset:pos_offset+T].to(dt)
        if states is None: states=[None]*self.num_ssm
        if h_slow is None: h_slow=torch.zeros(self.N, B, D, device=x.device, dtype=dt)
        h=x_emb; new_states=[] if return_state else None
        for i,layer in enumerate(self.layers):
            s_in=states[i] if return_state else None
            h,s=layer(h,s_in)
            if return_state: new_states.append(s)
        x_ash=self.final_norm(h)
        inp_seq=self.mem_input_proj(x_emb)
        H_slow,h_slow=self.slow_cell(inp_seq, h_slow)
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
    m=FRSMASHDenseStateMoE(VOCAB,512,8,8,4,n_state_experts=10).to(DEV)
    n=sum(p.numel() for p in m.parameters()); print(f'dense_statemoe params={n:,} ({n/1e6:.1f}M)')
    x=torch.randint(0,VOCAB,(2,512),device=DEV)
    with torch.no_grad():
        with torch.amp.autocast('cuda',dtype=torch.bfloat16): o=m(x)
    print('out',o.shape,'state_norm',m.slow_cell.diag.get('state_norm'))
