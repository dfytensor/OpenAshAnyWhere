"""FRSMASH-StateMoE: SlowMemory 改成 N=10 状态专家 + top-1 路由, 总状态容量 10×.
每专家是独立 LinearSlowMemory(d 维递归状态); 每 token 路由到 1 个专家, 只更新/读取它.
每专家用 masked 输入跑全序列递归 => 状态只被路由到它的 token 更新. 稀疏 => ~1× 激活算力, 10× 状态.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, os, sys
sys.path.insert(0, r'F:\rwkv\frsmash_v36')
from frsmash_v36 import FRSMASHv36, LinearSlowMemory


class StateMoESlow(nn.Module):
    def __init__(self, d, n_experts=10, rank=None):
        super().__init__(); self.N=n_experts; self.d=d
        self.experts=nn.ModuleList([LinearSlowMemory(d, rank) for _ in range(n_experts)])
        self.gate=nn.Linear(d, n_experts, bias=False)
        self.last_aux=0.0
    def forward(self, x_seq, h_states):   # x_seq:(B,T,d), h_states:(N,B,d)
        B,T,d=x_seq.shape; N=self.N
        logits=self.gate(x_seq)                                  # (B,T,N)
        idx=logits.argmax(-1)                                    # top-1 (B,T)
        probs=F.softmax(logits,-1)                               # (B,T,N) for aux
        out=torch.zeros_like(x_seq)
        new_states=[]
        per_exp_out=[]   # 每专家输出(被 mask 后)的 L2 范数, 诊断用
        for e in range(N):
            m=(idx==e).unsqueeze(-1).to(x_seq.dtype)             # (B,T,1)
            inp_e=x_seq*m                                        # 该专家只见其路由 token
            y_e, h_e = self.experts[e](inp_e, h_states[e])
            new_states.append(h_e)
            out=out + y_e*m                                      # 只路由 token 读此专家
            per_exp_out.append(float((y_e*m).norm().detach()))
        h_states=torch.stack(new_states, dim=0)                  # (N,B,D) 新张量, 无 in-place
        # 负载均衡 aux (GShard 式)
        self.last_aux = N * (probs.mean(dim=[0,1])**2).sum()
        # 诊断
        with torch.no_grad():
            load=(idx.unsqueeze(-1)==torch.arange(N,device=x_seq.device)).float().mean(dim=[0,1])  # 每专家 token 比例
            st_norms=h_states.flatten(1).norm(dim=1)             # 每专家状态范数 (N,)
            self.diag={'load':load.cpu(), 'state_norm':st_norms.cpu(),
                       'out_norm':torch.tensor(per_exp_out),
                       'route_ent': float(-(probs.clamp(1e-9).log()*probs).sum(-1).mean())}  # 路由熵
        return out, h_states


class FRSMASHStateMoE(FRSMASHv36):
    def __init__(self, voc_size, hidden_size, num_heads, num_layers, n_slots=4, n_state_experts=10):
        super().__init__(voc_size, hidden_size, num_heads, num_layers, n_slots)
        self.slow_cell=StateMoESlow(hidden_size, n_state_experts)
        self.N=n_state_experts
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
        H_slow,h_slow=self.slow_cell(inp_seq, h_slow)           # (B,T,D), (N,B,D)
        x_mem=self.mem_proj(H_slow)
        if return_state or recall_state is not None:
            recall_out,recall_state=self.recall(x_emb,initial_state=recall_state,return_state=True)
        else:
            recall_out=self.recall(x_emb)
        x_recall=self.recall_norm(recall_out)
        cat=torch.cat([x_ash,x_mem],-1); gate=self.fusion_gate(cat)
        fused=self.fusion_norm(gate*x_ash+(1-gate)*x_mem+x_emb)+x_recall
        logits=self.head(fused)
        aux=self.slow_cell.last_aux
        if return_state: return logits,new_states,h_slow,recall_state
        return logits, aux


if __name__=='__main__':
    DEV='cuda'; VOCAB=23005
    for tag,cls,kw in [('vanilla',FRSMASHv36,{}),('statemoe',FRSMASHStateMoE,{'n_state_experts':10})]:
        m=cls(VOCAB,512,8,8,n_slots=4,**kw).to(DEV)
        n=sum(p.numel() for p in m.parameters())
        x=torch.randint(0,VOCAB,(2,512),device=DEV)
        with torch.no_grad():
            o=m(x)
            o2=o[0] if isinstance(o,tuple) else o
        print(f'{tag}: params={n:,} ({n/1e6:.1f}M)  out={o2.shape}  ok')
