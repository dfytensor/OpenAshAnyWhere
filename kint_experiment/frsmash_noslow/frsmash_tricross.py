"""FRSMASH-TriCross: 三路交叉混合(SSM↔GLA↔SlowMem 互相喂深度表示).
原版: 三路并行从 x_emb 出发→各自独立→gate 融合(→坍缩).
本版: 三路串联交叉, 每路从前一路的深度输出取输入, 逐层递进增强.

flow:
  x_emb → SSM(8层) → x_ash
  x_ash → SlowMemory → x_mem      (从骨干输出取输入, 不是浅层 x_emb)
  x_mem → GLA recall → x_recall   (从 SlowMem 输出取输入)
  fused = norm(x_ash + x_mem + x_emb) + x_recall   (DirectAdd, 无 gate)

关键: 每路看到的是前一路处理过的深度表示, 不是平行竞争同一浅层输入.
"""
import torch, torch.nn as nn, torch.nn.functional as F, os, sys
sys.path.insert(0, r'F:\rwkv\frsmash_v36')
from frsmash_v36 import FRSMASHv36


class FRSMASHTriCross(FRSMASHv36):
    def __init__(self, voc_size, hidden_size, num_heads, num_layers, n_slots=4):
        super().__init__(voc_size, hidden_size, num_heads, num_layers, n_slots)
        self.mem_norm = nn.RMSNorm(hidden_size)
        # 交叉投影: 让每路适配前一路的输出分布
        self.recall_in_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, states=None, h_slow=None, recall_state=None, return_state=False, pos_offset=0):
        B,T=x.shape; D=self.D; dt=self.head.weight.dtype
        x_emb=self.em(x).to(dt)+self.pe[pos_offset:pos_offset+T].to(dt)
        if states is None: states=[None]*self.num_ssm
        if h_slow is None: h_slow=torch.zeros(B,D,device=x.device,dtype=dt)

        # 路1: SSM 骨干(从 x_emb)
        h=x_emb; new_states=[] if return_state else None
        for i,layer in enumerate(self.layers):
            s_in=states[i] if return_state else None
            h,s=layer(h,s_in)
            if return_state: new_states.append(s)
        x_ash=self.final_norm(h)

        # 路2: SlowMemory(从 x_ash 骨干深度输出, 不是 x_emb)
        inp_seq=self.mem_input_proj(x_ash)
        H_slow,h_slow=self.slow_cell(inp_seq,h_slow)
        x_mem=self.mem_norm(self.mem_proj(H_slow))

        # 路3: GLA recall(从 x_mem SlowMemory 输出, 交叉)
        recall_in=self.recall_in_proj(x_mem)
        if return_state or recall_state is not None:
            recall_out,recall_state=self.recall(recall_in,initial_state=recall_state,return_state=True)
        else:
            recall_out=self.recall(recall_in)
        x_recall=self.recall_norm(recall_out)

        # DirectAdd 融合(无 gate)
        fused=self.fusion_norm(x_ash+x_mem+x_emb)+x_recall
        logits=self.head(fused)
        if return_state: return logits,new_states,h_slow,recall_state
        return logits


if __name__=='__main__':
    DEV='cuda'; VOCAB=23005
    m=FRSMASHTriCross(VOCAB,512,8,8,4).to(DEV)
    n=sum(p.numel() for p in m.parameters())
    x=torch.randint(0,VOCAB,(2,512),device=DEV)
    with torch.no_grad(): o=m(x)
    print(f'TriCross params={n:,} ({n/1e6:.1f}M) ok')
