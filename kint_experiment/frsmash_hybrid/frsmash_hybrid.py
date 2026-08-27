"""FRSMASH-Hybrid: FRSMASH v3.6 骨干 + 局部窗口 softmax 注意力分支(Triton kernel, loop4-8 验证的 hybrid 成果).
FRSMASH 已有 GLA recall(全局线性), 缺的是 hybrid 里那半"局部窗口 softmax 精度"——本分支补上。
fusion: 原(x_ash·gate + x_mem·(1-gate) + x_emb) + x_recall + x_local   ← 新增 x_local
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, os, sys
sys.path.insert(0, r'F:\rwkv\frsmash_v36')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'arc_llm'))
from frsmash_v36 import FRSMASHv36
from triton_window_attn import window_attn as triton_window_attn


class LocalWindowAttn(nn.Module):
    """局部窗口 softmax 注意力(走 Triton O(T·W) kernel). heads 固定使 HD=32 匹配 kernel."""
    def __init__(self, d, heads=16, W=256):
        super().__init__(); self.h=heads; self.hd=32; self.W=W
        H=heads
        self.q=nn.Linear(d, H*32, bias=False); self.k=nn.Linear(d, H*32, bias=False)
        self.v=nn.Linear(d, H*32, bias=False); self.o=nn.Linear(H*32, d, bias=False)
    def forward(self, x):                       # x: (B, T, d)
        B,T,d=x.shape; H=self.h
        q=self.q(x).view(B,T,H,32).transpose(1,2).contiguous()
        k=self.k(x).view(B,T,H,32).transpose(1,2).contiguous()
        v=self.v(x).view(B,T,H,32).transpose(1,2).contiguous()
        # Triton 要求 WKV 是 2 的幂, 取覆盖 W+BQ 的下一个 2 的幂, 钳到 512
        wkv=1
        while wkv < self.W+64: wkv*=2
        WKV=min(wkv,512)
        a=triton_window_attn(q,k,v,self.W,BQ=64,WKV=WKV)     # (B,H,T,32)
        return self.o(a.transpose(1,2).reshape(B,T,H*32))


class FRSMASHHybrid(FRSMASHv36):
    def __init__(self, voc_size, hidden_size, num_heads, num_layers, n_slots=4,
                 local_heads=16, local_W=256):
        super().__init__(voc_size, hidden_size, num_heads, num_layers, n_slots)
        self.local_attn=LocalWindowAttn(hidden_size, local_heads, local_W)
        self.local_norm=nn.RMSNorm(hidden_size)

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
        inp_seq=self.mem_input_proj(x_emb); H_slow,h_slow=self.slow_cell(inp_seq,h_slow); x_mem=self.mem_proj(H_slow)
        if return_state or recall_state is not None:
            recall_out,recall_state=self.recall(x_emb,initial_state=recall_state,return_state=True)
        else:
            recall_out=self.recall(x_emb)
        x_recall=self.recall_norm(recall_out)
        x_local=self.local_norm(self.local_attn(x_emb))        # 新增: 局部窗口 softmax
        cat=torch.cat([x_ash,x_mem],dim=-1); gate=self.fusion_gate(cat)
        fused=self.fusion_norm(gate*x_ash+(1-gate)*x_mem+x_emb)+x_recall+x_local
        logits=self.head(fused)
        if return_state: return logits,new_states,h_slow,recall_state
        return logits


if __name__=='__main__':
    DEV='cuda'; VOCAB=23005
    for tag,h,L in [('vanilla d512/L8',512,8),('hybrid d512/L8',512,8)]:
        cls=FRSMASHv36 if tag.startswith('vanilla') else FRSMASHHybrid
        m=cls(VOCAB,h,8,L,n_slots=4).to(DEV)
        n=sum(p.numel() for p in m.parameters())
        x=torch.randint(0,VOCAB,(2,512),device=DEV)
        with torch.no_grad(): o=m(x)
        g=(o.mean()-o.mean()).abs().sum()  # 触发
        print(f'{tag}: params={n:,} ({n/1e6:.1f}M)  out={o.shape}  ok')
