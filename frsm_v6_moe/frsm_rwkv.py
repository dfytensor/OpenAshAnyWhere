"""
FRSM-RWKV Hybrid: RWKV 并行扫描 + FRSM 状态机结构
- chunk 内: 线性递推 H_t = A_t*H_{t-1} + B_t,用 cumsum 并行求解
- 状态每 token 都在演化(不再是冻结状态)
- block 边界: 可选 FRSM 非线性校正(用 H 计算门控)
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

class FRSM_V6_RWKV(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_scales=4,
                 n_experts=16, n_shared=1, block_size=64,
                 frsm_correct=True, top_k=4):
        super().__init__()
        self.d_model=d_model; self.num_scales=num_scales
        self.n_experts=n_experts; self.n_shared=n_shared
        self.block_size=block_size; self.frsm_correct=frsm_correct
        self.top_k=top_k; self.aux_loss=torch.tensor(0.0)
        E,S,D=n_experts,num_scales,d_model; dh=D//4
        self.embed=nn.Embedding(vocab_size,D); self.input_proj=nn.Linear(D,D)
        # 门控权重: (E,S,D,2D) — 前半给H,后半给inp(RWKV并行只用后半)
        for n in ['W_forget','W_input','W_cand']:
            setattr(self,n,nn.Parameter(torch.empty(E,S,D,2*D)))
            setattr(self,'b_'+n[2:],nn.Parameter(torch.empty(E,S,D)))
        self.gate_W1=nn.Parameter(torch.empty(E,S,dh,2*D))
        self.gate_b1=nn.Parameter(torch.empty(E,S,dh))
        self.gate_W2=nn.Parameter(torch.empty(E,S,1,dh))
        self.gate_b2=nn.Parameter(torch.empty(E,S,1))
        self.fusion_W=nn.Parameter(torch.empty(E,S*D,D))
        self.fusion_b=nn.Parameter(torch.empty(E,D))
        if n_shared>0:
            for n in ['W_forget','W_input','W_cand']:
                setattr(self,n+'_sh',nn.Parameter(torch.empty(n_shared,S,D,2*D)))
                setattr(self,'b_'+n.split('_')[1]+'_sh',nn.Parameter(torch.empty(n_shared,S,D)))
            self.gate_W1_sh=nn.Parameter(torch.empty(n_shared,S,dh,2*D))
            self.gate_b1_sh=nn.Parameter(torch.empty(n_shared,S,dh))
            self.gate_W2_sh=nn.Parameter(torch.empty(n_shared,S,1,dh))
            self.gate_b2_sh=nn.Parameter(torch.empty(n_shared,S,1))
            self.fusion_W_sh=nn.Parameter(torch.empty(n_shared,S*D,D))
            self.fusion_b_sh=nn.Parameter(torch.empty(n_shared,D))
        self.router=nn.Linear(D,E)
        self.output_norm=nn.LayerNorm(D); self.output_proj=nn.Linear(D,vocab_size)
        self._init_w()

    def _init_w(self):
        def _k(p):
            for e in range(p.size(0)):
                for s in range(self.num_scales):
                    nn.init.kaiming_uniform_(p[e,s],a=math.sqrt(5))
        for pn in ['W_forget','W_input','W_cand','gate_W1','gate_W2']:
            _k(getattr(self,pn))
        for e in range(self.n_experts):
            nn.init.kaiming_uniform_(self.fusion_W[e],a=math.sqrt(5))
        if self.n_shared>0:
            for pn in ['W_forget','W_input','W_cand','gate_W1','gate_W2']:
                _k(getattr(self,pn+'_sh'))
            for e in range(self.n_shared):
                nn.init.kaiming_uniform_(getattr(self,'fusion_W_sh')[e],a=math.sqrt(5))
        for n,p in self.named_parameters():
            if 'bias' in n: nn.init.zeros_(p)
        nn.init.zeros_(self.b_cand);nn.init.zeros_(self.gate_b1);nn.init.zeros_(self.gate_b2);nn.init.zeros_(self.fusion_b)
        nn.init.constant_(self.b_forget,1.0);nn.init.constant_(self.b_input,-2.0)
        if self.n_shared>0:
            nn.init.zeros_(self.b_cand_sh);nn.init.zeros_(self.gate_b1_sh);nn.init.zeros_(self.gate_b2_sh);nn.init.zeros_(self.fusion_b_sh)
            nn.init.constant_(self.b_forget_sh,1.0);nn.init.constant_(self.b_input_sh,-2.0)
        nn.init.normal_(self.router.weight,0,0.02);nn.init.normal_(self.embed.weight,0,0.02)
        nn.init.kaiming_uniform_(self.input_proj.weight,a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.output_proj.weight,a=math.sqrt(5))

    def _parallel_scan(self, A, B, H0):
        """并行求解 H_t = A_t * H_{t-1} + B_t (对角线性递推)
        A: (K, E, B, S, D)  B: (K, E, B, S, D)  H0: (E, B, S, D)
        返回: (K, E, B, S, D) — 全部 K 步的状态
        """
        K=A.shape[0]
        log_A=torch.log(A.clamp(min=1e-6))
        cum_log=torch.cumsum(log_A,dim=0)           # (K,...)
        prefix_A=torch.exp(cum_log)                   # ∏A_i
        B_scaled=B/(prefix_A+1e-8)                    # B_i/∏A_i
        cum_B=torch.cumsum(B_scaled,dim=0)            # ∑B_j/∏A_j
        return prefix_A*(H0.unsqueeze(0)+cum_B)       # (K,E,B,S,D)

    def _gates_rwkv(self, inp_block, Wf, Wi, Wc, bf, bi, bc, gW1, gb1, gW2, gb2):
        """RWKV 并行门控: 只用 inp(不用H),计算 A,B 线性递推系数
        inp_block: (B, K, D)
        返回 A: (K, E, B, S, D), B_rec: (K, E, B, S, D)
        """
        # 用 W 的后半(inp部分): W[:,:,:,D:] → (E,S,D,D)
        # f[b,t,e,s,i] = sum_d inp[b,t,d] * W[e,s,i,d+D]
        f=torch.sigmoid(torch.einsum('bkd,esid->bk esi',inp_block,Wf[:,:,:,self.d_model:])+bf)
        i=torch.sigmoid(torch.einsum('bkd,esid->bkesi',inp_block,Wi[:,:,:,self.d_model:])+bi)
        c=torch.tanh(torch.einsum('bkd,esid->bkesi',inp_block,Wc[:,:,:,self.d_model:])+bc)
        h1=F.gelu(torch.einsum('bkd,esid->bkesi',inp_block,gW1[:,:,:,self.d_model:])+gb1)
        st=torch.sigmoid(torch.einsum('bkesd,esod->bkeso',h1,gW2)+gb2)
        A=st*f+(1-st)        # (B,K,E,S,D)
        B_rec=st*i*c         # (B,K,E,S,D)
        # 转成 (K,E,B,S,D) 给 parallel_scan
        return A.permute(1,2,0,3,4), B_rec.permute(1,2,0,3,4)

    def _frsm_correct_step(self, H, inp, Wf, Wi, Wc, bf, bi, bc, gW1, gb1, gW2, gb2):
        """FRSM 非线性校正: 用 H 计算门控(捕捉长程依赖)
        H: (E,B,S,D)  inp: (B,D) → 返回校正后的 H
        """
        E,B,S,D=H.shape
        ie=inp.unsqueeze(0).unsqueeze(2).expand(E,B,S,D)
        g=torch.cat([H,ie],dim=-1)
        f=torch.sigmoid(torch.einsum('ebsj,esij->ebsi',g,Wf)+bf.unsqueeze(1))
        i=torch.sigmoid(torch.einsum('ebsj,esij->ebsi',g,Wi)+bi.unsqueeze(1))
        c=torch.tanh(torch.einsum('ebsj,esij->ebsi',g,Wc)+bc.unsqueeze(1))
        cand=f*H+i*c
        h1=F.gelu(torch.einsum('ebsj,esij->ebsi',g,gW1)+gb1.unsqueeze(1))
        st=torch.sigmoid(torch.einsum('ebsi,esoi->ebso',h1,gW2)+gb2.unsqueeze(1))
        return st*cand+(1-st)*H

    def _route(self, inp):
        l=self.router(inp)
        if self.training: l=l+torch.randn_like(l)*1.0
        probs=F.softmax(l,dim=-1)
        if self.top_k and self.top_k<self.n_experts:
            _,idx=probs.topk(self.top_k,dim=-1)
            mask=torch.zeros_like(probs);mask.scatter_(1,idx,1.0)
            probs=probs*mask;probs=probs/(probs.sum(-1,keepdim=True)+1e-9)
        return probs

    def _block_output(self, H_all, inp_block, fW, fb, B_sz):
        """计算 block 内所有 token 的输出
        H_all: (K,E,B,S,D)  inp_block: (B,K,D)
        返回 combined: (B,K,D)
        """
        K,E=H_all.shape[:2]; D=self.d_model
        Hf=H_all.reshape(K,E,B_sz,self.num_scales*D)
        fused=torch.einsum('kebp,epi->kebi',Hf,fW)+fb.unsqueeze(0).unsqueeze(2) # (K,E,B,D)
        probs=self._route(inp_block[:,0,:])  # (B,E)
        pe=probs.t().unsqueeze(0).unsqueeze(-1).expand(K,E,B_sz,D)  # (K,E,B,D)
        return ((pe*fused).sum(dim=1)).permute(1,0,2) # (B,K,D)

    def forward(self, x, targets=None):
        B,T=x.shape; E,S,D=self.n_experts,self.num_scales,self.d_model
        inp=self.input_proj(self.embed(x))
        H=torch.zeros(E,B,S,D,device=x.device,dtype=inp.dtype)
        Hs=torch.zeros(self.n_shared,B,S,D,device=x.device,dtype=inp.dtype) if self.n_shared>0 else None
        logits=None
        if targets is None:
            logits=torch.zeros(B,T,self.output_proj.out_features,device=x.device,dtype=torch.float16)
        aux=torch.zeros((),device=x.device,dtype=torch.float32)
        K=self.block_size; loss_val=None; ts=0

        for start in range(0,T,K):
            end=min(start+K,T); k=end-start
            ib=inp[:,start:end,:]  # (B,k,D)

            # === Pass 1: RWKV 并行(只用inp) ===
            A1,B1=self._gates_rwkv(ib,self.W_forget,self.W_input,self.W_cand,
                self.b_forget,self.b_input,self.b_cand,self.gate_W1,self.gate_b1,self.gate_W2,self.gate_b2)
            A1=A1[:k];B1=B1[:k]  # 截断到最后一个有效步
            H_rkv=self._parallel_scan(A1,B1,H)  # (k,E,B,S,D)

            # === Pass 2: FRSM 校正(可选,用H计算门控) ===
            if self.frsm_correct and self.training:
                H_last=H_rkv[-1]  # (E,B,S,D)
                H_corr=self._frsm_correct_step(H_last,ib[:,-1,:],
                    self.W_forget,self.W_input,self.W_cand,
                    self.b_forget,self.b_input,self.b_cand,self.gate_W1,self.gate_b1,self.gate_W2,self.gate_b2)
                H=H_corr
            else:
                H=H_rkv[-1]

            # 共享专家(同样RWKV并行)
            if self.n_shared>0:
                As1,Bs1=self._gates_rwkv(ib,self.W_forget_sh,self.W_input_sh,self.W_cand_sh,
                    self.b_forget_sh,self.b_input_sh,self.b_cand_sh,self.gate_W1_sh,self.gate_b1_sh,self.gate_W2_sh,self.gate_b2_sh)
                Hs_rkv=self._parallel_scan(As1[:k],Bs1[:k],Hs)
                if self.frsm_correct and self.training:
                    Hs=self._frsm_correct_step(Hs_rkv[-1],ib[:,-1,:],
                        self.W_forget_sh,self.W_input_sh,self.W_cand_sh,
                        self.b_forget_sh,self.b_input_sh,self.b_cand_sh,self.gate_W1_sh,self.gate_b1_sh,self.gate_W2_sh,self.gate_b2_sh)
                else:
                    Hs=Hs_rkv[-1]
                # 共享专家输出
                Hsf=Hs_rkv.reshape(k,self.n_shared,B,S*D)
                sfused=torch.einsum('kebp,epi->kebi',Hsf,self.fusion_W_sh)+self.fusion_b_sh.unsqueeze(0).unsqueeze(2)
                sf_out=sfused.sum(dim=1).permute(1,0,2) # (B,k,D)
            else:
                sf_out=0

            # 路由专家输出
            routed=self._block_output(H_rkv,ib,self.fusion_W,self.fusion_b,B)
            comb=routed+sf_out  # (B,k,D)
            lc=self.output_proj(self.output_norm(comb))

            if targets is not None:
                vs=self.output_proj.out_features
                cl=F.cross_entropy(lc.reshape(-1,vs),targets[:,start:end].reshape(-1),ignore_index=0)
                loss_val=(loss_val+cl*(k/T)) if loss_val is not None else cl*(k/T)
            else:
                logits[:,start:end,:]=lc

            probs=self._route(ib[:,0,:])
            tpe=probs.mean(0);aux=aux+E*torch.sum(tpe*probs.mean(0))
            ts=end

        self.aux_loss=aux/max(1,(T+K-1)//K)
        if targets is not None: return loss_val
        return logits

    @torch.no_grad()
    def generate_step(self, token, h_prev, top_k=4):
        """推理: 单步 FRSM(非线性,完整门控)"""
        H,Hs=h_prev; B=token.size(0)
        inp=self.input_proj(self.embed(token).squeeze(1))
        E=self.n_experts; k=min(top_k or E,E)
        # top-k 路由
        probs=self._route(inp)
        # 全专家 FRSM 更新(推理用完整非线性)
        Hn,fused=self._estep_full(H,inp,self.W_forget,self.W_input,self.W_cand,
            self.b_forget,self.b_input,self.b_cand,self.gate_W1,self.gate_b1,
            self.gate_W2,self.gate_b2,self.fusion_W,self.fusion_b)
        if self.n_shared>0:
            Hsn,sf=self._estep_full(Hs,inp,self.W_forget_sh,self.W_input_sh,self.W_cand_sh,
                self.b_forget_sh,self.b_input_sh,self.b_cand_sh,self.gate_W1_sh,self.gate_b1_sh,
                self.gate_W2_sh,self.gate_b2_sh,self.fusion_W_sh,self.fusion_b_sh)
            sf=sf.sum(dim=0)
        else: Hsn,sf=None,0
        comb=((probs.t().unsqueeze(-1)*fused).sum(dim=0))+sf
        return self.output_proj(self.output_norm(comb)),(Hn,Hsn)

    def _estep_full(self, H, inp, Wf, Wi, Wc, bf, bi, bc, gW1, gb1, gW2, gb2, fW, fb):
        """完整 FRSM 门控(H+inp),用于推理"""
        E,B=H.shape[:2]; S,D=self.num_scales,self.d_model
        ie=inp.unsqueeze(0).unsqueeze(2).expand(E,B,S,D)
        g=torch.cat([H,ie],dim=-1)
        f=torch.sigmoid(torch.einsum('ebsj,esij->ebsi',g,Wf)+bf.unsqueeze(1))
        i=torch.sigmoid(torch.einsum('ebsj,esij->ebsi',g,Wi)+bi.unsqueeze(1))
        c=torch.tanh(torch.einsum('ebsj,esij->ebsi',g,Wc)+bc.unsqueeze(1))
        cand=f*H+i*c
        h1=F.gelu(torch.einsum('ebsj,esij->ebsi',g,gW1)+gb1.unsqueeze(1))
        st=torch.sigmoid(torch.einsum('ebsi,esoi->ebso',h1,gW2)+gb2.unsqueeze(1))
        Hn=st*cand+(1-st)*H
        fused=torch.einsum('ebk,eki->ebi',Hn.reshape(E,B,S*D),fW)+fb.unsqueeze(1)
        return Hn,fused
