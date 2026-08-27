import math, torch, torch.nn as nn, torch.nn.functional as F

class FRSM_V6_DenseMoE(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_scales=4,
                 n_experts=16, n_shared=1, router_noise=1.0,
                 aux_loss_weight=0.01, chunk_size=0, top_k=4,
                 chunk_pattern=None, chunk_wave=None):
        super().__init__()
        self.d_model=d_model; self.num_scales=num_scales
        self.n_experts=n_experts; self.n_shared=n_shared; self.router_noise=router_noise
        self.aux_loss_weight=aux_loss_weight; self.chunk_size=chunk_size
        self.top_k=top_k; self.aux_loss=torch.tensor(0.0)
        # 生成三角波动 chunk pattern: chunk_wave=(min,max) → [1,2,4,8,16,8,4,2]
        if chunk_wave:
            lo,hi=chunk_wave
            up=[]; c=lo
            while c<=hi: up.append(c); c*=2
            down=up[-2::-1] if len(up)>1 else []
            self.chunk_pattern=up+down  # 完整三角波
        else:
            self.chunk_pattern=chunk_pattern
        E,S,D=n_experts,num_scales,d_model; dh=D//4
        self.embed=nn.Embedding(vocab_size,D); self.input_proj=nn.Linear(D,D)
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
        nn.init.zeros_(self.b_cand); nn.init.zeros_(self.gate_b1); nn.init.zeros_(self.gate_b2); nn.init.zeros_(self.fusion_b)
        nn.init.constant_(self.b_forget,1.0); nn.init.constant_(self.b_input,-2.0)
        if self.n_shared>0:
            nn.init.zeros_(self.b_cand_sh); nn.init.zeros_(self.gate_b1_sh); nn.init.zeros_(self.gate_b2_sh); nn.init.zeros_(self.fusion_b_sh)
            nn.init.constant_(self.b_forget_sh,1.0); nn.init.constant_(self.b_input_sh,-2.0)
        nn.init.normal_(self.router.weight,0,0.02); nn.init.normal_(self.embed.weight,0,0.02)
        nn.init.kaiming_uniform_(self.input_proj.weight,a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.output_proj.weight,a=math.sqrt(5))

    def _estep(self, H, inp, Wf, Wi, Wc, bf, bi, bc, gW1, gb1, gW2, gb2, fW, fb):
        E,B=H.shape[:2]; S,D=self.num_scales,self.d_model
        inp=inp.reshape(-1,D)
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
        return Hn, fused

    def _route(self, inp):
        l=self.router(inp)
        if self.training and self.router_noise>0: l=l+torch.randn_like(l)*self.router_noise
        probs=F.softmax(l,dim=-1)
        # soft top-k: 算全部专家,但只让 top-k 有非零权重(强制特化)
        if self.top_k is not None and self.top_k<self.n_experts:
            k=self.top_k
            _,idx=probs.topk(k,dim=-1)
            mask=torch.zeros_like(probs); mask.scatter_(1,idx,1.0)
            probs=probs*mask
            probs=probs/(probs.sum(dim=-1,keepdim=True)+1e-9)
        return probs

    def _chunk_compute(self, Hf, Hsf, inf, pbf):
        Hnf,fused_f=self._estep(Hf,inf,self.W_forget,self.W_input,self.W_cand,
            self.b_forget,self.b_input,self.b_cand,self.gate_W1,self.gate_b1,
            self.gate_W2,self.gate_b2,self.fusion_W,self.fusion_b)
        if self.n_shared>0:
            Hsnf,sf=self._estep(Hsf,inf,self.W_forget_sh,self.W_input_sh,self.W_cand_sh,
                self.b_forget_sh,self.b_input_sh,self.b_cand_sh,self.gate_W1_sh,self.gate_b1_sh,
                self.gate_W2_sh,self.gate_b2_sh,self.fusion_W_sh,self.fusion_b_sh)
            sf=sf.sum(dim=0)
        else: Hsnf,sf=None,0
        comb_f=((pbf.t().unsqueeze(-1)*fused_f).sum(dim=0))+sf
        return Hnf, Hsnf, comb_f

    def forward(self,x,h_prev=None,return_state=False,targets=None):
        B,T=x.shape; E,S,D=self.n_experts,self.num_scales,self.d_model
        xe=self.embed(x); iseq=self.input_proj(xe)
        if h_prev is None:
            H=torch.zeros(E,B,S,D,device=x.device,dtype=iseq.dtype)
            Hs=torch.zeros(self.n_shared,B,S,D,device=x.device,dtype=iseq.dtype) if self.n_shared>0 else None
        else: H,Hs=h_prev
        logits=None
        if targets is None:
            logits=torch.zeros(B,T,self.output_proj.out_features,device=x.device,dtype=torch.float16)
        aux=torch.zeros((),device=x.device,dtype=torch.float32)
        # 构建 chunk 序列: 支持 pattern [1,4,4,4,4] 混合模式
        if self.chunk_pattern:
            chunks=[]; pos=0; pi=0
            while pos<T:
                c=self.chunk_pattern[pi%len(self.chunk_pattern)]
                c=min(c,T-pos); chunks.append(c); pos+=c; pi+=1
        else:
            C=self.chunk_size if self.chunk_size>0 else max(1,int(math.sqrt(T)))
            chunks=[C]*((T+C-1)//C)
            chunks[-1]=T-sum(chunks[:-1]) if len(chunks)>1 else T
        loss_val=None

        ts=0
        for ch in chunks:
            te=ts+ch;
            ic=iseq[:,ts:te,:]; bch=B*ch; inf=ic.reshape(bch,D)
            Hf=H.unsqueeze(2).expand(E,B,ch,S,D).reshape(E,bch,S,D)
            Hsf=Hs.unsqueeze(2).expand(self.n_shared,B,ch,S,D).reshape(self.n_shared,bch,S,D) if Hs is not None else None

            probs=self._route(ic[:,0,:])
            pbf=probs.unsqueeze(1).expand(B,ch,E).reshape(bch,E)

            if self.training:
                out=torch.utils.checkpoint.checkpoint(self._chunk_compute,Hf,Hsf,inf,pbf,use_reentrant=True)
                Hnf,Hsnf,comb_f=out
            else:
                Hnf,fused_f=self._estep(Hf,inf,self.W_forget,self.W_input,self.W_cand,
                    self.b_forget,self.b_input,self.b_cand,self.gate_W1,self.gate_b1,
                    self.gate_W2,self.gate_b2,self.fusion_W,self.fusion_b)
                if self.n_shared>0:
                    Hsnf,sf=self._estep(Hsf,inf,self.W_forget_sh,self.W_input_sh,self.W_cand_sh,
                        self.b_forget_sh,self.b_input_sh,self.b_cand_sh,self.gate_W1_sh,self.gate_b1_sh,
                        self.gate_W2_sh,self.gate_b2_sh,self.fusion_W_sh,self.fusion_b_sh)
                    sf=sf.sum(dim=0)
                else: Hsnf,sf=None,0
                comb_f=((pbf.t().unsqueeze(-1)*fused_f).sum(dim=0))+sf

            comb=comb_f.reshape(B,ch,D)
            lc=self.output_proj(self.output_norm(comb))

            if targets is not None:
                vs=self.output_proj.out_features
                cl=F.cross_entropy(lc.reshape(-1,vs),targets[:,ts:te].reshape(-1),ignore_index=0)
                loss_val=(loss_val+cl*(ch/T)) if loss_val is not None else cl*(ch/T)
            else:
                logits[:,ts:te,:]=lc

            li=torch.arange(B,device=x.device)*ch+(ch-1)
            H=Hnf[:,li,:,:]
            Hs=Hsnf[:,li,:,:] if Hsnf is not None else None
            tpe=probs.mean(0); aux=aux+E*torch.sum(tpe*probs.mean(0))
            ts=te

        self.aux_loss=aux/max(1,len(chunks))
        if targets is not None: return loss_val
        if return_state: return logits,(H,Hs)
        return logits

    @torch.no_grad()
    def generate_step(self,token,h_prev,top_k=4):
        H,Hs=h_prev; B=token.size(0)
        xe=self.embed(token).squeeze(1); inp=self.input_proj(xe)
        E=self.n_experts; k=min(top_k,E)

        # router -> top-k 选中
        logits=self.router(inp)
        probs=F.softmax(logits,dim=-1)
        vals,idx=probs.topk(k,dim=-1)
        top_w=vals/(vals.sum(dim=-1,keepdim=True)+1e-9)

        # 只 gather 选中专家的参数
        H_sel=H[idx,torch.arange(B,device=token.device).unsqueeze(1).expand(B,k)]

        Wf=self.W_forget[idx]; Wi=self.W_input[idx]; Wc=self.W_cand[idx]
        bf=self.b_forget[idx]; bi=self.b_input[idx]; bc=self.b_cand[idx]
        gW1=self.gate_W1[idx]; gb1=self.gate_b1[idx]
        gW2=self.gate_W2[idx]; gb2=self.gate_b2[idx]
        fW=self.fusion_W[idx]; fb=self.fusion_b[idx]

        ie=inp.unsqueeze(1).unsqueeze(2).expand(B,k,self.num_scales,self.d_model)
        g=torch.cat([H_sel,ie],dim=-1)
        f=torch.sigmoid(torch.einsum('bksj,bksij->bksi',g,Wf)+bf)
        i=torch.sigmoid(torch.einsum('bksj,bksij->bksi',g,Wi)+bi)
        c=torch.tanh(torch.einsum('bksj,bksij->bksi',g,Wc)+bc)
        cand=f*H_sel+i*c
        h1=F.gelu(torch.einsum('bksj,bksij->bksi',g,gW1)+gb1)
        st=torch.sigmoid(torch.einsum('bksi,bksoi->bkso',h1,gW2)+gb2)
        Hn_sel=st*cand+(1-st)*H_sel
        Hf=Hn_sel.reshape(B,k,self.num_scales*self.d_model)
        fused=torch.einsum('bkp,bkpi->bki',Hf,fW)+fb

        # scatter 更新状态
        ar=torch.arange(B,device=token.device).unsqueeze(1)
        Hn=H.clone()
        Hn[idx,ar]=Hn_sel

        # 共享专家
        if self.n_shared>0:
            Hs_new,_=self._estep(Hs,inp,self.W_forget_sh,self.W_input_sh,self.W_cand_sh,
                self.b_forget_sh,self.b_input_sh,self.b_cand_sh,self.gate_W1_sh,self.gate_b1_sh,
                self.gate_W2_sh,self.gate_b2_sh,self.fusion_W_sh,self.fusion_b_sh)
        else: Hs_new=None

        combined=(top_w.unsqueeze(-1)*fused).sum(dim=1)
        if self.n_shared>0:
            _,sf=self._estep(Hs,inp,self.W_forget_sh,self.W_input_sh,self.W_cand_sh,
                self.b_forget_sh,self.b_input_sh,self.b_cand_sh,self.gate_W1_sh,self.gate_b1_sh,
                self.gate_W2_sh,self.gate_b2_sh,self.fusion_W_sh,self.fusion_b_sh)
            combined=combined+sf.sum(dim=0)
        return self.output_proj(self.output_norm(combined)),(Hn,Hs_new)
