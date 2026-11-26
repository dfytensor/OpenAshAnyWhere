import torch, torch.nn.functional as F, sys, time, math
sys.path.insert(0, r"F:\OpenASH2605\FRSMASH\model")
from frsmash_v32 import FRSMASHv32
torch.manual_seed(0); dev=torch.device("cuda")
def P(*a): print(*a, flush=True)
VOCAB,Hd,HEADS,LAYERS,NS=23005,768,8,12,4
m=FRSMASHv32(VOCAB,Hd,HEADS,LAYERS,n_slots=NS).to(dev).eval()
CH=4096; D=768; L=LAYERS

class Runner:
    def __init__(self, m, CH):
        self.m=m; self.CH=CH
        self.x=torch.zeros(1,CH,dtype=torch.long,device=dev)
        self.st=[torch.zeros(1,4,192,device=dev) for _ in range(L)]
        self.hs=torch.zeros(1,D,device=dev)
        self.out_logits=torch.zeros(1,CH,VOCAB,device=dev)
        self.out_st=[torch.zeros(1,4,192,device=dev) for _ in range(L)]
        self.out_hs=torch.zeros(1,D,device=dev)
        self.graph=torch.cuda.CUDAGraph()
        s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                with torch.no_grad(): self._fwd()
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(self.graph):
            with torch.no_grad(): self._fwd()
    def _fwd(self):
        m=self.m
        xe=m.em(self.x).to(m.head.weight.dtype)
        h=xe; ns=[]
        for i,layer in enumerate(m.layers):
            h,s=layer(h,self.st[i]); ns.append(s)
        x_ash=m.final_norm(h)
        inp=m.mem_input_proj(xe)
        hs=self.hs; outs=[]
        for t in range(self.CH):
            hs=m.slow_cell(inp[:,t],hs); outs.append(hs)
        x_mem=m.mem_proj(torch.stack(outs,1))
        cat=torch.cat([x_ash,x_mem],-1)
        g=m.fusion_gate(cat)
        fused=m.fusion_norm(g*x_ash+(1-g)*x_mem+xe)
        self.out_logits.copy_(m.head(fused))
        for i in range(L): self.out_st[i].copy_(ns[i])
        self.out_hs.copy_(hs)
    def run(self, x_tok, st, hs):
        self.x.copy_(x_tok)
        for i in range(L): self.st[i].copy_(st[i])
        self.hs.copy_(hs)
        self.graph.replay()
        return self.out_logits, [s.clone() for s in self.out_st], self.out_hs.clone()

r=Runner(m,CH)
# correctness vs eager
xt=torch.randint(1,VOCAB,(1,CH),device=dev)
with torch.no_grad():
    lg0,st0,hs0=m(xt,[None]*L,torch.zeros(1,D,device=dev),return_state=True)
lg,st,hs=r.run(xt,[torch.zeros(1,4,192,device=dev)]*L,torch.zeros(1,D,device=dev))
P(f"graph vs eager: max|logits|diff={(lg0-lg).abs().max().item():.2e} "
  f"max|state|diff={(st0[0]-st[0]).abs().max().item():.2e} "
  f"max|hs|diff={(hs0-hs).abs().max().item():.2e}")
def bench(n=30):
    st=[torch.zeros(1,4,192,device=dev)]*L; hs=torch.zeros(1,D,device=dev)
    for _ in range(5):
        lg,st,hs=r.run(torch.randint(1,VOCAB,(1,CH),device=dev),st,hs)
    torch.cuda.synchronize(); t0=time.time()
    st=[torch.zeros(1,4,192,device=dev)]*L; hs=torch.zeros(1,D,device=dev)
    tot=0
    for _ in range(n):
        xb=torch.randint(1,VOCAB,(1,CH),device=dev)
        lg,st,hs=r.run(xb,st,hs)
    torch.cuda.synchronize(); return (time.time()-t0)/n
t=bench(); tps=CH/t
P(f"\nCUDAGraph full forward/chunk: {t*1000:.1f}ms  ({tps:,.0f} tok/s)")
P(f"  -> 100M tokens wall-clock = {1e8/tps/60:.1f} min")
