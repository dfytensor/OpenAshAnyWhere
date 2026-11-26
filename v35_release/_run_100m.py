import torch, torch.nn.functional as F, sys, time, math, json
sys.path.insert(0, r"F:\OpenASH2605\FRSMASH\model")
from frsmash_v32 import FRSMASHv32
torch.manual_seed(0); dev=torch.device("cuda")
def P(*a): print(*a, flush=True)
VOCAB,Hd,HEADS,LAYERS,NS=23005,768,8,12,4
m=FRSMASHv32(VOCAB,Hd,HEADS,LAYERS,n_slots=NS).to(dev).eval()
CH=2048; D=768; L=LAYERS; TARGET=100_000_000; UNIF=math.log(VOCAB)
CKPT=r"F:\OpenASH2605\_ppl100m_ckpt.json"

class Runner:
    def __init__(self):
        self.x=torch.zeros(1,CH,dtype=torch.long,device=dev)
        self.st=[torch.zeros(1,4,192,device=dev) for _ in range(L)]
        self.hs=torch.zeros(1,D,device=dev)
        self.olg=torch.zeros(1,CH,VOCAB,device=dev)
        self.ost=[torch.zeros(1,4,192,device=dev) for _ in range(L)]
        self.ohs=torch.zeros(1,D,device=dev)
        self.g=torch.cuda.CUDAGraph(); mm=m
        s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                xe=mm.em(self.x).to(mm.head.weight.dtype); h=xe; ns=[]
                for i,ly in enumerate(mm.layers): h,s2=ly(h,self.st[i]); ns.append(s2)
                xa=mm.final_norm(h); inp=mm.mem_input_proj(xe); hs=self.hs; ot=[]
                for t in range(CH): hs=mm.slow_cell(inp[:,t],hs); ot.append(hs)
                xm=mm.mem_proj(torch.stack(ot,1)); cat=torch.cat([xa,xm],-1)
                gg=mm.fusion_gate(cat); fu=mm.fusion_norm(gg*xa+(1-gg)*xm+xe)
                self.olg.copy_(mm.head(fu))
                for i in range(L): self.ost[i].copy_(ns[i])
                self.ohs.copy_(hs)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(self.g):
            xe=mm.em(self.x).to(mm.head.weight.dtype); h=xe; ns=[]
            for i,ly in enumerate(mm.layers): h,s2=ly(h,self.st[i]); ns.append(s2)
            xa=mm.final_norm(h); inp=mm.mem_input_proj(xe); hs=self.hs; ot=[]
            for t in range(CH): hs=mm.slow_cell(inp[:,t],hs); ot.append(hs)
            xm=mm.mem_proj(torch.stack(ot,1)); cat=torch.cat([xa,xm],-1)
            gg=mm.fusion_gate(cat); fu=mm.fusion_norm(gg*xa+(1-gg)*xm+xe)
            self.olg.copy_(mm.head(fu))
            for i in range(L): self.ost[i].copy_(ns[i])
            self.ohs.copy_(hs)
    def run(self,xb,st,hs):
        self.x.copy_(xb)
        for i in range(L): self.st[i].copy_(st[i])
        self.hs.copy_(hs); self.g.replay()
        return self.olg,[s.clone() for s in self.ost],self.ohs.clone()

r=Runner()
P(f"start: target {TARGET:,} tokens, chunk {CH}, untrained NLL~{UNIF:.3f}")
st=[torch.zeros(1,4,192,device=dev)]*L; hs=torch.zeros(1,D,device=dev)
tot=0.0; mx=0.0; n=0; first=None; t0=time.time(); nchunk=TARGET//CH
for ci in range(nchunk):
    xb=torch.randint(1,VOCAB,(1,CH),device=dev)
    lg,st,hs=r.run(xb,st,hs)
    nll=-F.log_softmax(lg.float(),-1).gather(-1,xb.unsqueeze(-1)).squeeze(-1)[0]
    tot+=nll.sum().item(); mx=max(mx,nll.max().item()); n+=nll.numel()
    if n% (2_000_000) < CH:
        cur=tot/n
        if first is None: first=cur
        el=time.time()-t0; eta=el*(TARGET/n -1)
        nan=torch.isnan(nll).any().item()
        P(f"  {n/1e6:7.2f}M  NLL={cur:.4f}  maxTokNLL={mx:.2f}  "
          f"drift={cur-first:+.4f}  nan={nan}  ETA={eta/60:.0f}min")
    if n % 10_000_000 < CH:
        json.dump({"n":n,"mean_nll":tot/n,"max_nll":mx,"first":first,
                   "elapsed":time.time()-t0}, open(CKPT,"w"))
dt=time.time()-t0; cur=tot/n
P("\n"+"="*60)
P(f"TOTAL: {n:,} tokens = {n/1e6:.0f}M  in {dt/60:.1f} min  ({n/dt:,.0f} tok/s)")
P(f"mean NLL   = {cur:.5f}  (unif {UNIF:.3f})   PPL = {math.exp(min(cur,50)):.1f}")
P(f"max tok NLL= {mx:.4f}   (bounded<50 => no blow-up)")
P(f"drift 0->end = {first:.5f} -> {cur:.5f} ({cur-first:+.5f})")
ok = (not math.isnan(cur) and mx<50 and abs(cur-UNIF)<3)
P(f"=> 1亿(100M) context PPL {'STABLE' if ok else 'UNSTABLE'}")
P(f"=> peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB (O(chunk), flat)")
json.dump({"n":n,"mean_nll":tot/n,"max_nll":mx,"first":first,"stable":ok},
          open(CKPT,"w"))
