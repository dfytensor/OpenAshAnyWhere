import torch, sys, time, copy
sys.path.insert(0, r"F:\OpenASH2605\FRSMASH\model")
from frsmash_v32 import FRSMASHv32
torch.manual_seed(0); dev=torch.device("cuda")
def P(*a): print(*a, flush=True)
VOCAB,Hd,HEADS,LAYERS,NS=23005,768,8,12,4
m=FRSMASHv32(VOCAB,Hd,HEADS,LAYERS,n_slots=NS).to(dev).eval()
CH=4096; cell=m.slow_cell
inp_buf=torch.randn(1,CH,Hd,device=dev)
h0_buf =torch.zeros(1,Hd,device=dev)
out_buf=torch.zeros(1,CH,Hd,device=dev)
fin_buf=torch.zeros(1,Hd,device=dev)

def eager_loop():
    h=h0_buf.clone(); o=out_buf.clone()
    for t in range(CH):
        with torch.no_grad(): h=cell(inp_buf[:,t],h)
        o[:,t]=h
    return o,h

# build CUDAGraph of the slow loop
def build_graph():
    g=torch.cuda.CUDAGraph()
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            h=h0_buf.clone()
            for t in range(CH):
                with torch.no_grad(): h=cell(inp_buf[:,t],h)
                out_buf[:,t]=h
            fin_buf.copy_(h)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        h=h0_buf.clone()
        for t in range(CH):
            with torch.no_grad(): h=cell(inp_buf[:,t],h)
            out_buf[:,t]=h
        fin_buf.copy_(h)
    return g
g=build_graph()
def graph_loop():
    g.replay(); return out_buf, fin_buf

# correctness
torch.cuda.synchronize()
oe,he=eager_loop(); og,hg=graph_loop()
P(f"correctness graph vs eager: max|out|diff={(oe-og).abs().max().item():.2e}  "
  f"max|h|diff={(he-hg).abs().max().item():.2e}")

def bench(fn,n=20):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n
te=bench(eager_loop); tg=bench(graph_loop)
P(f"eager slow loop: {te*1000:.1f}ms ({CH/te:,.0f} tok/s)")
P(f"CUDAGraph loop : {tg*1000:.1f}ms ({CH/tg:,.0f} tok/s)  speedup={te/tg:.1f}x")
P(f"  -> SlowMemory 100M wall-clock = {1e8/(CH/tg)/3600:.3f} h")
