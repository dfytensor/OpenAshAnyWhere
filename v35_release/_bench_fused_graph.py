import torch, sys, time
sys.path.insert(0, r"F:\OpenASH2605\FRSMASH\model")
from frsmash_v32 import FRSMASHv32
torch.manual_seed(0); dev=torch.device("cuda")
torch.set_float32_matmul_precision("high")
def P(*a): print(*a, flush=True)
VOCAB,Hd,HEADS,LAYERS,NS=23005,768,8,12,4
m=FRSMASHv32(VOCAB,Hd,HEADS,LAYERS,n_slots=NS).to(dev).eval()
CH=4096; cell=m.slow_cell
inp_buf=torch.randn(1,CH,Hd,device=dev); h0_buf=torch.zeros(1,Hd,device=dev)
out_buf=torch.zeros(1,CH,Hd,device=dev); fin_buf=torch.zeros(1,Hd,device=dev)

def bench(fn,n=20):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n

# fused compiled cell
cell_c=torch.compile(cell,mode="max-autotune",dynamic=False)
with torch.no_grad(): _=cell_c(inp_buf[:,0],h0_buf); _=cell_c(inp_buf[:,0],h0_buf)

def build(cell):
    g=torch.cuda.CUDAGraph(); s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
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
# correctness vs eager
with torch.no_grad():
    he=cell(inp_buf[:,0],h0_buf); hc=cell_c(inp_buf[:,0],h0_buf)
P(f"compiled cell vs eager diff={(he-hc).abs().max().item():.2e}")
g=build(cell_c)
def greplay(): g.replay()
tg=bench(greplay,30)
P(f"CUDAGraph loop (compiled cell): {tg*1000:.1f}ms ({CH/tg:,.0f} tok/s)")
P(f"  -> SlowMemory 100M = {1e8/(CH/tg)/60:.1f} min")
