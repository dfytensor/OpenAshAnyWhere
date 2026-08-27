import torch, sys, time
sys.path.insert(0, r"F:\OpenASH2605\FRSMASH\model")
from frsmash_v32 import FRSMASHv32
torch.manual_seed(0); dev=torch.device("cuda")
def P(*a): print(*a, flush=True)
import triton; P("triton", triton.__version__, "OK")
VOCAB,Hd,HEADS,LAYERS,NS=23005,768,8,12,4
m=FRSMASHv32(VOCAB,Hd,HEADS,LAYERS,n_slots=NS).to(dev).eval()
CH=4096
inp=m.mem_input_proj(m.em(torch.randint(1,VOCAB,(1,CH),device=dev)).to(m.head.weight.dtype))
h0=torch.zeros(1,Hd,device=dev)
def loop(cell,inp,h):
    H=torch.empty(1,CH,Hd,device=dev)
    for t in range(CH): h=cell(inp[:,t],h); H[:,t]=h
    return H
def bench(fn,n=5):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n
with torch.no_grad():
    te=bench(lambda: loop(m.slow_cell,inp,h0))
P(f"eager slow loop: {te*1000:.0f}ms ({CH/te:,.0f} tok/s)")
for mode in ["default","reduce-overhead","max-autotune"]:
    try:
        cc=torch.compile(m.slow_cell,mode=mode,dynamic=False)
        with torch.no_grad():
            _=cc(inp[:,0],h0)  # compile warmup
            tc=bench(lambda: loop(cc,inp,h0),5)
        P(f"compile[{mode:16}]: {tc*1000:.0f}ms ({CH/tc:,.0f} tok/s) speedup={te/tc:.1f}x")
    except Exception as e:
        P(f"compile[{mode}] FAIL: {type(e).__name__}: {str(e)[:160]}")
