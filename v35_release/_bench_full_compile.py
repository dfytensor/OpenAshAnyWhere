import torch, sys, time
sys.path.insert(0, r"F:\OpenASH2605\FRSMASH\model")
from frsmash_v32 import FRSMASHv32
torch.manual_seed(0); dev=torch.device("cuda")
torch.set_float32_matmul_precision("high")
def P(*a): print(*a, flush=True)
VOCAB,Hd,HEADS,LAYERS,NS=23005,768,8,12,4
m=FRSMASHv32(VOCAB,Hd,HEADS,LAYERS,n_slots=NS).to(dev).eval()
CH=4096; x=torch.randint(1,VOCAB,(1,CH),device=dev)
def bench(fn,n=8):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n
with torch.no_grad():
    te=bench(lambda: m(x))
P(f"eager full forward: {te*1000:.0f}ms  ({CH/te:,.0f} tok/s)")
for mode in ["default","reduce-overhead"]:
    try:
        mc=torch.compile(m,mode=mode,dynamic=False)
        with torch.no_grad():
            _=mc(x); _=mc(x)            # warmup/compile
            tc=bench(lambda: mc(x),6)
        # correctness
        with torch.no_grad():
            a=m(x); b=mc(x)
        P(f"compile[{mode:16}]: {tc*1000:.0f}ms ({CH/tc:,.0f} tok/s) speedup={te/tc:.1f}x  "
          f"maxdiff={(a-b).abs().max().item():.2e}")
        P(f"   -> 100M wall-clock = {1e8/(CH/tc)/3600:.2f} h")
    except Exception as e:
        P(f"compile[{mode}] FAIL: {type(e).__name__}: {str(e)[:200]}")
