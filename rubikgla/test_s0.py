import sys, torch, time
sys.path.insert(0, r"F:\夸克\rubikgla")
import importlib
import delta_triton as DT
importlib.reload(DT)
from delta_triton import delta_scan_triton

DEV = "cuda"
torch.manual_seed(0)
B, L, KV, S, KVG, DH = 4, 128, 5, 2, 2, 32
G, H = KV * S, KV * KVG
dt = torch.bfloat16
k = (torch.randn(B, L, G, DH, device=DEV) * 0.3).to(dt).requires_grad_(True)
v = (torch.randn(B, L, G, DH, device=DEV) * 0.3).to(dt).requires_grad_(True)
r = (torch.rand(B, L, G, DH, device=DEV) * 0.3 + 1e-3).to(dt).requires_grad_(True)
e = (torch.rand(B, L, G, DH, device=DEV) * 0.3 + 0.05).to(dt).requires_grad_(True)
q = (torch.randn(B, L, H, DH, device=DEV) * 0.5).to(dt).requires_grad_(True)

# 1) 无 s0 (回归)
y1 = delta_scan_triton(k, v, r, e, q, KV, S, KVG)
y1.float().pow(2).mean().backward()
print("无s0 路径: OK, dk=%.3e" % k.grad.abs().mean().item())
for x in (k, v, r, e, q): x.grad = None

# 2) 带 s0 (接力): s0=随机 vs s0=0 应不同; s0 反传不应崩
S0 = (torch.randn(B, G, DH, DH, device=DEV) * 0.1).to(dt).requires_grad_(True)
y2 = delta_scan_triton(k, v, r, e, q, KV, S, KVG, s0=S0)
y2.float().pow(2).mean().backward()
print("带s0 路径: OK, dy 有限:", torch.isfinite(y2).all().item())

# 3) s0=0 应与无 s0 前向完全一致
S0z = torch.zeros(B, G, DH, DH, device=DEV, dtype=dt)
y3 = delta_scan_triton(k.detach(), v.detach(), r.detach(), e.detach(), q.detach(), KV, S, KVG, s0=S0z)
y1b = delta_scan_triton(k.detach(), v.detach(), r.detach(), e.detach(), q.detach(), KV, S, KVG)
print("s0=0 与 无s0 前向 max diff:", (y3.float() - y1b.float()).abs().max().item())

# 4) 速度 (B32)
kB = (torch.randn(32, 256, G, DH, device=DEV) * 0.3).to(dt).requires_grad_(True)
vB = (torch.randn(32, 256, G, DH, device=DEV) * 0.3).to(dt).requires_grad_(True)
rB = (torch.rand(32, 256, G, DH, device=DEV) * 0.3 + 1e-3).to(dt).requires_grad_(True)
eB = (torch.rand(32, 256, G, DH, device=DEV) * 0.3 + 0.05).to(dt).requires_grad_(True)
qB = torch.randn(32, 256, H, DH, device=DEV).to(dt).requires_grad_(True)
S0B = (torch.randn(32, G, DH, DH, device=DEV) * 0.1).to(dt)
for _ in range(3):
    y = delta_scan_triton(kB, vB, rB, eB, qB, KV, S, KVG, s0=S0B)
    y.float().pow(2).mean().backward()
    for x in (kB, vB, rB, eB, qB): x.grad = None
torch.cuda.synchronize(); t0 = time.time()
for _ in range(10):
    y = delta_scan_triton(kB, vB, rB, eB, qB, KV, S, KVG, s0=S0B)
    y.float().pow(2).mean().backward()
    for x in (kB, vB, rB, eB, qB): x.grad = None
torch.cuda.synchronize()
print("B32 带 s0: %.2f ms/站 (对照无 s0 ~4.9)" % ((time.time() - t0) / 10 * 1000))
print("ALL OK")
