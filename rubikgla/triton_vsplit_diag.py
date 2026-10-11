import sys, torch
sys.path.insert(0, r"F:\夸克\rubikgla")
import delta_triton as DT

DEV = "cuda"
torch.manual_seed(0)
B, L, KV, SLOTS, KVG, DH = 2, 16, 5, 2, 2, 32
G = KV * SLOTS
H = KV * KVG
dt = torch.bfloat16

k = (torch.randn(B, L, G, DH, device=DEV) * 0.3).to(dt).requires_grad_(True)
v = (torch.randn(B, L, G, DH, device=DEV) * 0.3).to(dt).requires_grad_(True)
r = (torch.rand(B, L, G, DH, device=DEV) * 0.3 + 1e-3).to(dt).requires_grad_(True)
e = (torch.rand(B, L, G, DH, device=DEV) * 0.3 + 0.05).to(dt).requires_grad_(True)
q = (torch.randn(B, L, H, DH, device=DEV) * 0.5).to(dt).requires_grad_(True)
dy = None

grads = {}
for vs in (1, 2):
    DT.VSPLIT = vs
    for x in (k, v, r, e, q):
        x.grad = None
    y = DT.delta_scan_triton(k, v, r, e, q, KV, SLOTS, KVG)
    if dy is None:
        dy = torch.randn_like(y)
    y.backward(dy)
    grads[vs] = {n: g.detach().float().clone() for n, g in [("dk", k.grad), ("dv", v.grad), ("dr", r.grad), ("de", e.grad), ("dq", q.grad)]}

for name in ("dk", "dv", "dr", "de", "dq"):
    a1, a2 = grads[1][name], grads[2][name]
    d = (a1 - a2).abs()
    rel = d.max().item() / (a1.abs().max().item() + 1e-9)
    print("%-4s max|d1-d2|=%.3e  relmax=%.3f  d1_absmax=%.2e d2_absmax=%.2e"
          % (name, d.max().item(), rel, a1.abs().max().item(), a2.abs().max().item()))
    if rel > 1e-2:
        idx = d.argmax().item()
        un = torch.unravel_index(torch.tensor(idx), a1.shape)
        print("      worst @ %s: d1=%.4e d2=%.4e ratio=%.3f"
              % (tuple(x.item() for x in un), a1.flatten()[idx].item(), a2.flatten()[idx].item(),
                 a2.flatten()[idx].item() / (a1.flatten()[idx].item() + 1e-12)))
        # 差异面统计: 哪些维出错
        per_bad = (d > 1e-2 * a1.abs().max()).float()
        print("      bad%% by axis dims: b=%.2f L=%.2f G=%.2f D=%.2f"
              % (per_bad.mean(dim=(1, 2, 3)).mean().item(),
                 per_bad.mean(dim=(0, 2, 3)).mean().item(),
                 per_bad.mean(dim=(0, 1, 3)).mean().item(),
                 per_bad.mean(dim=(0, 1, 2)).mean().item()))
