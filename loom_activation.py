# -*- coding: utf-8 -*-
"""loom_activation.py — 预注册判据 3（激活度）：dml 动态路由门的逐 token 标准差 / ‖w_h‖；dmq 的 query 范数量级"""
import sys

import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from train_ab import build

torch.manual_seed(2024)
raw = torch.load('/mnt/f/OpenASH2605/minimind_data/pretrain_cached_1270238_384.pt',
                 map_location='cpu', weights_only=True)
stream = torch.cat([d.long() for d in raw]).cuda()
val = stream[-200_000:]
ids = torch.stack([val[i:i + 385] for i in range(500, 500 + 32 * 397, 397)]).cuda()  # (32, 385)
AC = torch.autocast('cuda', dtype=torch.bfloat16)


def load(arch, path):
    torch.manual_seed(2024)
    m = build(arch)
    m.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=False)
    return m.eval().cuda()


def capture_x(m):
    """抓每层 self_attention_linear 的输入 x"""
    xs = {}

    def pre(mod, args, kwargs):
        x = args[0] if args else kwargs.get('x')
        xs.setdefault(id(mod), []).append(x.detach())
    hs = []
    for layer in m.decoder_layers:
        hs.append(layer.self_attention_linear.register_forward_pre_hook(pre, with_kwargs=True))
    with torch.no_grad():
        with AC:
            m(ids[:, :-1])
    for h in hs:
        h.remove()
    return [xs[id(layer.self_attention_linear)][0] for layer in m.decoder_layers]


def stats(g):
    # g: (B,S,H,1) → 每头 mean/std over tokens
    g = g[..., 0].float()
    return g.mean(dim=(0, 1)), g.std(dim=(0, 1))


# ---------------- dml：动态路由门 ----------------
m = load('v53dml', '/mnt/e/OpenASH_loom/full_v53dml_40k/model.pt')
X = capture_x(m)
print('=== v53dml：路由门（预注册判据 3）===')
print('层 | 静态门 σ(dm_route) 逐头 | 动态门 mean 逐头 | 逐 token std 逐头 | ‖w_h‖ 逐头')
all_std = []
for L, layer in enumerate(m.decoder_layers):
    a = layer.self_attention_linear
    x = X[L].float()
    dyn = torch.sigmoid(a.dm_route.view(1, 1, -1, 1).float()
                        + torch.einsum('bsd,hd->bsh', x, a.dm_route_w.float()).unsqueeze(-1))
    mu, sd = stats(dyn)
    all_std += sd.tolist()
    print('L%d | %s | %s | %s | %s' % (
        L, ' '.join('%.3f' % v for v in torch.sigmoid(a.dm_route).tolist()),
        ' '.join('%.3f' % v for v in mu.tolist()),
        ' '.join('%.4f' % v for v in sd.tolist()),
        ' '.join('%.3f' % v for v in a.dm_route_w.norm(dim=1).tolist())))
print('逐 token std 的全局均值 = %.4f（阈值 1e-3：低于则"端口未激活"）| max = %.4f' % (
    sum(all_std) / len(all_std), max(all_std)))
del m, X
torch.cuda.empty_cache()

# ---------------- dmq：侧记忆 query 的归一化前范数 ----------------
m = load('v53dmq', '/mnt/e/OpenASH_loom/full_v53dmq_40k/model.pt')
X = capture_x(m)
a = m.decoder_layers[0].self_attention_linear
print()
print('=== v53dmq：dm_qnorm=%s；侧记忆 query 归一化前的范数（每头在 token 上的均值/标准差）===' % a.dm_qnorm)
for L, layer in enumerate(m.decoder_layers):
    a = layer.self_attention_linear
    x = X[L].float()
    q = a.dm_q(x).view(x.shape[0], x.shape[1], a.dm_heads, a.dm_dim)
    n = q.norm(dim=-1)                       # (B,S,H)
    print('L%d | mean %s | std %s' % (L, ' '.join('%.2f' % v for v in n.mean(dim=(0, 1)).tolist()),
                                     ' '.join('%.2f' % v for v in n.std(dim=(0, 1)).tolist())))
# 对照：dms@40k 的侧记忆 query 范数
m2 = load('v53dms', '/mnt/f/OpenASH2605/full_v53dms_40k/model.pt')
X2 = capture_x(m2)
print('对照 dms@40k（无归一化）的侧记忆 query 范数：')
for L, layer in enumerate(m2.decoder_layers):
    a = layer.self_attention_linear
    x = X2[L].float()
    q = a.dm_q(x).view(x.shape[0], x.shape[1], a.dm_heads, a.dm_dim)
    n = q.norm(dim=-1)
    print('L%d | mean %s | std %s' % (L, ' '.join('%.2f' % v for v in n.mean(dim=(0, 1)).tolist()),
                                     ' '.join('%.2f' % v for v in n.std(dim=(0, 1)).tolist())))
