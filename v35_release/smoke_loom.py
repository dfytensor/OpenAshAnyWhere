# -*- coding: utf-8 -*-
"""smoke_loom.py — v53dml / v53dmq 冒烟
1) 参数与零初始化；2) dml 恒等起点（载入 dms@40k 权重后与 dms 逐位等价）；3) 梯度流；
4) dml 非零路由下 prefill vs 逐 token 解码一致性（验证解码侧动态路由实现）；
5) dmq 前后向 + 解码一致性"""
import sys

import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from train_ab import build

torch.manual_seed(2024)
raw = torch.load('/mnt/f/OpenASH2605/minimind_data/pretrain_cached_1270238_384.pt',
                 map_location='cpu', weights_only=True)
stream = torch.cat([d.long() for d in raw]).cuda()
ids = stream[1000:1264].view(1, -1)
AC = torch.autocast('cuda', dtype=torch.bfloat16)
sd_dms = torch.load('/mnt/f/OpenASH2605/full_v53dms_40k/model.pt', map_location='cpu', weights_only=True)


def load(arch, sd):
    torch.manual_seed(2024)
    m = build(arch)
    ret = m.load_state_dict(sd, strict=False)
    return m.eval().cuda(), ret


@torch.no_grad()
def logits(m, x, state=None, pos=0):
    with AC:
        lg, st = m(x, state=state, pos_offset=pos)
    return lg.float()[0], st


@torch.no_grad()
def consistency(m, T=256, P=128):
    full, _ = logits(m, ids[:, :T])
    _, st = logits(m, ids[:, :P])
    inc = []
    for t in range(P, T):
        lg, st = logits(m, ids[:, t:t + 1], state=st, pos=t)
        inc.append(lg[0])
    inc = torch.stack(inc, 0)
    A = full[P:]
    return (A - inc).abs().mean().item(), (A.argmax(-1) == inc.argmax(-1)).float().mean().item() * 100


# ---------- v53dml ----------
m, _ = load('v53dml', sd_dms)
n = sum(p.numel() for p in m.parameters())
rw = [p for nm, p in m.named_parameters() if nm.endswith('dm_route_w')]
print('v53dml params: %d (+%d vs dms) | dm_route_w tensors: %d, all zero: %s' % (
    n, n - 105197148, len(rw), all(bool((p == 0).all()) for p in rw)))
md, _ = load('v53dms', sd_dms)
with torch.no_grad():
    a, _ = logits(m, ids)
    b, _ = logits(md, ids)
print('2) 恒等起点: dml(载 dms 权重, route_w=0) vs dms logits |Δ|max = %g → %s' % (
    (a - b).abs().max().item(), '逐位等价 ✓' if torch.equal(a, b) else '有差异 ✗'))
x = torch.randint(0, 23000, (2, 64), device='cuda')
with AC:
    loss, _ = m.forward_loss(x, x, fused_ce=True)
loss.backward()
print('3) dm_route_w grad |mean| per layer:', ' '.join('%.1e' % p.grad.abs().mean().item() for p in rw))
with torch.no_grad():
    g = torch.Generator(device='cuda').manual_seed(7)
    for p in rw:
        p.copy_(torch.randn(p.shape, generator=g, device='cuda') * 0.3)
dmn, ag = consistency(m)
print('4) dml 非零路由: prefill vs 逐 token 解码 logits|Δ| mean %.4f | argmax 一致 %.2f%%' % (dmn, ag))
with torch.no_grad():
    for p in rw:
        p.zero_()
dmn0, ag0 = consistency(m)
print('   dml 零路由  : logits|Δ| mean %.4f | argmax 一致 %.2f%%（= dms 水平）' % (dmn0, ag0))
del m, md
torch.cuda.empty_cache()

# ---------- v53dmq ----------
mq, ret = load('v53dmq', sd_dms)
nq = sum(p.numel() for p in mq.parameters())
a0 = mq.decoder_layers[0].self_attention_linear
print('v53dmq params: %d | dm_qnorm=%s dm_dynroute=%s | load missing=%d unexpected=%d' % (
    nq, a0.dm_qnorm, a0.dm_dynroute, len(ret.missing_keys), len(ret.unexpected_keys)))
x = torch.randint(0, 23000, (2, 64), device='cuda')
with AC:
    loss, _ = mq.forward_loss(x, x, fused_ce=True)
loss.backward()
gq = mq.decoder_layers[0].self_attention_linear.dm_q.weight.grad
print('   loss %.3f | dm_q.weight grad |mean| %.2e' % (loss.item(), gq.abs().mean().item()))
dmn, ag = consistency(mq)
print('5) dmq: prefill vs 逐 token 解码 logits|Δ| mean %.4f | argmax 一致 %.2f%%' % (dmn, ag))
print('LOOM SMOKE OK')
