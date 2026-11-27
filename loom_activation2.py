# -*- coding: utf-8 -*-
"""loom_activation2.py — 动态路由门在 dml@40k 与 dml@80k 的统计对比（小数据文件，显存上限 8%）"""
import sys

import torch

torch.cuda.set_per_process_memory_fraction(0.08)
sys.path.insert(0, r'/mnt/f/OpenASH2605')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from train_ab import build

torch.manual_seed(2024)
raw = torch.load('/mnt/f/OpenASH2605/minimind_data/pretrain_cached_30000_384.pt',
                 map_location='cpu', weights_only=True)
stream = torch.cat([d.long() for d in raw])
print('small data stream tokens:', stream.shape[0])
ids = torch.stack([stream[i:i + 385] for i in range(1000, 1000 + 16 * 997, 997)]).cuda()  # (16, 385)
AC = torch.autocast('cuda', dtype=torch.bfloat16)


def gate_stats(path):
    torch.manual_seed(2024)
    m = build('v53dml')
    m.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=False)
    m.eval().cuda()
    X = [[] for _ in m.decoder_layers]

    def pre(mod, args, kwargs):
        x = args[0] if args else kwargs.get('x')
        pre.cur.append(x.detach())
    pre.cur = None
    hs = [l.self_attention_linear.register_forward_pre_hook(pre, with_kwargs=True) for l in m.decoder_layers]
    with torch.no_grad():
        for a in range(0, ids.shape[0], 4):          # 每批 ≤2048 token（本进程显存上限 8%）
            pre.cur = []
            with AC:
                m(ids[a:a + 4, :-1])
            for L in range(len(X)):
                X[L].append(pre.cur[L])
    for h in hs:
        h.remove()
    X = [torch.cat(xs, 0) for xs in X]
    rows = []
    for L, layer in enumerate(m.decoder_layers):
        a = layer.self_attention_linear
        x = X[L].float()
        dyn = torch.sigmoid(a.dm_route.view(1, 1, -1, 1).float()
                            + torch.einsum('bsd,hd->bsh', x, a.dm_route_w.float()).unsqueeze(-1))[..., 0]
        s0 = torch.sigmoid(a.dm_route)
        pre = torch.einsum('bsd,hd->bsh', x, a.dm_route_w.float())          # 门控 logit 的动态项
        rows.append(dict(
            static=s0.mean().item(), dyn_mean=dyn.mean().item(), dyn_std=dyn.std(dim=(0, 1)).mean().item(),
            closed=(dyn < 0.10).float().mean().item(), open=(dyn > 0.90).float().mean().item(),
            wnorm=a.dm_route_w.norm(dim=1).mean().item(), pre_std=pre.std(dim=(0, 1)).mean().item(),
            head_min=dyn.mean(dim=(0, 1)).min().item(), head_max=dyn.mean(dim=(0, 1)).max().item()))
    del m
    torch.cuda.empty_cache()
    return rows


for tag, path in (('dml@40k', '/mnt/e/OpenASH_loom/full_v53dml_40k/model.pt'),
                  ('dml@80k', '/mnt/e/OpenASH_loom/full_v53dml_80k/model.pt')):
    rows = gate_stats(path)
    print()
    print('=== %s ===' % tag)
    print('层 | 静态门均值 | 动态门均值 | 逐token std | 关(<0.10) | 开(>0.90) | 动态项std | ‖w_h‖ | 头均值范围')
    for L, r in enumerate(rows):
        print('L%d |   %.3f    |   %.3f    |   %.3f    |  %.1f%%   |  %.1f%%   |  %.2f   | %.2f | %.2f–%.2f' % (
            L, r['static'], r['dyn_mean'], r['dyn_std'], r['closed'] * 100, r['open'] * 100,
            r['pre_std'], r['wnorm'], r['head_min'], r['head_max']))
    print('均值: 动态门 %.3f | std %.3f | 关 %.1f%% | 开 %.1f%% | ‖w_h‖ %.2f' % (
        sum(r['dyn_mean'] for r in rows) / 7, sum(r['dyn_std'] for r in rows) / 7,
        sum(r['closed'] for r in rows) / 7 * 100, sum(r['open'] for r in rows) / 7 * 100,
        sum(r['wnorm'] for r in rows) / 7))
