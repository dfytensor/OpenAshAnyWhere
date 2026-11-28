# -*- coding: utf-8 -*-
"""add_loomport.py — v5.3DML（Loom 读端口移植：侧记忆→主头的逐 token 动态路由）
                  + v5.3DMQ（侧记忆 query L2 归一化，与 k 同口径）
prefill 与解码两条路径同步；旗标关闭时逐位不变。先 ast.parse 再写盘。"""
import ast
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
NL = chr(10)


def lead_of_line(src, needle, occ=1):
    """needle 第 occ 次出现所在行的前导空白"""
    i = -1
    for _ in range(occ):
        i = src.find(needle, i + 1)
    j = src.rfind(NL, 0, i)
    k = j + 1
    while src[k] == ' ':
        k += 1
    return src[j + 1:k]


def patch(path, edits):
    src = open(path, encoding='utf-8').read()
    for old, new in edits:
        assert src.count(old) == 1, '%s: anchor count=%d for %r' % (path.split(chr(92))[-1], src.count(old), old[:70])
        src = src.replace(old, new, 1)
    ast.parse(src)
    open(path, 'w', encoding='utf-8', newline='').write(src)
    print('patched', path.split(chr(92))[-1], '(%d edits, parsed OK before write)' % len(edits))


P = r'F:\OpenASH2605\open_ash_v3.py'
src = open(P, encoding='utf-8').read()

# —— 用实际缩进推导传递点的锚点 ——
L1 = lead_of_line(src, 'dm_gate=dm_gate)', occ=1)     # DecoderLayer -> MaxStateSuper
L2 = lead_of_line(src, 'dm_gate=dm_gate)', occ=2)     # OpenASH_V3 -> DecoderLayer
assert L1 != L2, (L1, L2)

ROUTE_STATIC_PF = '                    d_route = torch.sigmoid(self.dm_route).view(1, 1, -1, 1).to(d_out.dtype)'
ROUTE_STATIC_DC = '                    d_out_dec = d_out_dec * torch.sigmoid(self.dm_route).view(1, 1, -1, 1).to(d_out_dec.dtype)'

edits = [
    # (a) MaxStateSuper 签名
    ('                 kv_decay_grad=False, dm_gate=False):' + NL + '        super(MaxStateSuper, self).__init__()',
     '                 kv_decay_grad=False, dm_gate=False,' + NL +
     '                 dm_dynroute=False, dm_qnorm=False):' + NL + '        super(MaxStateSuper, self).__init__()'),
    # (b) 参数
    ('                    self.dm_route = nn.Parameter(torch.zeros(self.heads))  # 逐头路由门',
     '                    self.dm_route = nn.Parameter(torch.zeros(self.heads))  # 逐头路由门' + NL +
     '                    # v5.3DML（Loom 读端口移植）：侧记忆→主头的读端口由静态逐头标量改为逐 token 门控' + NL +
     '                    #        route = σ(dm_route_h + x·w_h)；w 零初始化 ⇒ 起点与静态路由逐位等价' + NL +
     '                    self.dm_dynroute = dm_dynroute' + NL +
     '                    if dm_dynroute:' + NL +
     '                        self.dm_route_w = nn.Parameter(torch.zeros(self.heads, dim_size))' + NL +
     '                    # v5.3DMQ：侧记忆 query L2 归一化（GDN 的 use_qk_l2norm_in_kernel 语义；与 k 同口径）' + NL +
     '                    self.dm_qnorm = dm_qnorm'),
    # (c) prefill q
    ('                    d_q = self.dm_q(x).view(bx, sx, self.dm_heads, self.dm_dim).to(torch.bfloat16)',
     '                    d_q = self.dm_q(x).view(bx, sx, self.dm_heads, self.dm_dim).to(torch.bfloat16)' + NL +
     '                    if self.dm_qnorm:' + NL +
     '                        d_q = (d_q / (d_q.norm(dim=-1, keepdim=True) + 1e-6)).to(torch.bfloat16)'),
    # (d) prefill route
    (ROUTE_STATIC_PF + NL + '                    d_out = d_out * d_route',
     '                    if self.dm_dynroute:' + NL +
     '                        # v5.3DML: 逐 token 读端口门控' + NL +
     '                        d_route = torch.sigmoid(self.dm_route.view(1, 1, -1, 1).float()' + NL +
     '                                                + torch.einsum(\'bsd,hd->bsh\', x.float(),' + NL +
     '                                                               self.dm_route_w.float()).unsqueeze(-1)' + NL +
     '                                                ).to(d_out.dtype)' + NL +
     '                    else:' + NL +
     '                        ' + ROUTE_STATIC_PF.strip() + NL +
     '                    d_out = d_out * d_route'),
    # (e) decode q
    ('                    d_q1 = self.dm_q(x).view(b, 1, self.dm_heads, self.dm_dim).to(torch.bfloat16)',
     '                    d_q1 = self.dm_q(x).view(b, 1, self.dm_heads, self.dm_dim).to(torch.bfloat16)' + NL +
     '                    if self.dm_qnorm:' + NL +
     '                        d_q1 = (d_q1 / (d_q1.norm(dim=-1, keepdim=True) + 1e-6)).to(torch.bfloat16)'),
    # (f) decode route
    (ROUTE_STATIC_DC,
     '                    if self.dm_dynroute:' + NL +
     '                        d_route = torch.sigmoid(self.dm_route.view(1, 1, -1, 1).float()' + NL +
     '                                                + torch.einsum(\'bsd,hd->bsh\', x.float(),' + NL +
     '                                                               self.dm_route_w.float()).unsqueeze(-1)' + NL +
     '                                                ).to(d_out_dec.dtype)' + NL +
     '                    else:' + NL +
     '                        d_route = torch.sigmoid(self.dm_route).view(1, 1, -1, 1).to(d_out_dec.dtype)' + NL +
     '                    d_out_dec = d_out_dec * d_route'),
    # (g) DecoderLayer 签名
    ('                 kv_decay_grad=False, dm_gate=False):' + NL + '        super(DecoderLayer, self).__init__()',
     '                 kv_decay_grad=False, dm_gate=False,' + NL +
     '                 dm_dynroute=False, dm_qnorm=False):' + NL + '        super(DecoderLayer, self).__init__()'),
    # (h) DecoderLayer -> MaxStateSuper
    (L1 + 'dm_gate=dm_gate)',
     L1 + 'dm_gate=dm_gate,' + NL + L1 + 'dm_dynroute=dm_dynroute,' + NL + L1 + 'dm_qnorm=dm_qnorm)'),
    # (i) OpenASH_V3 签名
    ('kv_gamma_init=1.0, pos_mod=False, kv_decay_grad=False,' + NL + '                 dm_gate=False):',
     'kv_gamma_init=1.0, pos_mod=False, kv_decay_grad=False,' + NL +
     '                 dm_gate=False, dm_dynroute=False, dm_qnorm=False):'),
    # (j) OpenASH_V3 -> DecoderLayer
    (L2 + 'dm_gate=dm_gate)',
     L2 + 'dm_gate=dm_gate,' + NL + L2 + 'dm_dynroute=dm_dynroute,' + NL + L2 + 'dm_qnorm=dm_qnorm)'),
]
patch(P, edits)

# train_ab 注册表
P2 = r'F:\OpenASH2605\train_ab.py'
s2 = open(P2, encoding='utf-8').read()
block_dms_start = s2.index("    if arch == 'v53dms':")
block_dms_end = s2.index("    if arch == 'v53dmh':")
dms_block = s2[block_dms_start:block_dms_end]
assert dms_block.count('dm_gate=True)') == 1
lines_b = dms_block.split(NL)
assert lines_b[0] == "    if arch == 'v53dms':" and lines_b[1].lstrip().startswith('# v5.3DMS'), lines_b[:2]
b1_lines = list(lines_b)
b1_lines[0] = "    if arch == 'v53dml':"
b1_lines[1] = ('        # v5.3DML: dms + Loom 读端口移植——侧记忆→主头的读端口由静态逐头标量改为逐 token 门控')
b1_lines[2] = ('        #        route = σ(dm_route_h + x·w_h)，w 零初始化（起点 = dms 逐位等价）；+43K 参数')
b1_lines[3] = ('        #        依据：LoomFormer 消融中"读端口动态化"是最重要因素（去掉后只保留 75.4%）')
b2_lines = list(lines_b)
b2_lines[0] = "    if arch == 'v53dmq':"
b2_lines[1] = ('        # v5.3DMQ: dms + 侧记忆 query L2 归一化（与 k 同口径；GDN 的 l2norm 语义）——单旋钮对照 dms@40k')
b1 = NL.join(b1_lines)
b2 = NL.join(b2_lines)
assert b1.count('                          dm_gate=True)') == 1 and b2.count('                          dm_gate=True)') == 1
b1 = b1.replace('                          dm_gate=True)', '                          dm_gate=True, dm_dynroute=True)', 1)
b2 = b2.replace('                          dm_gate=True)', '                          dm_gate=True, dm_qnorm=True)', 1)
assert b1 != dms_block and b2 != dms_block
s2 = s2[:block_dms_end] + b1 + b2 + s2[block_dms_end:]
ast.parse(s2)
open(P2, 'w', encoding='utf-8', newline='').write(s2)
print('train_ab.py: v53dml / v53dmq registered (parsed OK before write)')

# bench：E: 盘 checkpoint 根目录 + 两个新 arch 的映射
P3 = r'F:\OpenASH2605\bench_longrange.py'
s3 = open(P3, encoding='utf-8').read()
old = "        path = os.path.join(key, 'model.pt')"
assert s3.count(old) == 1
s3 = s3.replace(old, "        path = os.path.join(os.environ.get('CKPT_ROOT', '.'), key, 'model.pt')", 1)
old = "('full_v53dms_40k', 'v53dms'), ('full_v53dms_80k', 'v53dms'),"
assert s3.count(old) == 1
s3 = s3.replace(old, old + NL + "                    ('full_v53dml_40k', 'v53dml'), ('full_v53dmq_40k', 'v53dmq'),", 1)
ast.parse(s3)
open(P3, 'w', encoding='utf-8', newline='').write(s3)
print('bench_longrange.py: CKPT_ROOT env + v53dml/v53dmq mapping (parsed OK before write)')
