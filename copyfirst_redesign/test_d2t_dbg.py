import sys, torch
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import deepconv_triton as M

torch.manual_seed(7)
b, s, h, c, w, k, p = 1, 2, 8, 2, 8, 3, 1
x = (torch.randn(b, s, h, device='cuda') * 0.3).requires_grad_(True)
win = (torch.randn(w, device='cuda') * 0.2).requires_grad_(True)
wo = (torch.randn(w, device='cuda') * 0.2).requires_grad_(True)
k1 = (torch.randn(c, 1, k, k, device='cuda') * 0.2).requires_grad_(True)
k2 = (torch.randn(1, c, k, k, device='cuda') * 0.2).requires_grad_(True)
bias1 = (torch.randn(c, device='cuda') * 0.1).requires_grad_(True)

y = M._ConvD2Fn.apply(x, win, wo, k1, k2, bias1)
print('kernel out[0,0,:4]:', y[0, 0, :4].tolist())

# 手工参考 (与 kernel 相同公式逐步)
xp = torch.nn.functional.pad(x, (p, p), mode='replicate')
win_pad = torch.nn.functional.pad(win, (p, p))
wo_pad = torch.nn.functional.pad(wo, (p, p))
W1 = torch.zeros(k, c, w, device='cuda')
W2 = torch.zeros(k, c, w, device='cuda')
for kk_ in range(k):
    for j in range(k):
        v = j - p
        W1[kk_] += k1[:, 0, kk_, j][:, None] * torch.roll(win_pad, v)[:w][None, :]
        W2[kk_] += k2[0, :, kk_, j][:, None] * torch.roll(wo_pad, -v)[:w][None, :]

# 手工 stage1
T1 = torch.zeros(b, s, c, h, w, device='cuda')
for kk_ in range(k):
    T1 += xp[:, :, None, kk_:kk_ + h, None] * W1[kk_][None, None, :, None, :]
T1 = torch.relu(T1 + bias1.view(1, 1, c, 1, 1))
# 手工 stage2 (h zero-pad)
out = torch.zeros(b, s, h, device='cuda')
for kk_ in range(k):
    u = kk_ - p
    hu = torch.arange(h, device='cuda') + u
    m = (hu >= 0) & (hu < h)
    T1h = torch.zeros(b, s, c, h, w, device='cuda')
    idx = hu.clamp(0, h - 1)
    T1h = T1[:, :, :, idx, :] * m.view(1, 1, 1, h, 1)
    out += (T1h * W2[kk_][None, None, :, None, :]).sum((2, 4))
print('manual out[0,0,:4]:', out[0, 0, :4].tolist())

# 测试文件的参考 (test_d2t.py 的 ref)
T1r = torch.zeros(b, s, c, h, w, device='cuda')
for i in range(k):
    for j in range(k):
        v = j - p
        T1r += k1[:, 0, i, j].view(1, 1, c, 1, 1) * (
            xp[:, :, None, i:i + h, None].float() * torch.roll(win_pad, v)[:w][None, None, None, None, :].float())
T1r = torch.relu(T1r + bias1.view(1, 1, c, 1, 1))
T1wp = torch.nn.functional.pad(T1r, (p, p, p, p, 0, 0, 0, 0, 0, 0))
outr = torch.zeros(b, s, h, device='cuda')
for i in range(k):
    for j in range(k):
        outr += torch.einsum('bschw,cw->bsh', T1wp[:, :, :, i:i + h, j:j + w],
                             k2[0, :, i, j][:, None] * wo[None, :])
print('test-ref out[0,0,:4]:', outr[0, 0, :4].tolist())
print('manual vs test-ref:', (out - outr).abs().max().item())

# 手算单点核对: 逐 tap 对比
for i in range(k):
    for j in range(k):
        term_ref = torch.einsum('bschw,cw->bsh', T1wp[:, :, :, i:i + h, j:j + w],
                                k2[0, :, i, j][:, None] * wo[None, :])
        u = i - p
        hu = torch.arange(h, device='cuda') + u
        m = (hu >= 0) & (hu < h)
        T1h = T1[:, :, :, hu.clamp(0, h - 1), :] * m.view(1, 1, 1, h, 1)
        v = j - p
        term_man = (T1h * (k2[0, :, i, j][:, None] * torch.roll(wo_pad, -v)[:w][None, :])[None, None, :, None, :]).sum((2, 4))
        print('tap i=%d j=%d: ref=%.6f man=%.6f diff=%.2e' %
              (i, j, term_ref[0, 0, 1].item(), term_man[0, 0, 1].item(),
               (term_ref[0, 0] - term_man[0, 0]).abs().max().item()))

# 逐项: 检查 stage1
T1_alt = torch.zeros(b, s, c, h, w, device='cuda')
for i in range(k):
    for j in range(k):
        v = j - p
        T1_alt += k1[:, 0, i, j].view(1, 1, c, 1, 1) * (
            xp[:, :, None, i:i + h, None] * torch.roll(win_pad, v)[:w][None, None, None, None, :])
T1_alt = torch.relu(T1_alt + bias1.view(1, 1, c, 1, 1))
print('T1 vs T1_alt err:', (T1 - T1_alt).abs().max().item())
