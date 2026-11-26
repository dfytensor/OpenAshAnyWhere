import sys
sys.path.insert(0, r'F:\OpenASH2605')
import torch
import torch.nn.functional as F
from collections import Counter
from train_ab import build

torch.manual_seed(2024)
m = build('v41').cuda()
m.load_state_dict(torch.load(r'F:\OpenASH2605\big_v41\model.pt', map_location='cpu', weights_only=True))
m.eval()

path_flops = 64 * 768 * 2 + 768 * 2
layer_flops = 2304 * 768 * 2 + 3 * 768 * 768 * 2 + 768 * 96 * 2 * 2
print(f'pathway FLOPs share = {path_flops/layer_flops*100:.2f}% per layer')

val = __import__('bench_longrange').val_stream.cuda()
freq = Counter(val[:500000].tolist())
L = 384
win = torch.stack([val[i * 501:i * 501 + L + 1] for i in range(32)]).cuda()
inp, tgt = win[:, :L], win[:, 1:]
trained_gates = [lyr.self_attention_linear.iw_gate.detach().clone() for lyr in m.decoder_layers]

def ce_per_pos(gates):
    with torch.no_grad():
        for lyr, g in zip(m.decoder_layers, gates):
            lyr.self_attention_linear.iw_gate.copy_(g)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg, _ = m(inp)
    lp = F.log_softmax(lg.float()[:, :-1], -1)
    return -lp.gather(-1, tgt[:, :-1].unsqueeze(-1)).squeeze(-1)

ce_trained = ce_per_pos(trained_gates)
ce_pruned = ce_per_pos([torch.zeros_like(g) for g in trained_gates])
ce_on = ce_per_pos([torch.ones_like(g) for g in trained_gates])
delta_tp = ce_pruned - ce_trained            # >0: 剪枝有害
print(f'CE trained={ce_trained.mean().item():.4f} pruned={ce_pruned.mean().item():.4f} all-on={ce_on.mean().item():.4f}')
print(f'prune-vs-trained: mean delta={delta_tp.mean().item():+.4f} | p95 |delta|={delta_tp.abs().quantile(0.95).item():.4f}')

freqs = torch.tensor([[freq.get(t.item(), 0) for t in row] for row in inp[:, :-1]]).cuda()
q = torch.quantile(freqs.float().flatten(), torch.tensor([0.25, 0.75], device='cuda'))
for name, msk in (('low-freq', freqs <= q[0]), ('mid', (freqs > q[0]) & (freqs < q[1])), ('high-freq', freqs >= q[1])):
    d = delta_tp[msk]
    print(f'{name}: n={int(msk.sum()):6d} | prune delta mean={d.mean().item():+.4f} | mean|d|={d.abs().mean().item():.4f}')
