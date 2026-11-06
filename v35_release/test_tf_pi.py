"""TF-PI 公平性补强：给 big_tf 的 RoPE 做线性位置插值（PI, scale=4）后重评长度扩展。

PI 是标准外推适配（ bieten et al. 2023）：推理时位置 t → t/s。
对照：big_v35（架构自带长度无关，无任何适配）。
"""
import sys
import math
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
import torch
from open_ash_v3 import OpenASH_V3
from baseline_tf import TransformerLM, rope_cache

DEV = 'cuda'
DATA = r'minimind_data\pretrain_cached_1270238_384.pt'
VOC, HID, HEADS, LAYERS = 23004, 768, 8, 12
SCALE = 4   # 2048 / 512（训练有效位置 ~384）

torch.manual_seed(2024)
raw = torch.load(DATA, map_location='cpu', weights_only=True)
stream = torch.cat([d.long() for d in raw]).cuda()
val_stream = stream[-200_000:]

tf = TransformerLM(VOC, HID, HEADS, 11, max_seq_len=2048).to(DEV)
tf.load_state_dict(torch.load(r'big_tf\model.pt', map_location='cpu', weights_only=True))
tf.eval()
v35 = OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                 use_fused=True, use_decay=True, pre_norm=True, ffn_act='silu',
                 decay_init=(0.0001, 0.35), decay_v_std=0.04,
                 kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True).to(DEV)
v35.load_state_dict(torch.load(r'big_v35\model.pt', map_location='cpu', weights_only=True))
v35.eval()


@torch.no_grad()
def eval_len(m, L, n_win=512, bs=32, pi_scale=1.0):
    n = min(n_win, (val_stream.shape[0] - 1) // (L + 1))
    stride = max(1, (val_stream.shape[0] - L - 1) // n)
    starts = torch.arange(0, n, device=DEV) * stride
    # 每次按 (L, pi_scale) 重建 rope 表，避免分支间状态污染
    inv = 1.0 / (10000.0 ** (torch.arange(0, HID // HEADS, 2, device=DEV).float() / (HID // HEADS)))
    t = (torch.arange(0, L, device=DEV).float() / pi_scale)
    fr = torch.outer(t, inv)
    tf.rope_cos, tf.rope_sin = torch.cos(fr)[None, None], torch.sin(fr)[None, None]
    tot, cnt = 0.0, 0
    for s in range(0, n, bs):
        idx = starts[s:s + bs].tolist()
        win = torch.stack([val_stream[i:i + L + 1] for i in idx])
        ids, tgt = win[:, :L], win[:, 1:]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = m.forward_loss(ids, tgt, fused_ce=True)
        tot += loss.item() * ids.numel()
        cnt += ids.numel()
    return tot / cnt


print(f'{"L":>6} {"tf-raw":>10} {"tf-PI x4":>10} {"v35":>10}  {"v35 vs tf-PI"}')
for L in (512, 1024, 2048):
    ce_v = eval_len(v35, L)
    ce_t = eval_len(tf, L)
    ce_tp = eval_len(tf, L, pi_scale=SCALE)
    p = lambda c: math.exp(min(c, 20))
    print(f'{L:>6} {p(ce_t):>10.2f} {p(ce_tp):>10.2f} {p(ce_v):>10.2f}  '
          f'{"v35" if ce_v < ce_tp else "tf-PI"} ({abs(1 - ce_v/ce_tp)*100:.1f}%)')
