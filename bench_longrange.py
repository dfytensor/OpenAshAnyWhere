"""v3.1 长程能力测试：v3.0 pure-cummax vs v3.1 decay-max（各 2500 步 checkpoint）。

  A. 上下文长度扩展 PPL：L ∈ {64..2048}，长度越长信息越多，观察 loss 是否持续下降
     （>512 时位置编码 clamp(511) 饱和，属长度外推区）
  B. 远程复制召回：[prefix][span A][gap G][span A 重复]，度量"重复段 CE 相对新片段 CE 的下降"
     —— 直接探针远程信息在状态中的存活与利用，gap 即受控记忆距离
  C. 分位置 loss 曲线（L=384）：位置越靠后可用上下文越长，曲线斜率 = 上下文利用率
  D. v3.1 学习到的 γ 有效记忆长度（诊断：模型实际学到多长的衰减尺度）
"""
import json
import math
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn.functional as F

sys.path.insert(0, r'F:\OpenASH2605')
from open_ash_v3 import OpenASH_V3

DEV = 'cuda'
DATA = r'F:\OpenASH2605\minimind_data\pretrain_cached_1270238_384.pt'   # 与 big_* 训练同源，尾部 200k 为 held-out
CKPTS = {'v3.0 pure': (r'F:\OpenASH2605\minimind_v30_pure\v3_minimind.pt', False),
         'v3.1 decay': (r'F:\OpenASH2605\minimind_v31_decay\v3_minimind.pt', True)}
VOC, HID, HEADS, LAYERS = 23004, 768, 8, 12

torch.manual_seed(2024)
raw = torch.load(DATA, map_location='cpu', weights_only=True)
stream = torch.cat([d.long() for d in raw]).cuda()
val_stream = stream[-200_000:]
print(f'val tokens: {val_stream.shape[0]:,}')

results = {}


def load_model(key):
    """按 key 构建并加载模型。支持 CKPTS 旧条目与全部 big_*/ab_*/scale_* 结果目录。"""
    import os
    torch.manual_seed(2024)
    m = None
    wrapper = False
    if key in CKPTS:
        path, ud = CKPTS[key]
        m = OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=512, model_flag='train',
                       use_fused=True, use_decay=ud)
    else:
        path = os.path.join(key, 'model.pt')
        # v4.2+ 全部 scale_* 变体：直接走 train_ab.build（训练配置的唯一真相源）。
        # 加载诊断：strict=False 但打印 missing/unexpected——配置漂移必须可见，不允许静默错探。
        from train_ab import build as tab_build
        ARCH_MAP = [('scale_v37_15k', 'v37'), ('scale_v42_15k', 'v42'),
                    ('scale_v43_15k', 'v43'), ('scale_v44_15k', 'v44'),
                    ('scale_v45_15k', 'v45'), ('scale_v47_15k', 'v47'),
                    ('scale_v48_15k', 'v48'), ('scale_v49_15k', 'v49'),
                    ('scale_v50_15k', 'v50'), ('scale_v52_15k', 'v52'),
                    ('scale_v53_15k', 'v53'), ('scale_v54_15k', 'v54'),
                    # 归档的误跑目录（见 REPORT §14 复盘）
                    ('scale_v52lite_15k', 'v48'), ('scale_v53onv48_15k', 'v48b'),
                    # full epoch 终局
                    ('full_v53_40k', 'v53'),
                    # MEM 永久组增量隔离
                    ('scale_v53m_15k', 'v53m')]
        arch = None
        for pre, a in ARCH_MAP:
            if key.startswith(pre):
                arch = a
                break
        if arch is not None:
            torch.manual_seed(2024)
            m = tab_build(arch)
            sd = torch.load(path, map_location='cpu', weights_only=True)
            ret = m.load_state_dict(sd, strict=False)
            n_miss = len(ret.missing_keys)
            n_unexp = len(ret.unexpected_keys)
            n_par = sum(p.numel() for p in m.parameters())
            print(f'[load {key}] arch={arch} params={n_par:,} '
                  f'missing={n_miss} unexpected={n_unexp}')
            if n_miss or n_unexp:
                print(f'  !! CONFIG DRIFT: missing[:3]={ret.missing_keys[:3]} '
                      f'unexpected[:3]={ret.unexpected_keys[:3]}')
            m.eval()
            return m.cuda()
        # FLA 官方基线（wrapper 保存，键带 model. 前缀）
        if key.startswith('big_gdn') or key.startswith('scale_gdn') \
                or key.startswith('full_gdn'):
            from baseline_fla import build_gdn, FlaLMWrapper
            m = FlaLMWrapper(build_gdn())
            wrapper = True
        elif key.startswith('big_dn'):
            from baseline_fla import build_deltanet, FlaLMWrapper
            m = FlaLMWrapper(build_deltanet())
            wrapper = True
        elif key.startswith('big_tf') or key.startswith('ab_tf') or key.startswith('scale_tf') \
                or key.startswith('full_tf'):
            from baseline_tf import TransformerLM
            m = TransformerLM(VOC, HID, HEADS, 11, max_seq_len=2048)
            wrapper = True
        # v3 系（OpenASH_V3）
        else:
            ms = 2048
            backend = 'sum'
            beta_init = 0.0
            if key.startswith('big_v36') or key.startswith('scale_v36') \
                    or key.startswith('big_v36b1') or key.startswith('big_v37') \
                    or key.startswith('scale_v37') or key.startswith('full_v37'):
                backend = 'delta_fla'
            if key.endswith('b1'):
                beta_init = 4.0
            m = OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=ms, model_flag='train',
                           use_fused=True, use_decay=True,
                           pre_norm=(key not in ('ab_v30', 'ab_v31', 'ab_v32')),
                           ffn_act='silu' if key not in ('ab_v30', 'ab_v31', 'ab_v32') else 'relu',
                           decay_init=(0.0001, 0.35) if key not in ('ab_v30', 'ab_v31') else (0.001, 0.15),
                           decay_v_std=0.04 if key not in ('ab_v30', 'ab_v31') else 0.0,
                           kv_channel=(key not in ('ab_v30', 'ab_v31', 'ab_v32')),
                           kv_decay=0.98,
                           kv_norm_keys=(key not in ('ab_v30', 'ab_v31', 'ab_v32', 'ab_v33')),
                           kv_beta=(key not in ('ab_v30', 'ab_v31', 'ab_v32', 'ab_v33')),
                           kv_delta=('v35d' in key),
                           kv_dim=(48 if key.startswith('big_v36') or key.startswith('scale_v36')
                                   or key.startswith('big_v36b1') or key.startswith('big_v37')
                                   or key.startswith('scale_v37') or key.startswith('full_v37') else None),
                           kv_backend=backend, kv_beta_init=beta_init, short_conv=('v37' in key))
    sd = torch.load(os.path.join(key, 'model.pt') if key not in CKPTS
                    else CKPTS[key][0], map_location='cpu', weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    return m.cuda()



@torch.no_grad()
def eval_len(m, L, n_win=512, bs=32):
    n = min(n_win, (val_stream.shape[0] - 1) // (L + 1))
    stride = max(1, (val_stream.shape[0] - L - 1) // n)
    starts = torch.arange(0, n, device=DEV) * stride
    tot, cnt = 0.0, 0
    for s in range(0, n, bs):
        idx = starts[s:s + bs]
        win = torch.stack([val_stream[i:i + L + 1] for i in idx.tolist()])
        ids, tgt = win[:, :L], win[:, 1:]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = m.forward_loss(ids, tgt, fused_ce=True)
        tot += loss.item() * ids.numel()
        cnt += ids.numel()
    ce = tot / cnt
    return ce, math.exp(min(ce, 20))


@torch.no_grad()
def per_token_ce(m, ids):
    """逐位置 CE，返回 (B, S-1) fp32。"""
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits, _ = m(ids)
    tgt = ids[:, 1:]
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    ce = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return ce


# ============================ Test A: 长度扩展 PPL ============================
print('\n===== A. 上下文长度扩展 PPL =====')
MODEL_KEYS = ['scale_v49_15k', 'scale_v53_15k', 'scale_v53m_15k', 'scale_v54_15k']
models = {k: load_model(k) for k in MODEL_KEYS}
wrapper_keys = {k for k in MODEL_KEYS if not hasattr(models[k], 'decoder_layers')}
lengths = [64, 128, 256, 384, 512, 1024, 2048]
results['A_len_ppl'] = {}
hdr = 'L'.rjust(6) + ''.join(k.rjust(14) for k in models)
print(hdr)
for L in lengths:
    row = f'{str(L):>6}'
    for k, m in models.items():
        ce, ppl = eval_len(m, L)
        results['A_len_ppl'].setdefault(k, []).append((L, ce, ppl))
        row += f'  {ppl:9.2f}ppl'
    print(row)
print('（>512 为长度外推区：位置编码 clamp(511) 饱和，测 scan 状态本身的保持能力）')

# ==================== Test B: 远程复制召回（受控距离） ====================
print('\n===== B. 远程复制召回（重复 span CE 下降 = 召回强度） =====')
A_SPAN, PREFIX = 24, 64
gaps = [32, 64, 128, 256, 384]
BS, NBATCH = 16, 6
results['B_recall'] = {}
gen = torch.Generator(device='cpu').manual_seed(99)


@torch.no_grad()
def recall_ce(m, gap):
    """返回 (重复段 CE, 新片段 CE)：同批位置，tail 为重复 span vs 语料新片段。"""
    seq_len = PREFIX + A_SPAN + gap + A_SPAN
    ce_r_sum, ce_f_sum, n = 0.0, 0.0, 0
    lo = PREFIX + A_SPAN + gap - 1          # A2 段在 targets 坐标的起点
    hi = PREFIX + 2 * A_SPAN + gap - 1
    for _ in range(NBATCH):
        starts = torch.randint(0, stream.shape[0] - seq_len - 1, (BS,), generator=gen)
        rep, fre = [], []
        for i in starts.tolist():
            pre = stream[i:i + PREFIX]
            A = stream[i + PREFIX:i + PREFIX + A_SPAN]
            gap_tok = stream[i + PREFIX + A_SPAN:i + PREFIX + A_SPAN + gap]
            j = torch.randint(0, stream.shape[0] - A_SPAN - 1, (1,), generator=gen).item()
            rep.append(torch.cat([pre, A, gap_tok, A]))
            fre.append(torch.cat([pre, A, gap_tok, stream[j:j + A_SPAN]]))
        ce_r_sum += per_token_ce(m, torch.stack(rep).cuda())[:, lo:hi].sum().item()
        ce_f_sum += per_token_ce(m, torch.stack(fre).cuda())[:, lo:hi].sum().item()
        n += BS * (hi - lo)
    return ce_r_sum / n, ce_f_sum / n


for gap in gaps:
    row = f'gap={gap:>4}'
    results['B_recall'][gap] = {}
    for k, m in models.items():
        ce_r, ce_f = recall_ce(m, gap)
        gain = ce_f - ce_r
        results['B_recall'][gap][k] = (ce_r, ce_f, gain)
        row += f'  {k}: rep {ce_r:5.2f} / fresh {ce_f:5.2f} → gain {gain:+5.2f}'
    print(row)

# ==================== Test C: 分位置 loss 曲线 ====================
print('\n===== C. 分位置 loss（L=384，8 桶）—— 上下文利用曲线 =====')
L = 384
nb = 8
results['C_pos'] = {}
for k, m in models.items():
    n = min(256, (val_stream.shape[0] - 1) // (L + 1))
    stride = max(1, (val_stream.shape[0] - L - 1) // n)
    starts = (torch.arange(0, n) * stride).tolist()
    buckets = [[] for _ in range(nb)]
    for s in range(0, n, 16):
        idx = starts[s:s + 16]
        win = torch.stack([val_stream[i:i + L + 1] for i in idx]).cuda()
        ce = per_token_ce(m, win)
        blen = ce.shape[1] // nb
        for b in range(nb):
            lo, hi = b * blen, (b + 1) * blen if b < nb - 1 else ce.shape[1]
            buckets[b].append(ce[:, lo:hi].mean().item())
    results['C_pos'][k] = [sum(b) / len(b) for b in buckets]
    print(f'{k}: ' + ' '.join(f'{x:5.2f}' for x in results['C_pos'][k]))

# ============ Test D: v3 学习到的 γ 有效记忆长度 ============
print('\n===== D. γ 有效记忆长度（真实语料统计） =====')
results['D_gamma'] = {}
for k in [key for key in MODEL_KEYS if key not in wrapper_keys]:
    m = models[k]
    ids = stream[:8192].view(1, -1)
    horizons = []
    with torch.no_grad():
        for li, layer in enumerate(m.decoder_layers):
            att = layer.self_attention_linear
            DH, H = att.d_head, att.heads
            c_raw = att.state_em(ids)
            if att.state_proj is not None:
                c_raw = att.state_proj(c_raw)
            c = c_raw.view(1, -1, DH, H).permute(0, 2, 1, 3) * att.em_scale
            v = att.decay_v.view(DH, H)
            b = att.decay_b.view(DH, H)
            gam = torch.exp(-F.softplus(c * v.view(1, DH, 1, H) + b.view(1, DH, 1, H)))
            gm = gam.mean().item()
            gq = torch.quantile(gam.flatten().float(),
                                torch.tensor([0.1, 0.5, 0.9], device=DEV))
            hor = (math.log(0.5) / math.log(max(gm, 1e-6))) if gm < 1 else float('inf')
            horizons.append((gm, hor))
            print(f'  {k} layer {li:2d}: gamma_mean={gm:.4f}  '
                  f'q10/50/90={gq[0]:.3f}/{gq[1]:.3f}/{gq[2]:.3f}  半衰期≈{hor:7.1f} tok')
    results['D_gamma'][k] = horizons

with open(r'F:\OpenASH2605\longrange_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
print('\n结果已存 longrange_results.json')
