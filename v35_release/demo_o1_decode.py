"""v3.5 O(1) 增量解码端到端演示（big_v35）：
- 双状态携带（max-scan e + KV 矩阵 S）逐 token 生成
- 证明每步延迟恒定（不随已生成长度增长 → O(1)）
- 对比：tf 每步延迟随长度线性增长（O(S)）
"""
import sys, time, math
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
import torch
from open_ash_v3 import OpenASH_V3
from baseline_tf import TransformerLM
from open_ash_voc import OpenASHVoc
from config import agent_voc_path

DEV = 'cuda'
voc = OpenASHVoc(agent_voc_path=agent_voc_path)

v35 = OpenASH_V3(23004, 768, 8, 12, max_seq_len=2048, model_flag='train', use_fused=True,
                 use_decay=True, pre_norm=True, ffn_act='silu',
                 decay_init=(0.0001, 0.35), decay_v_std=0.04,
                 kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True).to(DEV)
v35.load_state_dict(torch.load(r'big_v35\model.pt', map_location='cpu', weights_only=True))
v35.eval()
tf = TransformerLM(23004, 768, 8, 11, max_seq_len=2048).to(DEV)
tf.load_state_dict(torch.load(r'big_tf\model.pt', map_location='cpu', weights_only=True))
tf.eval()


@torch.no_grad()
def gen_v35(prompt, n_new=256, temperature=0.8, top_k=50):
    ids = voc.encode(prompt)
    x = torch.tensor([ids], device=DEV)
    state = None
    lat = []
    out, state = v35(x)
    for i in range(n_new):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        logits = out[0, -1].float() / temperature
        kk = min(top_k, logits.shape[-1])
        vv, _ = torch.topk(logits, kk)
        logits[logits < vv[-1]] = -float('inf')
        nxt = torch.multinomial(torch.softmax(logits, -1), 1).item()
        out, state = v35(torch.tensor([[nxt]], device=DEV), state=state,
                         pos_offset=len(ids) + i)
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
        generated = None
    # 状态大小统计
    e_sizes = sum(s[0].numel() for s in state)
    kv_sizes = sum(s[1].numel() for s in state)
    return lat, e_sizes, kv_sizes


@torch.no_grad()
def gen_tf(prompt, n_new=256, temperature=0.8, top_k=50):
    """tf 无状态缓存实现（每步全前缀重算 → O(S²) 总量，单步 O(S)）。"""
    ids = voc.encode(prompt)
    x = torch.tensor([ids], device=DEV)
    lat = []
    for i in range(n_new):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = tf(x)
        logits = logits[0, -1].float() / temperature
        kk = min(top_k, logits.shape[-1])
        vv, _ = torch.topk(logits, kk)
        logits[logits < vv[-1]] = -float('inf')
        nxt = torch.multinomial(torch.softmax(logits, -1), 1).item()
        x = torch.cat([x, torch.tensor([[nxt]], device=DEV)], dim=1)
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
    return lat


prompt = '人工智能是'
lat35, e_sz, kv_sz = gen_v35(prompt)
lattf = gen_tf(prompt)

print('状态大小/层: max-scan e =', e_sz // 12, '数值 | KV S =', kv_sz // 12, '数值 | 12 层合计',
      (e_sz + kv_sz), '数值（常数，与长度无关）')
print()
print('每 token 延迟（ms）：前 64 / 中段(96-160) / 末段(192-256)')
print(f'  v3.5 增量O(1): {min(lat35[:64])*1000:6.2f} / '
      f'{sum(lat35[96:160])/64*1000:6.2f} / {sum(lat35[192:])/64*1000:6.2f}')
print(f'  tf 全前缀重算: {min(lattf[:64])*1000:6.2f} / '
      f'{sum(lattf[96:160])/64*1000:6.2f} / {sum(lattf[192:])/64*1000:6.2f}')
r_early = sum(lattf[96:160]) / 64 / (sum(lat35[96:160]) / 64)
r_late = sum(lattf[192:]) / 64 / (sum(lat35[192:]) / 64)
print(f'  tf/v35 延迟比：中段 {r_early:.1f}x → 末段 {r_late:.1f}x（随长度扩大）')
print()
# 生成样例
ids = voc.encode(prompt)
x = torch.tensor([ids], device=DEV)
state = None
out, state = v35(x)
gen = []
for i in range(120):
    logits = out[0, -1].float() / 0.8
    kk = min(50, logits.shape[-1])
    vv, _ = torch.topk(logits, kk)
    logits[logits < vv[-1]] = -float('inf')
    nxt = torch.multinomial(torch.softmax(logits, -1), 1).item()
    gen.append(nxt)
    out, state = v35(torch.tensor([[nxt]], device=DEV), state=state, pos_offset=len(ids) + i)
print('生成示例:', prompt + voc.decode(gen)[:150])
