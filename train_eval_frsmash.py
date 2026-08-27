"""
FRSMASH v1/v2/v3 训练 + 评测
同条件训练 3000 steps, 然后对比:
  1. 训练 loss 收敛
  2. PPL 外推 (128~8192)
  3. 状态稳定性 (5000步)
  4. 长程 Copy (4~4096)
"""
import os, sys, time, math, json, random, torch
import torch.nn.functional as F
sys.path.insert(0, '.')
from frsmash import FRSMASH as V1
from frsmash_v2 import FRSMASH as V2
from frsmash_v3 import FRSMASH as V3
from open_ash_voc import OpenASHVoc
from config import agent_voc_path

DEVICE = 'cuda'
SEED = 42
STEPS = 2000
BS = 64
SEQ = 256
LR = 6e-4
H, HEADS, LAYERS, K = 256, 8, 4, 8

voc = OpenASHVoc(agent_voc_path=agent_voc_path)
VS = len(voc.token_to_id) + 1
print(f'Vocab: {VS}', flush=True)

# ========== 数据 ==========
print('Loading data...', flush=True)
data_path = os.path.join('minimind_data', 'pretrain_t2t_mini.jsonl')
all_seqs = []
with open(data_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 50000: break
        try: text = json.loads(line).get('text', '')
        except: continue
        if len(text) < 50: continue
        ids = voc.encode(text)
        if len(ids) >= 16:
            all_seqs.append(ids)
print(f'Sequences: {len(all_seqs)}', flush=True)

# Long texts for PPL eval
eval_texts, ls = [], []
random.seed(0)
shuffled = all_seqs.copy()
random.shuffle(shuffled)
for seq in shuffled:
    ls.extend(seq)
    if len(ls) >= 16384:
        eval_texts.append(ls[:16384]); ls = []
        if len(eval_texts) >= 3: break
print(f'Eval texts: {len(eval_texts)}', flush=True)


def make_batch():
    xs = []
    for _ in range(BS):
        seq = random.choice(all_seqs)
        if len(seq) > SEQ:
            start = random.randint(0, len(seq) - SEQ)
            xs.append(seq[start:start+SEQ])
        else:
            xs.append(seq + [0] * (SEQ - len(seq)))
    return torch.tensor(xs, device=DEVICE)


def gpu_mem():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0


def train_model(model_cls, name):
    torch.manual_seed(SEED)
    random.seed(SEED)
    model = model_cls(VS, H, HEADS, LAYERS, K).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        x = make_batch()
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, VS), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        losses.append(loss.item())
        if step == 10:
            peak = gpu_mem()
            print(f'  [{name}] GPU peak mem: {peak:.2f} GB (BS={BS}, SEQ={SEQ})', flush=True)
        if step % 500 == 0:
            avg = sum(losses[-500:]) / 500
            spd = step / (time.time() - t0)
            print(f'  [{name}] step {step}/{STEPS} loss={avg:.4f} ({spd:.1f} step/s)', flush=True)
    model.eval()
    final_loss = sum(losses[-200:]) / 200
    print(f'  [{name}] DONE final_loss={final_loss:.4f} ({time.time()-t0:.0f}s)', flush=True)
    return model, final_loss


# ========== 训练 ==========
print(f'\n{"="*60}')
print(f'Training {STEPS} steps each (BS={BS}, SEQ={SEQ}, LR={LR})')
print(f'{"="*60}')

m1, l1 = train_model(V1, 'v1-cummax')
m2, l2 = train_model(V2, 'v2-flayer')
m3, l3 = train_model(V3, 'v3-dual')

print(f'\nFinal training losses:')
print(f'  v1 cummax:  {l1:.4f}')
print(f'  v2 F-layer: {l2:.4f}')
print(f'  v3 dual:    {l3:.4f}')

# ========== 评测 ==========
models = {'v1 cummax': m1, 'v2 F-layer': m2, 'v3 dual': m3}

# ---- 1. PPL 外推 ----
print(f'\n{"="*60}')
print('Eval 1: PPL Extrapolation')
print(f'{"="*60}')
ctx_lens = [128, 384, 1024, 2048, 4096, 8192]
print(f'  {"Ctx":>6s}', end='')
for n in models: print(f'  {n:>12s}', end='')
print()
for cl in ctx_lens:
    print(f'  {cl:>6d}', end='')
    for name, m in models.items():
        ns = min(len(eval_texts), 3) if cl > 4096 else min(len(eval_texts), 3)
        losses = []
        for seq in eval_texts[:ns]:
            if len(seq) < cl + 8: continue
            with torch.no_grad():
                full = torch.tensor([seq[:cl+7]], device=DEVICE)
                logits = m(full)
                loss = sum(F.cross_entropy(logits[:, cl-1+i, :],
                    torch.tensor([seq[cl+i]], device=DEVICE)).item() for i in range(8)) / 8
            if loss < 20: losses.append(loss)
        ppl = math.exp(sum(losses)/len(losses)) if losses else 999
        print(f'  {ppl:>12.2f}', end='')
    print()

# ---- 2. 状态稳定性 ----
print(f'\n{"="*60}')
print('Eval 2: State Stability (5000 steps)')
print(f'{"="*60}')
for name, m in models.items():
    token = torch.tensor([[42]], device=DEVICE)
    ash = [None] * LAYERS
    hs = torch.zeros(1, H, device=DEVICE)
    print(f'\n  {name}:')
    prev = 0
    for cp in [10, 500, 2000, 5000]:
        for _ in range(cp - prev):
            with torch.no_grad():
                logits, ash, hs = m.generate_step(token, ash, hs)
                token = logits.argmax(dim=-1, keepdim=True)
        prev = cp
        sn = hs.norm().item()
        if ash[0] is not None:
            if isinstance(ash[0], tuple):
                sf = ash[0][0].norm().item()
                sc = ash[0][1].norm().item()
                bn = f'F={sf:.1f} C={sc:.1f}'
            else:
                bn = f'BB={ash[0].norm().item():.1f}'
        else:
            bn = 'N/A'
        nan = torch.isnan(hs).any().item()
        print(f'    step={cp:>5d} | slow={sn:.3f} | {bn} | {"FAIL" if nan else "OK"}', flush=True)

# ---- 3. 长程 Copy ----
print(f'\n{"="*60}')
print('Eval 3: Long-range Copy (target logit gap)')
print(f'{"="*60}')
TA, PAD = 66, 0
dists = [4, 64, 256, 1024, 4096]
print(f'  {"Dist":>6s}', end='')
for n in models: print(f'  {n:>12s}', end='')
print()
for d in dists:
    print(f'  {d:>6d}', end='')
    for name, m in models.items():
        tgt_l, rnd_l = [], []
        for _ in range(5):
            rt = random.randint(10, 100)
            while rt == TA: rt = random.randint(10, 100)
            s = [TA] + [PAD] * d
            with torch.no_grad():
                log, ash, hs = m.generate_step(
                    torch.tensor([[s[0]]], device=DEVICE), [None]*LAYERS, torch.zeros(1,H,device=DEVICE))
                for t in s[1:]:
                    log, ash, hs = m.generate_step(torch.tensor([[t]], device=DEVICE), ash, hs)
            tgt_l.append(log[0, TA].item())
            rnd_l.append(log[0, rt].item())
        gap = sum(tgt_l)/len(tgt_l) - sum(rnd_l)/len(rnd_l)
        print(f'  {gap:>+12.3f}', end='')
    print()

# ---- v3 融合参数 ----
print(f'\n{"="*60}')
print('v3 Learned Fusion Parameters')
print(f'{"="*60}')
for i, layer in enumerate(m3.ash_layers):
    a = torch.sigmoid(layer.attn.fuse_logit).item()
    s = (torch.nn.functional.softplus(layer.attn.cm_scale) + 0.5).item()
    print(f'  Layer {i}: alpha={a:.3f} (cummax={a*100:.0f}%, flayer={100-a*100:.0f}%) | cm_scale={s:.2f}')

print(f'\nAll done!', flush=True)
