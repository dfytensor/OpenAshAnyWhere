"""
FRSM v2 快速消融: 1层 vs 3层 vs 5层
小数据 + 少步数, 看 loss 趋势即可
"""
import os, sys, time, math, torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, 'F:/OpenASH2605')
from config import agent_voc_path
from open_ash_voc import OpenASHVoc
from frsm.dataset import PretrainDataset
from frsm_v2.model import MultiLayerFRSM

device = torch.device("cuda")
voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1

def get_lr_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps: return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 小规模数据
dataset = PretrainDataset("minimind_data/pretrain_t2t_mini.jsonl", voc, max_len=384, max_lines=2000)
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=PretrainDataset.collate_fn, drop_last=True)

torch.manual_seed(42)

print(f"{'='*55}")
print(f"  FRSM Depth Ablation: 1 vs 3 vs 5 layers")
print(f"  d_model=256, 4 scales, 500 steps, bs=4")
print(f"{'='*55}")

results = {}
for num_layers in [1, 3, 5]:
    torch.manual_seed(42)
    model = MultiLayerFRSM(vocab_size=vs, d_model=256, num_scales=4,
                           num_layers=num_layers).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"\n  [{num_layers}-layer] {n:,} params", flush=True)
    
    model.train()
    opt = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
    sch = get_lr_schedule(opt, 50, 500)
    
    step = 0; loss_hist = []; best = float('inf')
    data_iter = iter(loader)
    t0 = time.time()
    
    while step < 500:
        try: x, t = next(data_iter)
        except StopIteration: data_iter = iter(loader); x, t = next(data_iter)
        
        x, t = x.to(device), t.to(device)
        logits, _, crit = model(x, return_state=True, compute_critical_loss=True)
        lm_loss = F.cross_entropy(logits.reshape(-1, vs), t.reshape(-1), ignore_index=0)
        total = lm_loss + model.critical_reg_coeff * crit
        
        opt.zero_grad(set_to_none=True); total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        
        step += 1
        if lm_loss.item() < best: best = lm_loss.item()
        
        if step % 100 == 0:
            print(f"    step{step:4d} lm={lm_loss.item():.4f} best={best:.4f} {time.time()-t0:.0f}s", flush=True)
    
    results[num_layers] = {'best': best, 'time': time.time()-t0, 'params': n}
    del model; torch.cuda.empty_cache()

# 对比
print(f"\n{'='*55}")
print(f"  Results")
print(f"{'='*55}")
print(f"  {'Layers':>8} | {'Params':>10} | {'Best LM':>10} | {'Δ vs 1L':>10} | {'Time(s)':>10} | {'T/s/param':>10}")
print(f"  " + "-" * 68)
bl_1 = results[1]['best']
for nl in [1, 3, 5]:
    r = results[nl]
    delta = r['best'] - bl_1
    tpp = r['time'] / r['params'] * 1e6 if r['params'] > 0 else 0
    print(f"  {nl:8d} | {r['params']:>10,} | {r['best']:10.4f} | {delta:+10.4f} | {r['time']:10.0f} | {tpp:10.2f}")

improvement = (bl_1 - results[3]['best']) / bl_1 * 100
print(f"\n  3-layer improves loss by {improvement:.1f}% over 1-layer")
if improvement > 5:
    print(f"  => Depth significantly helps. The single-layer architecture was bottlenecked.")
elif improvement > 1:
    print(f"  => Depth helps modestly. Architecture is not severely bottlenecked but benefits from stacking.")
else:
    print(f"  => Adding layers has minimal effect. The bottleneck is elsewhere (model size, data, etc).")

print(f"\nDone.")
