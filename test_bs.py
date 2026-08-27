"""FRSM bs=64 训练吞吐实测"""
import os, sys, time, math, torch, json
import torch.nn.functional as F
sys.path.insert(0, 'F:/OpenASH2605')
from config import agent_voc_path
from open_ash_voc import OpenASHVoc
from frsm.model import FractalRecursiveStateMachine
from frsm.dataset import create_dataloaders
from frsm.config import FRSMConfig

device = torch.device("cuda")
voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1

model = FractalRecursiveStateMachine(vocab_size=vs, d_model=256, num_scales=4)
model = model.to(device)
model.train()
print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

config = FRSMConfig(d_model=256, num_scales=4, max_seq_len=384, batch_size=64, max_pretrain_lines=5000)
loader = create_dataloaders(voc, mode='pretrain', config=config)

opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01, betas=(0.9, 0.95))

print(f"\nWarmup...", flush=True)
x, t = next(iter(loader))
x = x.to(device); t = t.to(device)

# warmup
for _ in range(3):
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, vs), t.reshape(-1), ignore_index=0)
    loss.backward()
    opt.zero_grad(set_to_none=True)
torch.cuda.synchronize()

print(f"Benchmark 20 steps...", flush=True)
torch.cuda.synchronize()
t0 = time.time()
total_tokens = 0

for step in range(20):
    try:
        x, t = next(loader_iter)
    except:
        loader_iter = iter(loader)
        x, t = next(loader_iter)
    
    x = x.to(device, non_blocking=True)
    t = t.to(device, non_blocking=True)
    
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, vs), t.reshape(-1), ignore_index=0)
    loss.backward()
    opt.zero_grad(set_to_none=True)
    
    total_tokens += x.numel()

torch.cuda.synchronize()
elapsed = time.time() - t0
tok_per_s = total_tokens / elapsed

print(f"\n=== Results ===", flush=True)
print(f"  batch_size: 64", flush=True)
print(f"  seq_len: {x.size(1)}", flush=True)
print(f"  tokens/step: {x.numel():,}", flush=True)
print(f"  total tokens: {total_tokens:,}", flush=True)
print(f"  time: {elapsed:.1f}s", flush=True)
print(f"  step time: {elapsed/20:.2f}s", flush=True)
print(f"  throughput: {tok_per_s:.0f} tok/s", flush=True)

# 估算全量训练时间
total_data = 250_000_000  # 2.5亿 token
epochs = 3
total_training_tokens = total_data * epochs
days = total_training_tokens / tok_per_s / 86400

print(f"\n  Estimated full training ({total_data/1e6:.0f}M token, {epochs} epochs):", flush=True)
print(f"    {days:.1f} days", flush=True)
print(f"Done.", flush=True)
