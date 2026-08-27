"""快速吞吐基准: 测不同 batch / compile 下的 tok/s, 找最优训练配置."""
import os, sys, time
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONUTF8', '1')
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import FRSMASHv37

VS, H, L, HD, SEQ = 23006, 448, 7, 8, 512
dev = 'cuda'
m = FRSMASHv37(VS, H, HD, L).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
scaler = torch.amp.GradScaler('cuda')


def bench(bs, ga, compile_on, steps=15):
    mm = m
    tag = ''
    if compile_on:
        try:
            mm = torch.compile(m, mode='reduce-overhead')
            tag = '+compile'
        except Exception as e:
            print(f'  compile skip: {e}'); return
    mm.train()
    # warmup
    for _ in range(3):
        x = torch.randint(0, VS, (bs, SEQ), device=dev)
        t = torch.randint(0, VS, (bs, SEQ), device=dev)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = F.cross_entropy(mm(x).reshape(-1, VS), t.reshape(-1), ignore_index=0) / ga
        scaler.scale(loss).backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(ga):
            x = torch.randint(0, VS, (bs, SEQ), device=dev)
            t = torch.randint(0, VS, (bs, SEQ), device=dev)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = F.cross_entropy(mm(x).reshape(-1, VS), t.reshape(-1), ignore_index=0) / ga
            scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    el = time.time() - t0
    tps = steps * ga * bs * SEQ / el
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f'  bs={bs} ga={ga}{tag}: {tps/1e3:.0f}k tok/s  ({el/steps:.2f}s/step)  mem={mem:.1f}GB')
    torch.cuda.reset_peak_memory_stats()


print('=== 吞吐基准 (FRSMASH v3.7 60M, seq=512, 4090) ===')
for bs, ga in [(32, 4), (64, 2), (96, 2), (128, 1)]:
    try:
        bench(bs, ga, False)
    except RuntimeError as e:
        print(f'  bs={bs}: OOM/skip ({str(e)[:40]})'); torch.cuda.empty_cache()
print('--- torch.compile ---')
for bs, ga in [(64, 2), (128, 1)]:
    try:
        bench(bs, ga, True)
    except Exception as e:
        print(f'  bs={bs} compile: skip ({str(e)[:40]})'); torch.cuda.empty_cache()
