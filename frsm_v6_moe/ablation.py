"""
FRSMASH 消融实验 — 记忆/逻辑解耦验证

实验组:
  A. 逻辑轴: 固定记忆宽度, 变 OpenASH 层数 (L=2,4,6,8)
  B. 记忆轴: 固定逻辑层数, 变 d_model (D=256,384,512,640)
  C. 组件消融: 去OpenASH / 去Slow / 去FusionGate
  D. 快慢比: 3F+1S / 2F+2S / 1F+1S / 0F+4S(HybridFRSM)

每组训练固定 token 量, 对比最终 loss
"""
import os, sys, time, math, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys as _sys; _sys.stdout.reconfigure(line_buffering=True)

import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

torch.set_float32_matmul_precision('high')

# ============ 数据加载(缓存) ============
def get_loader(data_dir, max_lines, max_seq_len, batch_size):
    cache = Path(data_dir) / f"pretrain_cached_{max_lines}_{max_seq_len}.pt"
    if cache.exists():
        data = torch.load(cache, weights_only=True)
    else:
        raise FileNotFoundError(f"Cache not found: {cache}")
    class DS(torch.utils.data.Dataset):
        def __len__(s): return len(data)
        def __getitem__(s, i):
            ids = data[i]
            return ids[:max_seq_len+1] if len(ids) > max_seq_len+1 else ids
        @staticmethod
        def collate_fn(items):
            padded = pad_sequence(items, batch_first=True, padding_value=0)
            return padded[:, :-1], padded[:, 1:]
    ds = DS()
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=0, collate_fn=DS.collate_fn, drop_last=True)

# ============ 模型构建 ============
def build_frsmash(vs, H, L, K=8, heads=8):
    from frsmash import FRSMASH
    return FRSMASH(vs, H, heads, L, K=K)

def build_hybrid(vs, D, nf, ns, K=8):
    from frsm_linear import HybridFRSM_LM
    return HybridFRSM_LM(vs, D, nf, ns, K)

def build_ash_only(vs, H, L, heads=8):
    """OpenASH 无慢记忆"""
    from frsmash import FRSMASH
    m = FRSMASH(vs, H, heads, L, K=999999)  # K极大=慢尺度几乎不更新
    return m

def build_slow_only(vs, D):
    """纯慢尺度无 OpenASH (≈V6单尺度)"""
    from frsm_linear import HybridFRSM_LM
    return HybridFRSM_LM(vs, D, 0, 1, slow_update_freq=1)

def build_hybrid_ash(vs, H, nf, na, K=8):
    """FRSMASH Hybrid: 浅层Fast + 深层OpenASH"""
    from frsmash_f import FastLayer
    from frsmash import ASHDecoderLayer, SlowMemoryCell
    import torch.nn as nn
    class FRSMASH_Hybrid(nn.Module):
        def __init__(self):
            super().__init__(); self.D=H; self.K=K
            self.em=nn.Embedding(vs,H,padding_idx=0)
            self.fast=nn.ModuleList([FastLayer(H) for _ in range(nf)])
            self.ash=nn.ModuleList([ASHDecoderLayer(H,8,'train') for _ in range(na)])
            self.bn=nn.LayerNorm(H); self.mp=nn.Linear(H,H); self.cell=SlowMemoryCell(H); self.mo=nn.Linear(H,H)
            self.fg=nn.Sequential(nn.Linear(H*2,H//4),nn.GELU(),nn.Linear(H//4,1),nn.Sigmoid())
            self.fn=nn.LayerNorm(H); self.hd=nn.Linear(H,vs,bias=False)
        def forward(self,x):
            B,T=x.shape; D=self.D; h=self.em(x)
            for l in self.fast: h=l(h)+h
            for l in self.ash: h1,_=l(h); h=h1+h
            xb=self.bn(h); iseq=self.mp(self.em(x)); hs=torch.zeros(B,D,device=x.device); Hs=torch.zeros(B,T,D,device=x.device); p=0
            for t in range(0,T,self.K): hs=self.cell(iseq[:,t],hs); Hs[:,p:t+1]=hs.unsqueeze(1); p=t+1
            if p<T: Hs[:,p:]=hs.unsqueeze(1)
            xm=self.mo(Hs); c=torch.cat([xb,xm],-1); g=self.fg(c); ft=self.fn(g*xb+(1-g)*xm+self.em(x))
            return self.hd(ft)
    return FRSMASH_Hybrid()  # 0快+1慢, K=1每步更新

# ============ 训练单次 ============
def train_one(model, vs, loader, device, total_steps, lr=5e-4, warmup=100):
    model = model.to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    lr_fn = lambda s: min(s/warmup, 1.0) if s < warmup else max(0, 0.5*(1+math.cos(math.pi*(s-warmup)/(total_steps-warmup))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    model.train()
    step = 0; loss_acc = 0.0; t0 = time.time()
    data_iter = iter(loader)
    losses = []

    while step < total_steps:
        try: x, t = next(data_iter)
        except StopIteration: data_iter = iter(loader); x, t = next(data_iter)
        x, t = x.to(device), t.to(device)

        if hasattr(model, 'frsm'):  # HybridFRSM_LM
            logits = model(x)
        else:  # FRSMASH
            logits = model(x)

        loss = F.cross_entropy(logits.reshape(-1, vs), t.reshape(-1), ignore_index=0)
        if torch.isnan(loss): continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        step += 1; loss_acc += loss.item()

        if step % 200 == 0:
            avg = loss_acc / 200; losses.append(avg)
            if step % 1000 == 0:
                tok_s = step * x.size(0) * x.size(1) / (time.time()-t0)
                print(f"    step {step:>5d}/{total_steps} loss={avg:.4f} {tok_s:.0f}tok/s")
            loss_acc = 0.0

    elapsed = time.time() - t0
    return {'final_loss': losses[-1] if losses else 0, 'losses': losses, 'time': elapsed}

# ============ 主实验 ============
def run():
    device = torch.device("cuda")
    data_dir = "../minimind_data"
    max_lines = 30000  # 用已有缓存
    max_seq_len = 384
    batch_size = 64
    total_steps = 3000  # 每组训练3000步(~110M tokens)
    vs = 23005

    print("="*70)
    print(f"FRSMASH 消融实验 — {total_steps} steps, {max_lines} lines, B={batch_size}")
    print("="*70)

    loader = get_loader(data_dir, max_lines, max_seq_len, batch_size)
    print(f"Dataset: {len(loader.dataset)} samples\n")

    results = {}
    if os.path.exists("ablation_results.json"):
        with open("ablation_results.json") as f: results = json.load(f)
        print(f"接续: 已完成 {list(results.keys())}")

    def done(tag):
        if tag in results:
            print(f"\n>>> {tag}: 已完成, 跳过 (loss={results[tag]['final_loss']:.4f})")
            return True
        return False

    # ====== A. 逻辑轴: H=512 固定, L=2,4,6,8 ======
    print("="*50)
    print("A. 逻辑轴 (H=512, NS=1, 变 OpenASH 层数)")
    print("="*50)
    for L in [2, 4, 6, 8]:
        tag = f"A_L{L}"
        if done(tag): continue
        print(f"\n>>> {tag}: H=512 L={L} heads=8")
        m = build_frsmash(vs, 512, L, K=8)
        p = sum(pp.numel() for pp in m.parameters())/1e6
        print(f"    Params: {p:.1f}M")
        r = train_one(m, vs, loader, device, total_steps)
        r['params'] = p; r['config'] = f"H=512 L={L}"
        results[tag] = r
        print(f"    => final_loss={r['final_loss']:.4f} time={r['time']:.0f}s")
        del m; torch.cuda.empty_cache()
        with open("ablation_results.json", "w") as f: json.dump(results, f, indent=2)

    # ====== B. 记忆轴: L=4 固定, H=256,384,512,640 ======
    print("\n" + "="*50)
    print("B. 记忆轴 (L=4, NS=1, 变 d_model)")
    print("="*50)
    for H in [256, 384, 512, 640]:
        tag = f"B_H{H}"
        if done(tag): continue
        print(f"\n>>> {tag}: H={H} L=4 heads=8")
        m = build_frsmash(vs, H, 4, K=8)
        p = sum(pp.numel() for pp in m.parameters())/1e6
        print(f"    Params: {p:.1f}M")
        r = train_one(m, vs, loader, device, total_steps)
        r['params'] = p; r['config'] = f"H={H} L=4"
        results[tag] = r
        print(f"    => final_loss={r['final_loss']:.4f} time={r['time']:.0f}s")
        del m; torch.cuda.empty_cache()
        with open("ablation_results.json", "w") as f: json.dump(results, f, indent=2)

    # ====== C. 组件消融 ======
    print("\n" + "="*50)
    print("C. 组件消融 (H=512)")
    print("="*50)
    configs = [
        ("C_full",    "FRSMASH完整",     lambda: build_frsmash(vs, 512, 4, K=8)),
        ("C_ash_only","OpenASH无记忆",   lambda: build_ash_only(vs, 512, 4)),
        ("C_slow_only","纯慢尺度(V6)",   lambda: build_slow_only(vs, 512)),
    ]
    for tag, name, builder in configs:
        if done(tag): continue
        print(f"\n>>> {tag}: {name}")
        m = builder()
        p = sum(pp.numel() for pp in m.parameters())/1e6
        print(f"    Params: {p:.1f}M")
        r = train_one(m, vs, loader, device, total_steps)
        r['params'] = p; r['config'] = name
        results[tag] = r
        print(f"    => final_loss={r['final_loss']:.4f} time={r['time']:.0f}s")
        del m; torch.cuda.empty_cache()
        with open("ablation_results.json", "w") as f: json.dump(results, f, indent=2)

    # ====== D. 快慢比 (HybridFRSM) ======
    print("\n" + "="*50)
    print("D. 快慢尺度比 (d=512)")
    print("="*50)
    for nf, ns in [(3,1), (2,2), (1,1), (0,1)]:
        tag = f"D_{nf}F{ns}S"
        if done(tag): continue
        print(f"\n>>> {tag}: d=512 {nf}F+{ns}S")
        m = build_hybrid(vs, 512, nf, ns, K=8)
        p = sum(pp.numel() for pp in m.parameters())/1e6
        print(f"    Params: {p:.1f}M")
        r = train_one(m, vs, loader, device, total_steps)
        r['params'] = p; r['config'] = f"d=512 {nf}F+{ns}S"
        results[tag] = r
        print(f"    => final_loss={r['final_loss']:.4f} time={r['time']:.0f}s")
        del m; torch.cuda.empty_cache()
        with open("ablation_results.json", "w") as f: json.dump(results, f, indent=2)

    # ====== E. Fast/OpenASH 混合比 ======
    print("\n" + "="*50)
    print("E. Fast/OpenASH 混合比 (H=512, L=4)")
    print("="*50)
    for nf, na in [(4,0),(3,1),(2,2),(1,3),(0,4)]:
        tag = f"E_{nf}F{na}A"
        if done(tag): continue
        print(f"\n>>> {tag}: H=512 {nf}F+{na}A")
        m = build_hybrid_ash(vs, 512, nf, na, K=8)
        p = sum(pp.numel() for pp in m.parameters())/1e6
        print(f"    Params: {p:.1f}M")
        r = train_one(m, vs, loader, device, total_steps)
        r['params'] = p; r['config'] = f"H=512 {nf}F+{na}A"
        results[tag] = r
        print(f"    => final_loss={r['final_loss']:.4f} time={r['time']:.0f}s")
        del m; torch.cuda.empty_cache()
        with open("ablation_results.json", "w") as f: json.dump(results, f, indent=2)

    # ====== 汇总 ======
    print("\n" + "="*70)
    print("消融实验汇总")
    print("="*70)
    print(f"{'实验':>12} {'配置':>16} {'参数(M)':>8} {'最终loss':>10} {'时间(s)':>8}")
    print("-"*60)
    for tag, r in results.items():
        print(f"{tag:>12} {r['config']:>16} {r['params']:>8.1f} {r['final_loss']:>10.4f} {r['time']:>8.0f}")

    with open("ablation_results.json", "w") as f: json.dump(results, f, indent=2)
    print(f"\nResults saved to ablation_results.json")

if __name__ == "__main__":
    run()
