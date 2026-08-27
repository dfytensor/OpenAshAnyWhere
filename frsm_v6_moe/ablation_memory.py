"""
FRSMASH 消融实验 — 记忆维度(追测)
复用已有结果, 只跑记忆相关新实验
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

# ============ 数据 ============
def get_loader(data_dir, max_lines, max_seq_len, batch_size):
    cache = Path(data_dir) / f"pretrain_cached_{max_lines}_{max_seq_len}.pt"
    data = torch.load(cache, weights_only=True)
    class DS(torch.utils.data.Dataset):
        def __len__(s): return len(data)
        def __getitem__(s, i): d = data[i]; return d[:max_seq_len+1] if len(d) > max_seq_len+1 else d
        @staticmethod
        def collate_fn(items): p = pad_sequence(items, batch_first=True, padding_value=0); return p[:, :-1], p[:, 1:]
    return DataLoader(DS(), batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=DS.collate_fn, drop_last=True)

# ============ 模型 ============
def build_frsmash(vs, H, L, K=8):
    from frsmash import FRSMASH; return FRSMASH(vs, H, 8, L, K=K)
def build_ash_only(vs, H, L):
    from frsmash import FRSMASH; return FRSMASH(vs, H, 8, L, K=999999)
def build_slow_only(vs, D):
    from frsm_linear import HybridFRSM_LM; return HybridFRSM_LM(vs, D, 0, 1, slow_update_freq=1)

# ============ 记忆任务: CopyFirst ============
def memory_test(model, vs, device, T=384):
    """CopyFirst 测试: 模型能否记住序列第一个 token
    生成随机序列 → 取第一个 token 作为 target
    → 用后续 token 驱动状态 → 最后一步能否预测第一个 token
    返回正确 token 的 logit 平均值(越高=记忆越强)
    """
    model.eval()
    B = 32
    # 生成测试序列: 所有 token 随机,但 target 是第一个 token
    x = torch.randint(1, vs, (B, T), device=device)
    target = x[:, 0]  # (B,)
    
    # 用完整 forward 获取 logits
    with torch.no_grad():
        logits = model(x)  # (B, T, vs)
        # 取最后一个位置的 logit,检查 target token 的概率
        last_logits = logits[:, -1, :]  # (B, vs)
        # target 的 logit (未归一化概率)
        target_logits = last_logits[torch.arange(B), target]
    return target_logits.mean().item()

# ============ 训练 + 记忆测试 ============
def train_one(model, vs, loader, device, total_steps, do_memory=True):
    model = model.to(device); model.train()
    opt = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01, betas=(0.9, 0.95))
    warmup = 100
    lr_fn = lambda s: min(s/warmup, 1.0) if s < warmup else max(0, 0.5*(1+math.cos(math.pi*(s-warmup)/(total_steps-warmup))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    it = iter(loader); step = 0; acc = 0; t0 = time.time(); losses = []
    
    while step < total_steps:
        x, t = next(it, (None, None))
        if x is None: it = iter(loader); continue
        x, t = x.to(device), t.to(device)
        lg = model(x)
        loss = F.cross_entropy(lg.reshape(-1, vs), t.reshape(-1), ignore_index=0)
        if torch.isnan(loss): continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1; acc += loss.item()
        if step % 200 == 0:
            avg = acc/200; losses.append(avg); acc = 0
    
    elapsed = time.time() - t0
    # 记忆测试
    mem_score = memory_test(model, vs, device) if do_memory else 0
    return {'final_loss': losses[-1] if losses else 0, 'losses': losses, 'time': elapsed, 'mem_score': mem_score}

# ============ 主实验 ============
def run():
    device = torch.device("cuda")
    data_dir = "../minimind_data"
    max_lines, max_seq_len, batch_size, total_steps = 30000, 384, 64, 1500
    vs = 23005
    loader = get_loader(data_dir, max_lines, max_seq_len, batch_size)
    
    # 加载已有结果
    old_file = Path("ablation_results.json")
    results = json.loads(old_file.read_text()) if old_file.exists() else {}
    print(f"已加载 {len(results)} 项已有结果\n")

    # ====== F. 慢记忆 K 值 (更新频率) ======
    print("="*50)
    print("F. 慢记忆 K 值 (H=512 L=4)")
    print("="*50)
    for K in [1, 2, 4, 8, 16, 999999]:
        tag = f"F_K{K}"
        if tag in results: print(f"  {tag} 已有, 跳过"); continue
        print(f"\n>>> {tag}: K={K} (更新周期)")
        m = build_frsmash(vs, 512, 4, K=K)
        p = sum(pp.numel() for pp in m.parameters())/1e6
        r = train_one(m, vs, loader, device, total_steps, do_memory=True)
        r['params'] = p; r['config'] = f"H=512 L=4 K={K}"
        results[tag] = r
        print(f"    => loss={r['final_loss']:.4f} mem_score={r['mem_score']:.2f} time={r['time']:.0f}s")
        del m; torch.cuda.empty_cache()
        old_file.write_text(json.dumps(results, indent=2))

    # ====== G. 慢尺度数量 (H=512 L=4, K=8) ======
    # 手动构建不同 NS 的模型
    print("\n" + "="*50)
    print("G. 慢尺度数量 (H=512 L=4 K=8)")
    print("="*50)
    for ns_val in [1, 2, 4]:
        tag = f"G_NS{ns_val}"
        if tag in results: print(f"  {tag} 已有, 跳过"); continue
        print(f"\n>>> {tag}: ns={ns_val}")
        # 多慢尺度: 用 HybridFRSM 0F+nS (纯慢,无 OpenASH,只看记忆能力)
        from frsm_linear import HybridFRSM_LM
        m = HybridFRSM_LM(vs, 512, 0, ns_val, slow_update_freq=8)
        p = sum(pp.numel() for pp in m.parameters())/1e6
        r = train_one(m, vs, loader, device, total_steps, do_memory=True)
        r['params'] = p; r['config'] = f"0F+{ns_val}S K=8"
        results[tag] = r
        print(f"    => loss={r['final_loss']:.4f} mem_score={r['mem_score']:.2f} time={r['time']:.0f}s")
        del m; torch.cuda.empty_cache()
        old_file.write_text(json.dumps(results, indent=2))

    # ====== H. 记忆任务消融 (完整 vs 去Slow → CopyFirst) ======
    print("\n" + "="*50)
    print("H. 记忆任务消融 (H=512 L=4 K=8)")
    print("="*50)
    mem_configs = [
        ("H_full",    "完整FRSMASH",  lambda: build_frsmash(vs, 512, 4, K=8)),
        ("H_ash_only","去Slow",       lambda: build_ash_only(vs, 512, 4)),
        ("H_slow_only","去OpenASH",   lambda: build_slow_only(vs, 512)),
    ]
    for tag, name, builder in mem_configs:
        if tag in results: print(f"  {tag} 已有, 跳过"); continue
        print(f"\n>>> {tag}: {name}")
        m = builder()
        p = sum(pp.numel() for pp in m.parameters())/1e6
        r = train_one(m, vs, loader, device, total_steps, do_memory=True)
        r['params'] = p; r['config'] = name
        results[tag] = r
        print(f"    => loss={r['final_loss']:.4f} mem_score={r['mem_score']:.2f} time={r['time']:.0f}s")
        del m; torch.cuda.empty_cache()
        old_file.write_text(json.dumps(results, indent=2))

    # ====== 汇总 ======
    print("\n" + "="*70)
    print("记忆维度消融汇总")
    print("="*70)
    for grp, keys in [("F. K值", [f"F_K{k}" for k in [1,2,4,8,16,999999]]),
                       ("G. 慢尺度数", [f"G_NS{n}" for n in [1,2,4]]),
                       ("H. 记忆消融", ["H_full","H_ash_only","H_slow_only"])]:
        print(f"\n{grp}:")
        print(f"{'exp':>12} {'config':>16} {'loss':>8} {'mem_score':>10} {'time':>8}")
        for k in keys:
            if k in results:
                r = results[k]
                print(f"{k:>12} {r['config']:>16} {r['final_loss']:>8.4f} {r.get('mem_score',0):>10.2f} {r['time']:>8.0f}s")

    old_file.write_text(json.dumps(results, indent=2))
    print("\nDone.")

if __name__ == "__main__":
    run()
