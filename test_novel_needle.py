"""FRSM Novel Test V2: 多token针 + 大文件 + PPL 追踪"""
import os, sys, math, torch, time, json
import torch.nn.functional as F

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, 'F:/OpenASH2605')
from config import agent_voc_path
from open_ash_voc import OpenASHVoc
from frsm.model import FractalRecursiveStateMachine

device = torch.device("cuda")
voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1

ckpt = torch.load("frsm_checkpoints/frsm_pretrain_final.pt", map_location='cpu')
model = FractalRecursiveStateMachine(vocab_size=vs, d_model=256, num_scales=4)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model = model.to(device).eval()
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

# 找一本较长的小说 (1-5MB)
novel_dir = r"F:\小说\女生小说"
candidates = []
for f in os.listdir(novel_dir):
    if not f.endswith('.txt'): continue
    path = os.path.join(novel_dir, f)
    size = os.path.getsize(path)
    if 500*1024 < size < 5*1024*1024:  # 500KB - 5MB
        candidates.append((f, path, size))

candidates.sort(key=lambda x: x[2])

print(f"\n=== Novel Needle Test V2 ===", flush=True)
print(f"Candidates ({500}KB-5MB): {len(candidates)} novels", flush=True)

# 选 2 本: 一本短的、一本长一点的
for novel_name, path, file_size in candidates[::max(1, len(candidates)//2)][:2]:
    print(f"\n{'='*60}", flush=True)
    print(f"  Novel: {novel_name} ({file_size/1024:.0f} KB)", flush=True)
    
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    max_chars = 500000  # ~50万字符 ≈ 35万token
    text = text[:max_chars]
    ids = voc.encode(text)
    
    print(f"  Encoded: {len(ids):,} tokens", flush=True)
    
    # === 多 token 针 + PPL 追踪 ===
    # 在文本开头附近找一段内容当作"针"
    needle_len = 30  # 30 token 的针
    needle_start = min(200, len(ids) - needle_len - 1)
    needle = ids[needle_start:needle_start + needle_len]
    
    # 创建版本: with needle vs without (把针内容替换为随机ID)
    ids_with = ids.copy()
    ids_without = ids.copy()
    for i in range(needle_len):
        ids_without[needle_start + i] = (needle_start + i) % vs  # 替换为噪音
    
    print(f"  Needle: {needle_len} tokens at pos {needle_start}", flush=True)
    
    # 在后续各位置测量 PPL 差异
    # PPL of "needle-like" completion after long context
    check_positions = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    eval_len = 32
    
    print(f"\n  PPL difference (with_needle - without_needle):", flush=True)
    print(f"  {'Pos':>7} | {'PPL_with':>9} | {'PPL_without':>9} | {'Delta':>9} | {'Effect':>10}", flush=True)
    print(f"  " + "-" * 55, flush=True)
    
    for pos in check_positions:
        if pos + eval_len > len(ids_with): 
            pos = len(ids_with) - eval_len - 1
            if pos <= needle_start: continue
        
        ctx_with = torch.tensor([ids_with[:pos]], dtype=torch.long, device=device)
        ctx_without = torch.tensor([ids_without[:pos]], dtype=torch.long, device=device)
        tgt = torch.tensor(ids_with[pos:pos+eval_len], dtype=torch.long, device=device)
        
        with torch.no_grad():
            # with needle
            logits_w, h_w, _ = model(ctx_with, return_state=True, compute_critical_loss=False)
            loss_w = 0.0
            for i in range(len(tgt)):
                if i == 0: pred = logits_w[:, -1, :]
                else: pred, h_w = model.generate_step(torch.tensor([[tgt[i-1].item()]], device=device), h_w)
                loss_w += F.cross_entropy(pred, tgt[i:i+1], reduction='sum').item()
            ppl_w = math.exp(loss_w / eval_len) if loss_w / eval_len < 20 else 99999
            
            # without needle
            logits_wo, h_wo, _ = model(ctx_without, return_state=True, compute_critical_loss=False)
            loss_wo = 0.0
            for i in range(len(tgt)):
                if i == 0: pred = logits_wo[:, -1, :]
                else: pred, h_wo = model.generate_step(torch.tensor([[tgt[i-1].item()]], device=device), h_wo)
                loss_wo += F.cross_entropy(pred, tgt[i:i+1], reduction='sum').item()
            ppl_wo = math.exp(loss_wo / eval_len) if loss_wo / eval_len < 20 else 99999
        
        delta = ppl_w - ppl_wo
        effect = "↓ better" if delta < -1 else ("↑ worse" if delta > 1 else "→ same")
        print(f"  {pos:7d} | {ppl_w:9.1f} | {ppl_wo:9.1f} | {delta:+9.1f} | {effect:>10}", flush=True)
    
    # 整本小说速度
    print(f"\n  Processing full novel ({len(ids):,} tokens, chunked)...", flush=True)
    chunk_size = 4096
    h = [torch.zeros(1, model.d_model, device=device) for _ in range(model.num_scales)]
    torch.cuda.synchronize(); t0 = time.time()
    tokens_processed = 0
    for start in range(0, len(ids), chunk_size):
        end = min(start + chunk_size, len(ids))
        chunk = torch.tensor(ids[start:end]).unsqueeze(0).to(device)
        with torch.no_grad():
            _, h, _ = model(chunk, h_prev=h, return_state=True, compute_critical_loss=False)
        tokens_processed += (end - start)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"    {tokens_processed:,} tokens in {elapsed:.1f}s ({tokens_processed/elapsed:.0f} tok/s)", flush=True)
    for s in range(model.num_scales):
        print(f"    S{s} final norm: {h[s].norm(dim=-1).mean().item():.4f}", flush=True)

print(f"\nDone.", flush=True)
