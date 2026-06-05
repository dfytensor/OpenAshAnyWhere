#!/usr/bin/env python3
"""
WDLM-Neural 60M  vs  OpenASH 84M  —  Training Speed Benchmark
同规模对比: 相同数据、相同 batch_size、相同 seq_len、相同优化器配置
"""
import os, sys, time, json, gc, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
_ORIG_ROOT = r"F:\OpenASH2605"

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src_openash'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src_wdlm'))
os.chdir(_ORIG_ROOT)

from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from wdlm_neural import WaveDynamicsLanguageModel
from open_ash import OpenASH

DEV = "cuda" if torch.cuda.is_available() else "cpu"
VOC_PATH = os.path.join(SCRIPT_DIR, "open_ash_voc_agent.json")
OPENASH_WEIGHT = os.path.join(SCRIPT_DIR, "full_sft_768_12.pth")
WDLM_WEIGHT = os.path.join(SCRIPT_DIR, "wdlm60m_sft_final.pth")
PT_DATA = os.path.join(_ORIG_ROOT, "minimind_data", "pretrain_t2t_mini.jsonl")
SFT_DATA = os.path.join(_ORIG_ROOT, "minimind_data", "sft_t2t_mini.jsonl")

BATCH_SIZE = 8
SEQ_LEN = 512
STEPS = 200
LR = 3e-4


def load_data(path, voc, data_type, n_samples=5000):
    sp_ids = {
        "im_start": voc.token_to_id.get('<|im_start|>', 1),
        "im_end": voc.token_to_id.get('<|im_end|>', 2),
        "user": 5,
        "agent": voc.token_to_id.get('<|agent|>', 6),
        "think_s": voc.token_to_id.get('<|think|>', 3),
        "think_e": voc.token_to_id.get('<|end_think|>', 4),
    }
    samples = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                if data_type == 'pretrain':
                    ids = voc.encode(obj.get('text', ''))
                else:
                    convs = obj.get('conversations', [])
                    ids = []
                    for msg in convs:
                        r = msg.get('role', '')
                        ct = msg.get('content', '')
                        if r == 'user':
                            ids += [sp_ids["im_start"], sp_ids["user"]] + voc.encode(ct) + [sp_ids["im_end"]]
                        elif r == 'assistant':
                            ids += [sp_ids["im_start"], sp_ids["agent"]]
                            if msg.get('reasoning_content'):
                                ids += [sp_ids["think_s"]] + voc.encode(msg['reasoning_content']) + [sp_ids["think_e"]]
                            ids += voc.encode(ct) + [sp_ids["im_end"]]
                if len(ids) >= 4:
                    samples.append(torch.tensor(ids[:SEQ_LEN + 1], dtype=torch.long))
                if len(samples) >= n_samples: break
            except: pass
    return samples


def collate_fn(items):
    padded = pad_sequence(items, batch_first=True, padding_value=0)
    return padded[:, :-1].clamp(0, 23004), padded[:, 1:].clamp(0, 23004)


def train_bench(model, loader, vs, steps, tag, model_name):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler()

    losses = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    it = iter(loader)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        try:
            x, t = next(it)
        except StopIteration:
            it = iter(loader)
            x, t = next(it)
        x = x[:, :SEQ_LEN].to(DEV)
        t = t[:, :SEQ_LEN].to(DEV)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out_raw = model(x, state=None)
            logits = out_raw[0] if isinstance(out_raw, tuple) else out_raw
            loss = F.cross_entropy(logits.reshape(-1, vs), t.reshape(-1), ignore_index=0)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        losses.append(loss.item())
        if step % 50 == 0:
            elapsed = time.perf_counter() - t0
            tok = step * BATCH_SIZE * SEQ_LEN
            avg = sum(losses[-50:]) / 50
            print(f"    [{model_name}] {tag} step {step:>3d}/{steps} loss={avg:.4f} "
                  f"{tok/elapsed:.0f} tok/s  {elapsed:.1f}s")

    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0
    total_tok = steps * BATCH_SIZE * SEQ_LEN
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    avg_loss = sum(losses) / len(losses)
    final_loss = sum(losses[-50:]) / min(50, len(losses))
    tok_per_s = total_tok / total_time

    model.eval()
    return {
        "total_time": total_time,
        "tok_per_s": tok_per_s,
        "avg_loss": avg_loss,
        "final_loss": final_loss,
        "peak_mem_mb": peak_mem,
        "losses": losses,
    }


# ============================================================
print("=" * 70)
print("  Training Speed Benchmark  (same data, same config)")
print(f"  batch={BATCH_SIZE}  seq={SEQ_LEN}  steps={STEPS}  lr={LR}")
print("=" * 70)

# --- Vocab ---
print("\n[0] Loading vocabulary...")
voc = OpenASHVoc(agent_voc_path=VOC_PATH)
vs = len(voc.token_to_id) + 1
print(f"  Vocab size: {vs}")

# --- Data ---
print("\n[1] Loading data...")
pt_data = load_data(PT_DATA, voc, 'pretrain', n_samples=5000)
sft_data = load_data(SFT_DATA, voc, 'sft', n_samples=5000)
print(f"  Pretrain: {len(pt_data)} samples  |  SFT: {len(sft_data)} samples")

pt_loader = DataLoader(pt_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
sft_loader = DataLoader(sft_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)

# ============================================================
# TEST 1: Pretrain Speed
# ============================================================
print("\n" + "=" * 70)
print("  TEST 1: Pretrain Training Speed")
print("=" * 70)

print("\n  --- OpenASH (H=768, L=12) ---")
openash = OpenASH(voc_size=vs, hidden_size=768, num_heads=8, num_layers=12, model_flag="train")
openash.to(DEV).train()
r_pt_o = train_bench(openash, pt_loader, vs, STEPS, "pretrain", "OpenASH")
del openash; gc.collect(); torch.cuda.empty_cache()

print("\n  --- WDLM-Neural (H=512, L=10) ---")
wdlm = WaveDynamicsLanguageModel(vs, hidden_dim=512, num_layers=10)
wdlm.to(DEV).train()
r_pt_w = train_bench(wdlm, pt_loader, vs, STEPS, "pretrain", "WDLM")
del wdlm; gc.collect(); torch.cuda.empty_cache()

print(f"\n  Pretrain Summary:")
print(f"  {'Model':<20} {'tok/s':>10} {'Avg Loss':>10} {'Final Loss':>12} {'Peak Mem':>10} {'Time':>8}")
print(f"  {'-'*70}")
print(f"  {'OpenASH 768/12':<20} {r_pt_o['tok_per_s']:>10.0f} {r_pt_o['avg_loss']:>10.4f} "
      f"{r_pt_o['final_loss']:>12.4f} {r_pt_o['peak_mem_mb']:>8.0f}MB {r_pt_o['total_time']:>6.1f}s")
print(f"  {'WDLM-Neural 512/10':<20} {r_pt_w['tok_per_s']:>10.0f} {r_pt_w['avg_loss']:>10.4f} "
      f"{r_pt_w['final_loss']:>12.4f} {r_pt_w['peak_mem_mb']:>8.0f}MB {r_pt_w['total_time']:>6.1f}s")

# ============================================================
# TEST 2: SFT Speed
# ============================================================
print("\n" + "=" * 70)
print("  TEST 2: SFT Training Speed")
print("=" * 70)

print("\n  --- OpenASH (H=768, L=12) ---")
openash = OpenASH(voc_size=vs, hidden_size=768, num_heads=8, num_layers=12, model_flag="train")
openash.load_state_dict(torch.load(OPENASH_WEIGHT, map_location=DEV), strict=False)
openash.to(DEV).train()
r_sft_o = train_bench(openash, sft_loader, vs, STEPS, "sft", "OpenASH")
del openash; gc.collect(); torch.cuda.empty_cache()

print("\n  --- WDLM-Neural (H=512, L=10) ---")
wdlm = WaveDynamicsLanguageModel(vs, hidden_dim=512, num_layers=10)
ckp = torch.load(WDLM_WEIGHT, map_location=DEV)
wdlm.load_state_dict(ckp['model'] if 'model' in ckp else ckp)
wdlm.to(DEV).train()
r_sft_w = train_bench(wdlm, sft_loader, vs, STEPS, "sft", "WDLM")
del wdlm; gc.collect(); torch.cuda.empty_cache()

print(f"\n  SFT Summary:")
print(f"  {'Model':<20} {'tok/s':>10} {'Avg Loss':>10} {'Final Loss':>12} {'Peak Mem':>10} {'Time':>8}")
print(f"  {'-'*70}")
print(f"  {'OpenASH 768/12':<20} {r_sft_o['tok_per_s']:>10.0f} {r_sft_o['avg_loss']:>10.4f} "
      f"{r_sft_o['final_loss']:>12.4f} {r_sft_o['peak_mem_mb']:>8.0f}MB {r_sft_o['total_time']:>6.1f}s")
print(f"  {'WDLM-Neural 512/10':<20} {r_sft_w['tok_per_s']:>10.0f} {r_sft_w['avg_loss']:>10.4f} "
      f"{r_sft_w['final_loss']:>12.4f} {r_sft_w['peak_mem_mb']:>8.0f}MB {r_sft_w['total_time']:>6.1f}s")

# ============================================================
# TEST 3: GPU Memory (Training vs Inference)
# ============================================================
print("\n" + "=" * 70)
print("  TEST 3: GPU Memory — Training vs Inference")
print("=" * 70)

def mem_test(model, vs, batch, seq, tag):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    model.train()
    x = torch.randint(1, 100, (batch, seq), device=DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = model(x, state=None)
        logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(logits.reshape(-1, vs), x.reshape(-1), ignore_index=0)
    scaler.scale(loss).backward()
    scaler.step(opt)
    train_mem = torch.cuda.max_memory_allocated() / 1024**2

    torch.cuda.reset_peak_memory_stats()
    model.eval()
    with torch.no_grad():
        _ = model(x, state=None)
    infer_mem = torch.cuda.max_memory_allocated() / 1024**2
    return train_mem, infer_mem

print(f"  {'Config':<35} {'Train Mem':>10} {'Infer Mem':>10} {'Overhead':>10}")
print(f"  {'-'*65}")

openash = OpenASH(voc_size=vs, hidden_size=768, num_heads=8, num_layers=12, model_flag="train")
openash.load_state_dict(torch.load(OPENASH_WEIGHT, map_location=DEV), strict=False)
openash.to(DEV)
for bs in [1, 4, 8]:
    tm, im = mem_test(openash, vs, bs, SEQ_LEN, "OpenASH")
    print(f"  OpenASH batch={bs} seq={SEQ_LEN}   {tm:>8.0f}MB  {im:>8.0f}MB  {(tm-im):>8.0f}MB")
del openash; gc.collect(); torch.cuda.empty_cache()

wdlm = WaveDynamicsLanguageModel(vs, hidden_dim=512, num_layers=10)
ckp = torch.load(WDLM_WEIGHT, map_location=DEV)
wdlm.load_state_dict(ckp['model'] if 'model' in ckp else ckp)
wdlm.to(DEV)
for bs in [1, 4, 8]:
    tm, im = mem_test(wdlm, vs, bs, SEQ_LEN, "WDLM")
    print(f"  WDLM   batch={bs} seq={SEQ_LEN}   {tm:>8.0f}MB  {im:>8.0f}MB  {(tm-im):>8.0f}MB")
del wdlm; gc.collect(); torch.cuda.empty_cache()

# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 70)
print("  TRAINING BENCHMARK SUMMARY")
print("=" * 70)
print(f"  Config: batch={BATCH_SIZE}  seq={SEQ_LEN}  steps={STEPS}  lr={LR}")
print(f"  {'Metric':<30} {'OpenASH 768/12':>18} {'WDLM 512/10':>18}")
print(f"  {'-'*66}")
print(f"  {'Pretrain tok/s':<30} {r_pt_o['tok_per_s']:>18.0f} {r_pt_w['tok_per_s']:>18.0f}")
print(f"  {'Pretrain final loss':<30} {r_pt_o['final_loss']:>18.4f} {r_pt_w['final_loss']:>18.4f}")
print(f"  {'Pretrain peak mem (MB)':<30} {r_pt_o['peak_mem_mb']:>18.0f} {r_pt_w['peak_mem_mb']:>18.0f}")
print(f"  {'SFT tok/s':<30} {r_sft_o['tok_per_s']:>18.0f} {r_sft_w['tok_per_s']:>18.0f}")
print(f"  {'SFT final loss':<30} {r_sft_o['final_loss']:>18.4f} {r_sft_w['final_loss']:>18.4f}")
print(f"  {'SFT peak mem (MB)':<30} {r_sft_o['peak_mem_mb']:>18.0f} {r_sft_w['peak_mem_mb']:>18.0f}")
print(f"  {'Train speed ratio':<30} {'1.00x':>18} {r_pt_w['tok_per_s']/r_pt_o['tok_per_s']:>17.2f}x")
print(f"{'=' * 70}")
