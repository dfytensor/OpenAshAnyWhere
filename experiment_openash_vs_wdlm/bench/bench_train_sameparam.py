#!/usr/bin/env python3
"""
WDLM-Neural vs OpenASH  —  Same-Param Training Benchmark
同参数量对比: 每组匹配到 ±5% 参数量
  Group A ~60M:  WDLM(H=512,L=10) 60.3M  vs  OpenASH(H=640,L=10) 58.2M
  Group B ~85M:  WDLM(H=576,L=12) 82.3M  vs  OpenASH(H=768,L=12) 84.9M
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
PT_DATA = os.path.join(_ORIG_ROOT, "minimind_data", "pretrain_t2t_mini.jsonl")
SFT_DATA = os.path.join(_ORIG_ROOT, "minimind_data", "sft_t2t_mini.jsonl")

BATCH_SIZE = 8
SEQ_LEN = 512
STEPS = 200
LR = 3e-4
VS = 23005

CONFIGS = [
    {
        "name": "~60M",
        "wdlm": {"hidden_dim": 512, "num_layers": 10},
        "openash": {"hidden_size": 640, "num_heads": 8, "num_layers": 10},
    },
    {
        "name": "~85M",
        "wdlm": {"hidden_dim": 576, "num_layers": 12},
        "openash": {"hidden_size": 768, "num_heads": 8, "num_layers": 12},
    },
]


def load_data(path, voc, n_samples=5000):
    sp = {
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
                if 'conversations' in obj:
                    convs = obj['conversations']
                    ids = []
                    for msg in convs:
                        r = msg.get('role', '')
                        ct = msg.get('content', '')
                        if r == 'user':
                            ids += [sp["im_start"], sp["user"]] + voc.encode(ct) + [sp["im_end"]]
                        elif r == 'assistant':
                            ids += [sp["im_start"], sp["agent"]]
                            if msg.get('reasoning_content'):
                                ids += [sp["think_s"]] + voc.encode(msg['reasoning_content']) + [sp["think_e"]]
                            ids += voc.encode(ct) + [sp["im_end"]]
                else:
                    ids = voc.encode(obj.get('text', ''))
                if len(ids) >= 4:
                    samples.append(torch.tensor(ids[:SEQ_LEN + 1], dtype=torch.long))
                if len(samples) >= n_samples: break
            except: pass
    return samples


def collate_fn(items):
    padded = pad_sequence(items, batch_first=True, padding_value=0)
    return padded[:, :-1].clamp(0, 23004), padded[:, 1:].clamp(0, 23004)


def train_bench(model, loader, steps, tag, model_name):
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
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), t.reshape(-1), ignore_index=0)
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
    model.eval()
    return {
        "tok_per_s": total_tok / total_time,
        "avg_loss": avg_loss,
        "final_loss": final_loss,
        "peak_mem_mb": peak_mem,
        "total_time": total_time,
    }


def mem_bench(model, batch):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    model.train()
    x = torch.randint(1, 100, (batch, SEQ_LEN), device=DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = model(x, state=None)
        logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x.reshape(-1), ignore_index=0)
    scaler.scale(loss).backward()
    scaler.step(opt)
    return torch.cuda.max_memory_allocated() / 1024**2


# ============================================================
print("=" * 74)
print("  Same-Param Training Benchmark")
print(f"  batch={BATCH_SIZE}  seq={SEQ_LEN}  steps={STEPS}  lr={LR}")
print("=" * 74)

print("\n[0] Loading vocabulary + data...")
voc = OpenASHVoc(agent_voc_path=VOC_PATH)
all_data = load_data(PT_DATA, voc, n_samples=10000)
loader = DataLoader(all_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
print(f"  Data: {len(all_data)} samples")

# ============================================================
for cfg in CONFIGS:
    gname = cfg["name"]
    print(f"\n{'=' * 74}")
    print(f"  GROUP: {gname}")
    print(f"{'=' * 74}")

    # --- WDLM ---
    wc = cfg["wdlm"]
    wdlm = WaveDynamicsLanguageModel(VS, **wc)
    wp = sum(x.numel() for x in wdlm.parameters())
    print(f"\n  WDLM-Neural H={wc['hidden_dim']} L={wc['num_layers']}: {wp/1e6:.1f}M params")
    wdlm.to(DEV)
    r_w = train_bench(wdlm, loader, STEPS, gname, "WDLM")
    mem_w = mem_bench(wdlm, BATCH_SIZE)
    del wdlm; gc.collect(); torch.cuda.empty_cache()

    # --- OpenASH ---
    oc = cfg["openash"]
    oash = OpenASH(VS, **oc, model_flag="train")
    op = sum(x.numel() for x in oash.parameters())
    print(f"\n  OpenASH H={oc['hidden_size']} L={oc['num_layers']} heads={oc['num_heads']}: {op/1e6:.1f}M params")
    oash.to(DEV)
    r_o = train_bench(oash, loader, STEPS, gname, "OpenASH")
    mem_o = mem_bench(oash, BATCH_SIZE)
    del oash; gc.collect(); torch.cuda.empty_cache()

    # --- Comparison ---
    print(f"\n  --- {gname} Summary ---")
    print(f"  {'Metric':<25} {'WDLM':>14} {'OpenASH':>14} {'ratio':>8}")
    print(f"  {'-'*61}")
    print(f"  {'Params':<25} {wp/1e6:>12.1f}M {op/1e6:>12.1f}M {wp/op:>7.2f}")
    print(f"  {'Train tok/s':<25} {r_w['tok_per_s']:>14.0f} {r_o['tok_per_s']:>14.0f} {r_w['tok_per_s']/r_o['tok_per_s']:>7.2f}x")
    print(f"  {'Total time (s)':<25} {r_w['total_time']:>14.1f} {r_o['total_time']:>14.1f} {r_o['total_time']/r_w['total_time']:>7.2f}x")
    print(f"  {'Final loss':<25} {r_w['final_loss']:>14.4f} {r_o['final_loss']:>14.4f}")
    print(f"  {'Avg loss':<25} {r_w['avg_loss']:>14.4f} {r_o['avg_loss']:>14.4f}")
    print(f"  {'Peak train mem (MB)':<25} {r_w['peak_mem_mb']:>14.0f} {r_o['peak_mem_mb']:>14.0f} {r_w['peak_mem_mb']/r_o['peak_mem_mb']:>7.2f}")
    print(f"  {'Mem bs=8 (MB)':<25} {mem_w:>14.0f} {mem_o:>14.0f} {mem_w/mem_o:>7.2f}")

# ============================================================
print(f"\n{'=' * 74}")
print(f"  DONE")
print(f"{'=' * 74}")
