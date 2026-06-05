"""
WDLM-Neural+gen @ 60M — 全量 MiniMind 多轮预训练 + SFT + 生成测试 + PPL
  支持: torch.compile (auto), bf16 AMP, 梯度累积, 断点续训, 数据缓存, 多轮
"""
import torch, time, sys, math, os, json, argparse, gc, tempfile, shutil
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, 'F:/OpenASH2605')
sys.path.insert(0, 'F:/OpenASH2605/wdlm_verification')
os.chdir('F:/OpenASH2605')
from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from wdlm_neural import WaveDynamicsLanguageModel as WN

# ============================================================
# Config
# ============================================================
DATA_DIR = './minimind_data'
OUT_DIR = './train_60m'
CACHE_DIR = './train_60m/cache'

HIDDEN_DIM = 512
NUM_LAYERS = 10
PRETRAIN_SEQ = 512
SFT_SEQ = 768
BATCH_SIZE = 32
GRAD_ACCUM = 4
LR = 3e-5
WEIGHT_DECAY = 0.01
PRETRAIN_EPOCHS = 3
SFT_EPOCHS = 2
SAVE_EVERY = 500

def safe_save(obj, path):
    fd, tmp = tempfile.mkstemp(suffix='.tmp', dir=os.path.dirname(path))
    try:
        os.close(fd)
        torch.save(obj, tmp)
        if os.path.exists(path): os.remove(path)
        os.rename(tmp, path)
    except:
        if os.path.exists(tmp): os.remove(tmp)
        raise
LOG_EVERY = 20


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Data
# ============================================================
class CachedDataset(Dataset):
    def __init__(self, path, tok, seq_len, data_type='pretrain', cache_name=None, max_lines=None):
        self.tok = tok; self.seq_len = seq_len; self.data = []
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = f'{CACHE_DIR}/{cache_name or os.path.basename(path)}_{seq_len}.pt'

        if os.path.exists(cache_path):
            print(f'Loading cached data from {cache_path}')
            self.data = torch.load(cache_path, weights_only=False)
            if max_lines: self.data = self.data[:max_lines]
            print(f'Dataset: {len(self.data)} samples (from cache)')
            return

        print(f'Pre-tokenizing {path} (seq={seq_len})...')
        is_ = tok.token_to_id.get('<|im_start|>'); ie_ = tok.token_to_id.get('<|im_end|>')
        uid_ = tok.token_to_id.get('<|user|>'); aid_ = tok.token_to_id.get('<|agent|>')
        ts_ = tok.token_to_id.get('<|think|>'); te_ = tok.token_to_id.get('<|end_think|>')

        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
        if max_lines: lines = lines[:max_lines]
        total = len(lines)
        skipped = 0

        # Chunked saving: every 50K samples, save to disk and clear memory
        chunk_files = []
        buffer = []
        chunk_idx = 0

        for i, line in enumerate(lines):
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                if data_type == 'pretrain':
                    ids = tok.encode(obj.get('text', ''))
                else:
                    convs = obj.get('conversations', []); ids = []
                    for msg in convs:
                        r = msg.get('role', ''); ct = msg.get('content', '')
                        if r == 'user': ids += [is_, uid_] + tok.encode(ct) + [ie_]
                        elif r == 'assistant':
                            ids += [is_, aid_]
                            if msg.get('reasoning_content'): ids += [ts_] + tok.encode(msg['reasoning_content']) + [te_]
                            ids += tok.encode(ct) + [ie_]
                if len(ids) >= 4:
                    buffer.append(torch.tensor(ids[:seq_len+1], dtype=torch.long))
            except Exception:
                skipped += 1

            # Save chunk and clear memory
            if len(buffer) >= 50000:
                chunk_path = f'{cache_path}.chunk{chunk_idx}'
                torch.save(buffer, chunk_path)
                chunk_files.append(chunk_path)
                print(f'  ... {i+1}/{total} ({int((i+1)/total*100)}%) chunk{chunk_idx}: {len(buffer)} samples, {skipped} skipped', flush=True)
                buffer.clear(); chunk_idx += 1

        # Save remaining
        if buffer:
            chunk_path = f'{cache_path}.chunk{chunk_idx}'
            torch.save(buffer, chunk_path)
            chunk_files.append(chunk_path)

        # Merge all chunks into final cache
        print(f'Merging {len(chunk_files)} chunks...', flush=True)
        all_data = []
        for cf in chunk_files:
            all_data.extend(torch.load(cf, weights_only=False))
        torch.save(all_data, cache_path)
        for cf in chunk_files:
            os.remove(cf)
        self.data = all_data[:max_lines] if max_lines else all_data
        print(f'Done: {len(self.data)} samples, {skipped} skipped')

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i][:self.seq_len + 1]
    @staticmethod
    def collate(items):
        p = pad_sequence(items, batch_first=True, padding_value=0)
        return p[:, :-1], p[:, 1:]


# ============================================================
# Training
# ============================================================
def train(model, ds, dev, vs, tag, epochs, seq_len):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler()

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                        collate_fn=CachedDataset.collate, drop_last=True, pin_memory=True)
    steps_per_epoch = len(loader)
    total_steps = epochs * steps_per_epoch
    global_step = 0; best_loss = float('inf')

    # Resume
    ckp_path = f'{OUT_DIR}/wdlm60m_{tag}_latest.pth'
    if os.path.exists(ckp_path):
        ckp_data = open(ckp_path, 'rb')
        ckp = torch.load(ckp_data, map_location=dev)
        ckp_data.close(); del ckp_data
        model.load_state_dict(ckp['model']); opt.load_state_dict(ckp['optimizer'])
        scaler.load_state_dict(ckp['scaler'])
        global_step = ckp.get('step', 0); best_loss = ckp.get('best_loss', float('inf'))
        del ckp
        print(f'[Resume] {tag} from step {global_step}, best_loss={best_loss:.4f}')

    opt.zero_grad(set_to_none=True)
    t0 = time.time(); running_loss = 0.0
    start_epoch = global_step // steps_per_epoch
    print(f'[{tag}] {epochs} epochs, {steps_per_epoch} steps/epoch, total {total_steps} steps', flush=True)

    for epoch in range(start_epoch, epochs):
        it = iter(loader)

        for step_in_epoch in range(steps_per_epoch):
            if global_step >= total_steps: break

            for micro in range(GRAD_ACCUM):
                try: x, t = next(it)
                except StopIteration: it = iter(loader); x, t = next(it)
                x = x[:, :seq_len].to(dev, non_blocking=True)
                t = t[:, :seq_len].to(dev, non_blocking=True)
                x = x.clamp(0, vs - 1)
                t = t.clamp(0, vs - 1)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = model(x)
                    o = out[0] if isinstance(out, tuple) else out
                    loss = F.cross_entropy(o.reshape(-1, vs), t.reshape(-1), ignore_index=0) / GRAD_ACCUM
                scaler.scale(loss).backward()
                running_loss += loss.item() * GRAD_ACCUM

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            global_step += 1

            lr = LR * (0.1 + 0.45 * (1 + math.cos(math.pi * global_step / total_steps)))
            for pg in opt.param_groups: pg['lr'] = lr

            if global_step % LOG_EVERY == 0:
                avg = running_loss / LOG_EVERY / GRAD_ACCUM
                elapsed = time.time() - t0
                tok = global_step * GRAD_ACCUM * BATCH_SIZE * seq_len / elapsed
                print(f'  [{tag}] e{epoch+1}/{epochs} s{global_step:>6d}/{total_steps} '
                      f'loss={avg:.4f} lr={lr:.2e} {tok:.0f}tok/s', flush=True)
                running_loss = 0.0
                if avg < best_loss: best_loss = avg

            if global_step % SAVE_EVERY == 0:
                safe_save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                            'scaler': scaler.state_dict(), 'step': global_step, 'best_loss': best_loss}, ckp_path)
                print(f'  [Save] step {global_step}', flush=True)

        safe_save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                    'scaler': scaler.state_dict(), 'step': global_step, 'best_loss': best_loss}, ckp_path)
        print(f'  [{tag}] EPOCH {epoch+1}/{epochs} complete', flush=True)

    final_path = f'{OUT_DIR}/wdlm60m_{tag}_final.pth'
    safe_save({'model': model.state_dict(), 'step': global_step, 'best_loss': best_loss}, final_path)
    print(f'[{tag}] Final: {final_path}')
    return model


# ============================================================
# Generation + PPL
# ============================================================
@torch.no_grad()
def test_generation(model, voc, dev):
    model.eval()
    for prompt_text in ['人工智能技术的未来发展趋势包括', '请解释一下量子计算的基本原理', '写一首关于春天的诗']:
        ids = torch.tensor([voc.encode(prompt_text)], dtype=torch.long).to(dev)
        state = None; generated = ids
        for _ in range(80):
            ctx = generated[:, -SFT_SEQ:] if generated.size(1) > SFT_SEQ else generated
            out, state = model(ctx, state)
            logits = out[:, -1, :] / 0.8
            v, _ = torch.topk(logits, 40)
            logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
            probs = F.softmax(logits, dim=-1)
            nt = torch.multinomial(probs, 1)
            generated = torch.cat([generated, nt], dim=1)
            if (nt == 2).any(): break
        print(f'\n[Prompt] {prompt_text}')
        print(f'[Output] {voc.decode(generated[0].tolist())[:300]}\n')


@torch.no_grad()
def calc_ppl(model, data_path, voc, dev, vs, seq_len, n_samples=200):
    model.eval()
    is_ = voc.token_to_id.get('<|im_start|>'); ie_ = voc.token_to_id.get('<|im_end|>')
    uid_ = voc.token_to_id.get('<|user|>'); aid_ = voc.token_to_id.get('<|agent|>')
    ts_ = voc.token_to_id.get('<|think|>'); te_ = voc.token_to_id.get('<|end_think|>')
    with open(data_path, encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines()[-5000:] if l.strip()]
    total_loss = 0; total_tokens = 0
    for line in lines[:n_samples]:
        convs = json.loads(line).get('conversations', []); ids = []
        for msg in convs:
            r = msg.get('role', ''); ct = msg.get('content', '')
            if r == 'user': ids += [is_, uid_] + voc.encode(ct) + [ie_]
            elif r == 'assistant':
                ids += [is_, aid_]
                if msg.get('reasoning_content'): ids += [ts_] + voc.encode(msg['reasoning_content']) + [te_]
                ids += voc.encode(ct) + [ie_]
        if len(ids) < 4: continue
        ids = ids[:seq_len+1]; t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(dev)
        out = model(t[:, :-1])
        o = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(o.view(-1, vs), t[:, 1:].view(-1), ignore_index=0)
        total_loss += loss.item() * (t[:, 1:] != 0).sum().item()
        total_tokens += (t[:, 1:] != 0).sum().item()
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_epochs', type=int, default=PRETRAIN_EPOCHS)
    parser.add_argument('--sft_epochs', type=int, default=SFT_EPOCHS)
    parser.add_argument('--skip_pretrain', action='store_true')
    parser.add_argument('--skip_sft', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--compile', type=int, default=1, choices=[0, 1])
    parser.add_argument('--max_lines_pretrain', type=int, default=0)
    parser.add_argument('--max_lines_sft', type=int, default=0)
    args = parser.parse_args()

    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUT_DIR, exist_ok=True)
    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    vs = len(voc.token_to_id) + 1
    print(f'Device: {dev} | Vocab: {vs}')

    torch.manual_seed(42)
    model = WN(vs, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(dev)
    print(f'Model: WDLM-Neural+gen H={HIDDEN_DIM} L={NUM_LAYERS} = {count_params(model):,} params')

    compile_ok = False
    if args.compile and hasattr(torch, 'compile'):
        try: model = torch.compile(model); compile_ok = True; print('torch.compile: ON')
        except Exception as e: print(f'torch.compile: skipped ({e})')
    else: print('torch.compile: OFF')

    if args.test_only:
        pt_path = f'{OUT_DIR}/wdlm60m_pretrain_final.pth'
        if os.path.exists(pt_path):
            model.load_state_dict(torch.load(pt_path, map_location=dev)['model'])
        test_generation(model, voc, dev)
        sys.exit(0)

    # === Pretrain ===
    if not args.skip_pretrain:
        print(f'\n{"="*60}\n  Pretrain seq={PRETRAIN_SEQ} epochs={args.pretrain_epochs}\n{"="*60}')
        pt_ds = CachedDataset(f'{DATA_DIR}/pretrain_t2t_mini.jsonl', voc, PRETRAIN_SEQ,
                              'pretrain', 'pt_cache', args.max_lines_pretrain or None)
        model = train(model, pt_ds, dev, vs, 'pretrain', args.pretrain_epochs, PRETRAIN_SEQ)
        gc.collect(); torch.cuda.empty_cache()

    # === SFT ===
    if not args.skip_sft:
        print(f'\n{"="*60}\n  SFT seq={SFT_SEQ} epochs={args.sft_epochs}\n{"="*60}')
        if args.skip_pretrain and os.path.exists(f'{OUT_DIR}/wdlm60m_pretrain_final.pth'):
            model.load_state_dict(torch.load(f'{OUT_DIR}/wdlm60m_pretrain_final.pth', map_location=dev)['model'])
        sft_ds = CachedDataset(f'{DATA_DIR}/sft_t2t_mini.jsonl', voc, SFT_SEQ,
                                'sft', 'sft_cache', args.max_lines_sft or None)
        model = train(model, sft_ds, dev, vs, 'sft', args.sft_epochs, SFT_SEQ)
        gc.collect(); torch.cuda.empty_cache()

    # === PPL + Generation ===
    print(f'\n{"="*60}\n  Evaluation\n{"="*60}')
    ppl = calc_ppl(model, f'{DATA_DIR}/sft_t2t_mini.jsonl', voc, dev, vs, SFT_SEQ, n_samples=100)
    print(f'  PPL: {ppl:.2f}')
    test_generation(model, voc, dev)
