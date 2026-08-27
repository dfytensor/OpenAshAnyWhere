"""
OpenASH 30M with cap+decay — 集成 state norm cap + decay 的训练脚本
  模型: OpenASH H=432 L=8 heads=8 ≈ 30M
  Cap: 每个 chunk 后对 cummax state 做范数截断 (train时集成，推理时无需干预)
  Decay: 每个 chunk 后对 state 乘以衰减系数 (抑制长程累积)

用法:
  python train_30m_cap_decay/train.py --pretrain_epochs 3 --sft_epochs 2 --compile 0
  python train_30m_cap_decay/train.py --skip_pretrain --sft_epochs 2
  python train_30m_cap_decay/train.py --test_only
"""
import torch, time, sys, math, os, json, argparse, gc, tempfile
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, 'F:/OpenASH2605')
os.chdir('F:/OpenASH2605')
from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from open_ash import OpenASH
from open_ash_infer import _sp

DATA_DIR = './minimind_data'
OUT_DIR = './train_30m_cap_decay'
CACHE_DIR = './train_30m_cap_decay/cache'

HIDDEN_SIZE = 432
NUM_LAYERS = 8
NUM_HEADS = 8
PRETRAIN_SEQ = 512
SFT_SEQ = 768
BATCH_SIZE = 32
GRAD_ACCUM = 4
LR = 3e-5
WEIGHT_DECAY = 0.01
PRETRAIN_EPOCHS = 3
SFT_EPOCHS = 2
SAVE_EVERY = 500
LOG_EVERY = 20
CHUNK = 64

STATE_CAP = 150
STATE_DECAY = 0.97


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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
        uid_ = tok.token_to_id.get('_eval'); aid_ = tok.token_to_id.get('<|agent|>')
        ts_ = tok.token_to_id.get('<|think|>'); te_ = tok.token_to_id.get('<|end_think|>')

        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
        if max_lines: lines = lines[:max_lines]
        total = len(lines)
        skipped = 0

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
                ids += voc.encode(ct) + [ie_]
                if len(ids) >= 4:
                    buffer.append(torch.tensor(ids[:seq_len+1], dtype=torch.long))
            except Exception:
                skipped += 1

            if len(buffer) >= 50000:
                chunk_path = f'{cache_path}.chunk{chunk_idx}'
                torch.save(buffer, chunk_path)
                chunk_files.append(chunk_path)
                print(f'  ... {i+1}/{total} ({int((i+1)/total*100)}%) chunk{chunk_idx}: {len(buffer)} samples, {skipped} skipped', flush=True)
                buffer.clear(); chunk_idx += 1

        if buffer:
            chunk_path = f'{cache_path}.chunk{chunk_idx}'
            torch.save(buffer, chunk_path)
            chunk_files.append(chunk_path)

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


def apply_cap_decay(state, n_layers):
    """Apply state norm cap + decay to cummax states after each chunk."""
    for i in range(n_layers):
        if state[i] is not None:
            s = state[i]
            sn = s.norm()
            if sn > STATE_CAP:
                s = s * (STATE_CAP / sn)
            state[i] = s * STATE_DECAY


def train(model, ds, dev, vs, tag, epochs, seq_len):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler()

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                        collate_fn=CachedDataset.collate, drop_last=True, pin_memory=True)
    steps_per_epoch = len(loader)
    total_steps = epochs * steps_per_epoch
    n_layers = len(model.decoder_layers)

    global_step = 0; best_loss = float('inf')

    ckp_path = f'{OUT_DIR}/openash30m_cd_{tag}_latest.pth'
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
    print(f'[{tag}] state_cap={STATE_CAP}, state_decay={STATE_DECAY}, chunk={CHUNK}', flush=True)

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
                    states = [None] * n_layers
                    chunk_logits = []
                    for c0 in range(0, x.size(1), CHUNK):
                        c = x[:, c0:c0+CHUNK]
                        h = model.em(c)
                        for i, layer in enumerate(model.decoder_layers):
                            h2, s = layer(h, states[i])
                            h = h2 + h
                            states[i] = s
                        apply_cap_decay(states, n_layers)
                        chunk_logits.append(model.head_score(h))
                    logits = torch.cat(chunk_logits, dim=1)
                    loss = F.cross_entropy(logits.reshape(-1, vs), t.reshape(-1), ignore_index=0) / GRAD_ACCUM
                scaler.scale(loss).backward()
                running_loss += loss.item() * GRAD_ACCUM

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            global_step += 1

            progress = global_step / total_steps
            lr = LR * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))
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

    final_path = f'{OUT_DIR}/openash30m_cd_{tag}_final.pth'
    safe_save({'model': model.state_dict(), 'step': global_step, 'best_loss': best_loss}, final_path)
    print(f'[{tag}] Final: {final_path}')
    return model


@torch.no_grad()
def test_generation(model, voc, dev):
    model.eval()
    n_layers = len(model.decoder_layers)
    for prompt_text in ['人工智能技术的未来发展趋势包括', '请解释一下量子计算的基本原理', '写一首关于春天的诗']:
        ids = torch.tensor([voc.encode(prompt_text)], dtype=torch.long).to(dev)
        states = [None] * n_layers; generated = ids
        for _ in range(80):
            ctx = generated[:, -SFT_SEQ:] if generated.size(1) > SFT_SEQ else generated
            c = ctx[:, -CHUNK:] if ctx.size(1) > CHUNK else ctx
            h = model.em(c)
            for i, layer in enumerate(model.decoder_layers):
                h2, s = layer(h, states[i])
                h = h2 + h
                states[i] = s
            apply_cap_decay(states, n_layers)
            logits = model.head_score(h)[:, -1, :] / 0.8
            v, _ = torch.topk(logits, 40)
            logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
            probs = F.softmax(logits, dim=-1)
            nt = torch.multinomial(probs, 1)
            generated = torch.cat([generated, nt], dim=1)
            if (nt == 2).any(): break
        print(f'\n[Prompt] {prompt_text}')
        print(f'[Output] {voc.decode(generated[0].tolist())[:300]}\n')


@torch.no_grad()
def calc_ppl(model, data_path, voc, dev, vs, seq_len, n_samples=200, use_cap_decay=True):
    model.eval()
    n_layers = len(model.decoder_layers)
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
        x = t[:, :-1].clamp(0, vs-1)
        states = [None] * n_layers
        cl = []
        for c0 in range(0, x.size(1), CHUNK):
            c = x[:, c0:c0+CHUNK]
            h = model.em(c)
            for i, layer in enumerate(model.decoder_layers):
                h2, s = layer(h, states[i])
                h = h2 + h
                states[i] = s
            if use_cap_decay:
                apply_cap_decay(states, n_layers)
            cl.append(model.head_score(h))
        clo = torch.cat(cl, dim=1)
        loss = F.cross_entropy(clo.view(-1, vs), t[:, 1:].clamp(0, vs-1).view(-1), ignore_index=0)
        total_loss += loss.item() * (t[:, 1:] != 0).sum().item()
        total_tokens += (t[:, 1:] != 0).sum().item()
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')


@torch.no_grad()
def extrap_test(model, data_path, voc, dev, vs):
    model.eval()
    n_layers = len(model.decoder_layers)
    sp = _sp(voc)
    all_ids = []
    with open(data_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                convs = obj.get('conversations', []); ids = []
                for msg in convs:
                    r = msg.get('role', ''); ct = msg.get('content', '')
                    if r == 'user': ids += [sp['im_start'], sp['user']] + voc.encode(ct) + [sp['im_end']]
                    elif r == 'assistant': ids += [sp['im_start'], sp['agent']] + voc.encode(ct) + [sp['im_end']]
                if ids: all_ids.extend(ids)
            except: pass
            if len(all_ids) >= 16384: break

    print(f'\n{"="*60}')
    print(f'  Extrapolation Test (cap={STATE_CAP}, decay={STATE_DECAY})')
    print(f'  Tokens available: {len(all_ids)}')
    print(f'{"="*60}')
    print(f'  {"Seq":>7}  {"PPL":>10}')
    print(f'  {"-"*22}')

    for sl in [512, 1024, 2048, 4096, 8192, 16384]:
        if sl > len(all_ids): continue
        s = all_ids[:sl]
        x = torch.tensor([s[:-1]], dtype=torch.long).to(dev).clamp(0, vs-1)
        t = torch.tensor([s[1:]], dtype=torch.long).to(dev).clamp(0, vs-1)
        states = [None] * n_layers
        cl = []
        for c0 in range(0, x.size(1), CHUNK):
            c = x[:, c0:c0+CHUNK]
            h = model.em(c)
            for i, layer in enumerate(model.decoder_layers):
                h2, s = layer(h, states[i])
                h = h2 + h
                states[i] = s
            apply_cap_decay(states, n_layers)
            cl.append(model.head_score(h))
        clo = torch.cat(cl, dim=1)
        nll = F.cross_entropy(clo.reshape(-1, clo.size(-1)), t.reshape(-1), ignore_index=0, reduction="sum").item()
        ntok = max((t != 0).sum().item(), 1)
        ppl = math.exp(nll / ntok)
        label = f'{sl//1024}K' if sl >= 1024 else str(sl)
        print(f'  {label:>7}  {ppl:>10.1f}')
        sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_epochs', type=int, default=PRETRAIN_EPOCHS)
    parser.add_argument('--sft_epochs', type=int, default=SFT_EPOCHS)
    parser.add_argument('--skip_pretrain', action='store_true')
    parser.add_argument('--skip_sft', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--compile', type=int, default=0, choices=[0, 1])
    parser.add_argument('--max_lines_pretrain', type=int, default=0)
    parser.add_argument('--max_lines_sft', type=int, default=0)
    parser.add_argument('--state_cap', type=float, default=STATE_CAP)
    parser.add_argument('--state_decay', type=float, default=STATE_DECAY)
    args = parser.parse_args()

    STATE_CAP = args.state_cap
    STATE_DECAY = args.state_decay

    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUT_DIR, exist_ok=True)
    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    vs = len(voc.token_to_id) + 1
    print(f'Device: {dev} | Vocab: {vs}')
    print(f'State cap: {STATE_CAP} | State decay: {STATE_DECAY}')

    torch.manual_seed(42)
    model = OpenASH(voc_size=vs, hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS,
                    num_layers=NUM_LAYERS, model_flag="train").to(dev)
    print(f'Model: OpenASH H={HIDDEN_SIZE} L={NUM_LAYERS} heads={NUM_HEADS} = {count_params(model):,} params')
    print(f'torch.compile: {"ON" if args.compile else "OFF"}')

    if args.compile and hasattr(torch, 'compile'):
        try: model = torch.compile(model); print('torch.compile: ON')
        except Exception as e: print(f'torch.compile: skipped ({e})')

    if args.test_only:
        ft_path = f'{OUT_DIR}/openash30m_cd_sft_final.pth'
        if os.path.exists(ft_path):
            model.load_state_dict(torch.load(ft_path, map_location=dev)['model'])
        test_generation(model, voc, dev)
        sys.exit(0)

    if not args.skip_pretrain:
        print(f'\n{"="*60}\n  Pretrain seq={PRETRAIN_SEQ} epochs={args.pretrain_epochs}\n{"="*60}')
        pt_ds = CachedDataset(f'{DATA_DIR}/pretrain_t2t_mini.jsonl', voc, PRETRAIN_SEQ,
                              'pretrain', 'pt_cache', args.max_lines_pretrain or None)
        model = train(model, pt_ds, dev, vs, 'pretrain', args.pretrain_epochs, PRETRAIN_SEQ)
        gc.collect(); torch.cuda.empty_cache()

    if not args.skip_sft:
        print(f'\n{"="*60}\n  SFT seq={SFT_SEQ} epochs={args.sft_epochs}\n{"="*60}')
        if args.skip_pretrain and os.path.exists(f'{OUT_DIR}/openash30m_cd_pretrain_final.pth'):
            model.load_state_dict(torch.load(f'{OUT_DIR}/openash30m_cd_pretrain_final.pth', map_location=dev)['model'])
        sft_ds = CachedDataset(f'{DATA_DIR}/sft_t2t_mini.jsonl', voc, SFT_SEQ,
                                'sft', 'sft_cache', args.max_lines_sft or None)
        model = train(model, sft_ds, dev, vs, 'sft', args.sft_epochs, SFT_SEQ)
        gc.collect(); torch.cuda.empty_cache()

    print(f'\n{"="*60}\n  Evaluation\n{"="*60}')
    test_generation(model, voc, dev)
    extrap_test(model, f'{DATA_DIR}/sft_t2t_mini.jsonl', voc, dev, vs)
    print(f'\nDone! Weights in {OUT_DIR}/')


