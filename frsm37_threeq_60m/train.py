"""
FRSMASH v3.7 @ 60M — 三问过滤数据预训练 + SFT
=============================================
模型: FRSMASHv37 (DirectAdd) H=448 L=7 heads=8 = 58.9M  (openashvoc 编码, vs=23005)
数据: minimind 经「三问过滤器」处理 (C类≤20%), 见 filtered_data/
对比: train_baseline.py 用未过滤数据训练同架构模型作对照.

Windows 运行需 PYTHONUTF8=1 (fla 编码). 本脚本开头已强制设置。
"""
import os
import sys
import gc
import time
import json
import math
import tempfile
import argparse

# 强制 UTF-8 模式 (fla 在 Windows 上读模板需 utf-8, 否则 GBK 报错)
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONUTF8', '1')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'F:\OpenASH2605')

from open_ash_voc import OpenASHVoc
from model import FRSMASHv37

# ============================================================
# Config
# ============================================================
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'filtered_data')
OUT_DIR = os.path.join(HERE, 'checkpoints')
CACHE_DIR = os.path.join(HERE, 'cache')

HIDDEN = 448
LAYERS = 7
HEADS = 8
VOC_TOKENS = 23005          # OpenASHVoc 词表
VS = VOC_TOKENS + 1         # +1 边界

PRETRAIN_SEQ = 512
SFT_SEQ = 768
BATCH = 64                  # 预训练: 4090 余量充足 (18.7GB)
GRAD_ACCUM = 2              # 有效批 128
SFT_BATCH = 16              # SFT seq 长, batch 64×768=27.5GB 会爆显存; 16 仅 7.5GB
SFT_GRAD_ACCUM = 8          # 有效批 128
LR = 3e-4
WD = 0.01
WARMUP = 200
SAVE_EVERY = 1000
LOG_EVERY = 20


def safe_save(obj, path):
    fd, tmp = tempfile.mkstemp(suffix='.tmp', dir=os.path.dirname(path))
    try:
        os.close(fd)
        torch.save(obj, tmp)
        if os.path.exists(path):
            os.remove(path)
        os.rename(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# 并行分词 worker (jieba 是瓶颈, 多进程加速)
# ============================================================
_TOK = None
_TOK_SPECS = None


def _init_worker(agent_voc_path, specs):
    global _TOK, _TOK_SPECS
    from open_ash_voc import OpenASHVoc
    _TOK = OpenASHVoc(agent_voc_path=agent_voc_path)
    _TOK_SPECS = specs


def _tok_one(args):
    """解析+分词一行, 返回 token id list 或 None."""
    line, data_type, seq_len = args
    global _TOK, _TOK_SPECS
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
        is_, ie_, uid_, aid_, ts_, te_ = _TOK_SPECS
        if data_type == 'pretrain':
            ids = _TOK.encode(obj.get('text', ''))
        else:
            convs = obj.get('conversations', [])
            ids = []
            for msg in convs:
                r = msg.get('role', '')
                ct = msg.get('content', '')
                if r == 'user':
                    ids += [is_, uid_] + _TOK.encode(ct) + [ie_]
                elif r == 'assistant':
                    ids += [is_, aid_]
                    if msg.get('reasoning_content'):
                        ids += [ts_] + _TOK.encode(msg['reasoning_content']) + [te_]
                    ids += _TOK.encode(ct) + [ie_]
        if len(ids) >= 4:
            return ids[:seq_len + 1]
    except Exception:
        return None
    return None


# ============================================================
# CachedDataset — 预分词缓存 (pretrain 用 text, sft 用 conversations)
# ============================================================
class CachedDataset(Dataset):
    def __init__(self, path, tok, seq_len, data_type='pretrain', cache_name=None, max_lines=None):
        self.tok = tok
        self.seq_len = seq_len
        self.data = []
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(CACHE_DIR, f'{cache_name}_{seq_len}_frsm37.pt')

        if os.path.exists(cache_path):
            print(f'加载缓存 {cache_path}', flush=True)
            self.data = torch.load(cache_path, weights_only=False)
            if max_lines and max_lines < len(self.data):
                import random as _r
                _r.seed(42)
                self.data = _r.sample(self.data, max_lines)   # 随机采样, 避免类别排序偏置
            print(f'数据集: {len(self.data)} 样本 (缓存)', flush=True)
            return

        print(f'预分词 {path} (seq={seq_len}, 多进程)...', flush=True)
        is_ = tok.token_to_id.get('<|im_start|>')
        ie_ = tok.token_to_id.get('<|im_end|>')
        uid_ = tok.token_to_id.get('博士')
        aid_ = tok.token_to_id.get('<|agent|>')
        ts_ = tok.token_to_id.get('<|think|>')
        te_ = tok.token_to_id.get('<|end_think|>')
        specs = (is_, ie_, uid_, aid_, ts_, te_)
        agent_voc_path = os.path.join(r'F:\OpenASH2605', 'open_ash_voc_agent.json')

        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
        if max_lines:
            import random as _r
            _r.seed(42)
            lines = [l for l in lines if l.strip()]
            if max_lines < len(lines):
                lines = _r.sample(lines, max_lines)
        total = len(lines)

        from concurrent.futures import ProcessPoolExecutor
        args_iter = [(ln, data_type, seq_len) for ln in lines]
        all_data = []
        skipped = 0
        done = 0
        nw = min(8, (os.cpu_count() or 4))
        with ProcessPoolExecutor(max_workers=nw, initializer=_init_worker,
                                 initargs=(agent_voc_path, specs)) as ex:
            for ids in ex.map(_tok_one, args_iter, chunksize=2000):
                done += 1
                if ids is None:
                    skipped += 1
                    continue
                all_data.append(torch.tensor(ids, dtype=torch.long))
                if done % 200000 == 0:
                    print(f'  ... {done}/{total} ({100*done//total}%) '
                          f'{len(all_data)} 样本, {skipped} 跳过', flush=True)

        import random as _r2
        _r2.seed(42)
        _r2.shuffle(all_data)   # 打乱, 消除源文件类别排序
        torch.save(all_data, cache_path)
        self.data = all_data
        print(f'完成: {len(self.data)} 样本, {skipped} 跳过', flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i][:self.seq_len + 1]

    @staticmethod
    def collate(items):
        p = pad_sequence(items, batch_first=True, padding_value=0)
        return p[:, :-1], p[:, 1:]


# ============================================================
# Training loop
# ============================================================
def train(model, ds, dev, tag, epochs, seq_len, lr=LR, max_steps=0,
          batch_size=BATCH, grad_accum=GRAD_ACCUM):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda')

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0,
                        collate_fn=CachedDataset.collate, drop_last=True, pin_memory=True)
    # 真 epoch: 每个 optimizer step 消耗 grad_accum 个 micro-batch
    steps_per_epoch = max(1, len(loader) // grad_accum)
    if max_steps and max_steps < epochs * steps_per_epoch:
        total_steps = max_steps
    else:
        total_steps = epochs * steps_per_epoch

    global_step = 0
    best_loss = float('inf')
    ckp_path = os.path.join(OUT_DIR, f'frsm37_60m_{tag}_latest.pth')

    if os.path.exists(ckp_path):
        with open(ckp_path, 'rb') as f:
            ckp = torch.load(f, map_location=dev)
        model.load_state_dict(ckp['model'])
        opt.load_state_dict(ckp['optimizer'])
        scaler.load_state_dict(ckp['scaler'])
        global_step = ckp.get('step', 0)
        best_loss = ckp.get('best_loss', float('inf'))
        del ckp
        print(f'[续训] {tag} from step {global_step}, best_loss={best_loss:.4f}', flush=True)

    opt.zero_grad(set_to_none=True)
    t0 = time.time()
    running_loss = 0.0
    start_epoch = global_step // steps_per_epoch
    print(f'[{tag}] {epochs} epochs, {steps_per_epoch} steps/epoch, 共 {total_steps} steps', flush=True)

    for epoch in range(start_epoch, epochs):
        it = iter(loader)
        for step_in_epoch in range(steps_per_epoch):
            if global_step >= total_steps:
                break
            for micro in range(grad_accum):
                try:
                    x, t = next(it)
                except StopIteration:
                    it = iter(loader)
                    x, t = next(it)
                x = x[:, :seq_len].to(dev, non_blocking=True)
                t = t[:, :seq_len].to(dev, non_blocking=True)
                x = x.clamp(0, VS - 1)
                t = t.clamp(0, VS - 1)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = model(x)
                    loss = F.cross_entropy(out.reshape(-1, VS), t.reshape(-1),
                                           ignore_index=0) / grad_accum
                scaler.scale(loss).backward()
                running_loss += loss.item() * grad_accum

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            global_step += 1

            # warmup + cosine
            if global_step <= WARMUP:
                cur_lr = lr * global_step / WARMUP
            else:
                prog = (global_step - WARMUP) / max(1, total_steps - WARMUP)
                cur_lr = lr * 0.1 * (1 + 0.9 * (1 + math.cos(math.pi * prog)))
            for pg in opt.param_groups:
                pg['lr'] = cur_lr

            if global_step % LOG_EVERY == 0:
                avg = running_loss / LOG_EVERY / grad_accum
                elapsed = time.time() - t0
                tps = global_step * grad_accum * batch_size * seq_len / elapsed
                print(f'  [{tag}] e{epoch+1}/{epochs} s{global_step:>6d}/{total_steps} '
                      f'loss={avg:.4f} lr={cur_lr:.2e} {tps:.0f}tok/s', flush=True)
                running_loss = 0.0
                if avg < best_loss:
                    best_loss = avg

            if global_step % SAVE_EVERY == 0:
                safe_save({
                    'model': model.state_dict(), 'optimizer': opt.state_dict(),
                    'scaler': scaler.state_dict(), 'step': global_step,
                    'best_loss': best_loss, 'config': {'H': HIDDEN, 'L': LAYERS, 'Hd': HEADS},
                }, ckp_path)
                print(f'  [保存] step {global_step}', flush=True)

        safe_save({
            'model': model.state_dict(), 'optimizer': opt.state_dict(),
            'scaler': scaler.state_dict(), 'step': global_step,
            'best_loss': best_loss, 'config': {'H': HIDDEN, 'L': LAYERS, 'Hd': HEADS},
        }, ckp_path)
        print(f'  [{tag}] 第 {epoch+1}/{epochs} 轮完成', flush=True)

    final_path = os.path.join(OUT_DIR, f'frsm37_60m_{tag}_final.pth')
    safe_save({
        'model': model.state_dict(), 'step': global_step,
        'best_loss': best_loss, 'config': {'H': HIDDEN, 'L': LAYERS, 'Hd': HEADS},
    }, final_path)
    print(f'[{tag}] 完成: {final_path}', flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pretrain_epochs', type=int, default=3)
    ap.add_argument('--sft_epochs', type=int, default=2)
    ap.add_argument('--skip_pretrain', action='store_true')
    ap.add_argument('--skip_sft', action='store_true')
    ap.add_argument('--max_lines_pretrain', type=int, default=0)
    ap.add_argument('--max_lines_sft', type=int, default=0)
    ap.add_argument('--lr', type=float, default=LR)
    ap.add_argument('--baseline', action='store_true',
                    help='用未过滤的原始 minimind 数据训练 (对照基线)')
    ap.add_argument('--refined', action='store_true',
                    help='用 正则+frsm 精炼后的数据训练')
    ap.add_argument('--max_steps', type=int, default=0,
                    help='限制总训练步数 (0=按epoch跑完)')
    args = ap.parse_args()

    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.baseline:
        tag_prefix = 'baseline'
    elif args.refined:
        tag_prefix = 'refined'
    else:
        tag_prefix = 'filtered'
    raw_dir = r'F:\OpenASH2605\minimind_data'
    if args.baseline:
        data_dir = raw_dir
        pt_name = 'pretrain_t2t_mini'
    elif args.refined:
        data_dir = DATA_DIR
        pt_name = 'pretrain_refined'
    else:
        data_dir = DATA_DIR
        pt_name = 'pretrain_filtered'   # 全量三问过滤数据
    sft_name = 'sft_t2t_mini' if args.baseline else 'sft_filtered'
    print(f'设备: {dev} | 词表: {VS} | 数据源: {tag_prefix}', flush=True)

    voc = OpenASHVoc(agent_voc_path=os.path.join(r'F:\OpenASH2605', 'open_ash_voc_agent.json'))
    torch.manual_seed(42)
    model = FRSMASHv37(VS, HIDDEN, HEADS, LAYERS).to(dev)
    print(f'模型: FRSMASHv37 H={HIDDEN} L={LAYERS} heads={HEADS} = {count_params(model):,} 参数', flush=True)

    if not args.skip_pretrain:
        print(f'\n{"="*60}\n  预训练 seq={PRETRAIN_SEQ} epochs={args.pretrain_epochs} [{tag_prefix}]\n{"="*60}')
        pt_ds = CachedDataset(os.path.join(data_dir, pt_name + '.jsonl'), voc,
                              PRETRAIN_SEQ, 'pretrain', f'pt_frsm37_{tag_prefix}',
                              args.max_lines_pretrain or None)
        pre_tag = {'baseline': 'baseline_pretrain', 'refined': 'refined_pretrain'}.get(tag_prefix, 'pretrain')
        model = train(model, pt_ds, dev, pre_tag,
                      args.pretrain_epochs, PRETRAIN_SEQ, args.lr, max_steps=args.max_steps)
        gc.collect()
        torch.cuda.empty_cache()

    if not args.skip_sft:
        print(f'\n{"="*60}\n  SFT seq={SFT_SEQ} epochs={args.sft_epochs} [{tag_prefix}]\n{"="*60}')
        pre_final = os.path.join(OUT_DIR, {
            'baseline': 'frsm37_60m_baseline_pretrain_final.pth',
            'refined': 'frsm37_60m_refined_pretrain_final.pth',
        }.get(tag_prefix, 'frsm37_60m_pretrain_final.pth'))
        if args.skip_pretrain and os.path.exists(pre_final):
            model.load_state_dict(torch.load(pre_final, map_location=dev)['model'])
        sft_ds = CachedDataset(os.path.join(data_dir, sft_name + '.jsonl'), voc,
                               SFT_SEQ, 'sft', f'sft_frsm37_{tag_prefix}',
                               args.max_lines_sft or None)
        sft_tag = {'baseline': 'baseline_sft', 'refined': 'refined_sft'}.get(tag_prefix, 'sft')
        model = train(model, sft_ds, dev, sft_tag,
                      args.sft_epochs, SFT_SEQ, args.lr * 0.5, max_steps=args.max_steps,
                      batch_size=SFT_BATCH, grad_accum=SFT_GRAD_ACCUM)
        gc.collect()
        torch.cuda.empty_cache()

    print(f'\n{"="*60}\n  完成! 权重在 {OUT_DIR}/\n{"="*60}')


if __name__ == '__main__':
    main()
