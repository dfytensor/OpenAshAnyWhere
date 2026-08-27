"""
FRSMASH v1 (cummax) vs v2 (F-layer) — 公平对比训练脚本
H=384 L=6 heads=8 K=8, 从头预训练 → SFT → Needle SFT → 测试
"""
import torch, sys, os, json, math, time, random, tempfile, gc, argparse
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

ROOT = r"F:\OpenASH2605"
sys.path.insert(0, ROOT); os.chdir(ROOT)
from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from open_ash_infer import _sp

HIDDEN = 384; HEADS = 8; LAYERS = 6; K_SLOW = 8
PT_SEQ = 512; SFT_SEQ = 768
BATCH = 32; GA = 1  # effective 32
LR = 6e-4; WD = 0.01
PT_EP = 1; PT_SAMPLES = 50000
SFT_EP = 1; SFT_SAMPLES = 20000
NDL_EP = 2; NDL_SAMPLES = 5000
LOG_EVERY = 10; SAVE_EVERY = 200
OUT_DIR = "./train_30m_cap_decay"
DATA_DIR = "./minimind_data"
CACHE_DIR = "./train_30m_cap_decay/cache"
DEV = "cuda"


class CachedDataset(Dataset):
    def __init__(self, path, voc, seq_len, data_type='pretrain', cache_name=None, max_lines=None):
        self.data = []
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = '{}/{}_{}.pt'.format(CACHE_DIR, cache_name or os.path.basename(path), seq_len)
        if os.path.exists(cache_path):
            self.data = torch.load(cache_path, weights_only=False)
            if max_lines: self.data = self.data[:max_lines]
            return
        is_ = tok.token_to_id.get('<|im_start|>'); ie_ = tok.token_to_id.get('<|im_end|>')
        uid_ = tok.token_to_id.get('_eval'); aid_ = tok.token_to_id.get('<|agent|>')
        ts_ = tok.token_to_id.get('<|think|>'); te_ = tok.token_to_id.get('<|end_think|>')
        with open(path, encoding='utf-8') as f: lines = f.readlines()
        if max_lines: lines = lines[:max_lines]
        skipped = 0; buffer = []
        for line in lines:
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
                if len(ids) >= 4: buffer.append(torch.tensor(ids[:seq_len+1], dtype=torch.long))
            except: skipped += 1
        torch.save(buffer, cache_path)
        self.data = buffer[:max_lines] if max_lines else buffer

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i][:self.data[i].size(0)]
    @staticmethod
    def collate(items):
        p = pad_sequence(items, batch_first=True, padding_value=0)
        return p[:, :-1], p[:, 1:]


def safe_save(obj, path):
    fd, tmp = tempfile.mkstemp(suffix='.tmp', dir=os.path.dirname(path))
    try:
        os.close(fd); torch.save(obj, tmp)
        if os.path.exists(path): os.remove(path)
        os.rename(tmp, path)
    except:
        if os.path.exists(tmp): os.remove(tmp)
        raise


def train_phase(model, ds, dev, vs, tag, epochs, seq_len, prefix):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler()
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0,
                        collate_fn=CachedDataset.collate, drop_last=True, pin_memory=True)
    steps_per_ep = len(loader); total_steps = epochs * steps_per_ep
    ckp_path = '{}/{}_latest.pth'.format(OUT_DIR, prefix)
    global_step = 0; best_loss = float('inf')

    if os.path.exists(ckp_path):
        ckp = torch.load(ckp_path, map_location=dev)
        model.load_state_dict(ckp['model']); opt.load_state_dict(ckp['optimizer'])
        scaler.load_state_dict(ckp['scaler']); global_step = ckp.get('step', 0)
        best_loss = ckp.get('best_loss', float('inf')); del ckp
        print('[{}] Resume step {} best_loss={:.4f}'.format(prefix, global_step, best_loss))

    opt.zero_grad(set_to_none=True)
    t0 = time.time(); running_loss = 0.0
    print('[{}] {} epochs, {} steps, seq={}'.format(prefix, epochs, total_steps, seq_len))

    for epoch in range(global_step // steps_per_ep, epochs):
        it = iter(loader)
        for _ in range(steps_per_ep):
            if global_step >= total_steps: break
            for micro in range(GA):
                try: x, t = next(it)
                except StopIteration: it = iter(loader); x, t = next(it)
                x = x[:, :seq_len].to(dev, non_blocking=True).clamp(0, vs - 1)
                t = t[:, :seq_len].to(dev, non_blocking=True).clamp(0, vs - 1)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, vs), t.reshape(-1), ignore_index=0) / GA
                scaler.scale(loss).backward()
                running_loss += loss.item() * GA

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            global_step += 1
            progress = global_step / total_steps
            lr = LR * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))
            for pg in opt.param_groups: pg['lr'] = lr

            if global_step % LOG_EVERY == 0:
                avg = running_loss / LOG_EVERY / GA
                elapsed = time.time() - t0
                print('  [{}] s{}/{} loss={:.4f} lr={:.2e}'.format(prefix, global_step, total_steps, avg, lr), flush=True)
                running_loss = 0.0
                if avg < best_loss: best_loss = avg
            if global_step % SAVE_EVERY == 0:
                safe_save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                           'scaler': scaler.state_dict(), 'step': global_step, 'best_loss': best_loss}, ckp_path)

        safe_save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                   'scaler': scaler.state_dict(), 'step': global_step, 'best_loss': best_loss}, ckp_path)

    final_path = '{}/{}_final.pth'.format(OUT_DIR, prefix)
    safe_save({'model': model.state_dict(), 'step': global_step, 'best_loss': best_loss}, final_path)
    return model


def needle_data(voc, sp, n, novel_chunks):
    NEEDLE_TEMPLATES = [
        ("我的手机密码是{val}，请记住这个密码。", "我的手机密码是什么？", "digit"),
        ("密码箱的密码是{val}，千万不要忘记。", "密码箱的密码是多少？", "digit"),
        ("今天是{val}，天气晴朗。", "今天的日期是哪天？", "date"),
        ("这本书的作者是{val}，出版社是清华大学出版社。", "这本书的作者是谁？", "name_person"),
        ("钥匙藏在门口{val}下面第三个位置。", "钥匙藏在哪里？", "location"),
        ("小明家的猫叫{val}，是一只橘色的胖猫。", "小明家的猫叫什么名字？", "name_cat"),
        ("张三的银行卡号是{val}。", "张三的银行卡号后四位是多少？", "card"),
        ("会议室在三楼{val}房间，下午两点开会。", "会议室在哪个房间？", "digit3"),
        ("那个城市叫{val}，在南方。", "那个城市叫什么？", "city"),
        ("冰箱里有{val}，记得吃。", "冰箱里有什么？", "food"),
    ]
    DIGITS = [str(random.randint(1000, 9999)) for _ in range(50)]
    CAT_NAMES = ["橘子", "小白", "花花", "咪咪", "大黄"]
    PERSON_NAMES = ["李华", "王明", "张伟", "刘芳", "陈静"]
    CITIES = ["深圳", "杭州", "成都", "武汉", "南京"]
    FOODS = ["苹果", "牛奶", "鸡蛋", "西瓜", "草莓"]
    LOCATIONS = ["花盆", "鞋柜", "书架", "地毯", "沙发垫"]
    pools = {"digit": DIGITS, "digit3": [str(random.randint(100, 500))], "date": ["{}年{}月{}日".format(random.randint(2020,2026), random.randint(1,12), random.randint(1,28))],
             "card": ["6225880"+str(random.randint(10000000,99999999))], "name_person": PERSON_NAMES, "name_cat": CAT_NAMES,
             "city": CITIES, "food": FOODS, "location": LOCATIONS}
    data = []
    for _ in range(n):
        t = random.choice(NEEDLE_TEMPLATES)
        val = random.choice(pools.get(t[2], DIGITS))
        stmt = t[0].format(val=val); q = t[1]
        needle_ids = voc.encode(stmt); question_ids = voc.encode(q)
        ctx = voc.encode(random.choice(novel_chunks))
        max_ctx = SFT_SEQ - len(needle_ids) - len(question_ids) - 10
        if max_ctx < 20: max_ctx = 20
        ctx = ctx[:max_ctx]
        pos = random.randint(0, max(1, len(ctx) - 1))
        ans = voc.encode(val)
        qa = [sp["im_start"], sp["user"]] + question_ids + [sp["im_end"], sp["im_start"], sp["agent"]] + ans + [sp["im_end"]]
        full = ctx[:pos] + needle_ids + ctx[pos:] + qa
        if len(full) > SFT_SEQ + 1: full = full[:SFT_SEQ + 1]
        while len(full) < SFT_SEQ + 1: full.append(0)
        data.append(torch.tensor(full[:SFT_SEQ + 1], dtype=torch.long))
    return data


class ND(Dataset):
    def __init__(self, d): self.d = d
    def __len__(self): return len(self.d)
    def __getitem__(self, i): return self.d[i]
    @staticmethod
    def collate(items):
        p = pad_sequence(items, batch_first=True, padding_value=0)
        return p[:, :-1], p[:, 1:]


@torch.no_grad()
def eval_needle(model, voc, sp, novel_chunks, dev, vs):
    model.eval()
    NEEDLES = [
        ("我的手机密码是8473，请记住这个密码。", "我的手机密码是什么？", "8473"),
        ("今天是2026年6月12日，天气晴朗。", "今天的日期是哪天？", "2026"),
        ("小明家的猫叫橘子，是一只橘色的胖猫。", "小明家的猫叫什么名字？", "橘子"),
        ("这本书的作者是李华，出版社是清华大学出版社。", "这本书的作者是谁？", "李华"),
        ("密码箱的密码是9527，千万不要忘记。", "密码箱的密码是多少？", "9527"),
        ("会议室在三楼302房间，下午两点开会。", "会议室在哪个房间？", "302"),
        ("张三的银行卡号是6225880137123456。", "张三的银行卡号后四位是多少？", "3456"),
        ("钥匙藏在门口花盆下面第三个位置。", "钥匙藏在哪里？", "花盆"),
    ]
    CTX_LENS = [512, 768, 1024, 2048]
    DEPTHS = [100, 95, 90, 85, 80, 70, 50, 30, 10]
    N = 15; D = HIDDEN; results = {}

    for cl in CTX_LENS:
        results[cl] = {}
        for d in DEPTHS:
            hits = 0
            for _ in range(N):
                ns, q, ans = random.choice(NEEDLES)
                nids = voc.encode(ns); qids = voc.encode(q)
                max_ctx = cl - len(nids) - len(qids) - 20
                if max_ctx < 20: max_ctx = 20
                cids = voc.encode(random.choice(novel_chunks))[:max_ctx]
                pos = int(len(cids) * d / 100)
                pos = max(0, min(pos, len(cids) - 1))
                qa = [sp["im_start"], sp["user"]] + qids + [sp["im_end"], sp["im_start"], sp["agent"]]
                full = cids[:pos] + nids + cids[pos:] + qa
                if len(full) > cl: full = full[:cl]
                while len(full) < 4: full.append(0)
                x = torch.tensor([full], dtype=torch.long, device=dev).clamp(0, vs - 1)
                ash_st = [None] * LAYERS; h_slow = torch.zeros(1, D, device=dev)
                gen = x
                for _ in range(60):
                    logits, ash_st, h_slow = model.generate_step(gen[:, -1:], ash_st, h_slow)
                    v, _ = torch.topk(logits, 40)
                    logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
                    logits = logits / 0.7
                    nt = torch.multinomial(F.softmax(logits, dim=-1), 1)
                    gen = torch.cat([gen, nt], dim=1)
                    if nt.item() == sp["im_end"]: break
                resp = voc.decode(gen[0].tolist()[len(full):]).strip()[:200]
                hits += 1 if ans in resp else 0
            acc = hits / N; results[cl][d] = acc
            lb = "{}K".format(cl // 1024) if cl >= 1024 else str(cl)
            print("  {:>5} @{:>3}%  ACC={:>5.0%}".format(lb, d, acc))
        print()

    print(); print("=" * 60)
    hdr = "{:>5}".format("Depth")
    for cl in CTX_LENS: hdr += "  {:>6}".format("{}K".format(cl // 1024) if cl >= 1024 else str(cl))
    print(hdr); print("-" * (5 + 8 * len(CTX_LENS)))
    for d in DEPTHS:
        row = "{:>4}%".format(d)
        for cl in CTX_LENS: row += "  {:>5.0%}".format(results[cl].get(d, 0))
        print(row)
    model.train()
    return results


@torch.no_grad()
def test_ppl(model, dev, vs, all_ids):
    model.eval()
    print(); print("  PPL stability test:")
    for sl in [512, 1024, 4096, 16384, 65536]:
        if sl > len(all_ids): break
        ids = all_ids[:sl]
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=dev).clamp(0, vs - 1)
        t = torch.tensor([ids[1:]], dtype=torch.long, device=dev).clamp(0, vs - 1)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vs), t.reshape(-1), ignore_index=0)
        ppl = math.exp(loss.item())
        mx = logits.max().item()
        lb = "{}K".format(sl // 1024) if sl >= 1024 else str(sl)
        print("  {:>8} PPL={:>10.1f} logit_max={:.3f} {}".format(
            lb, ppl, mx, "STABLE" if mx < 50 else "BURST"))
    model.train()


def main(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    vs = len(voc.token_to_id) + 1; sp = _sp(voc)
    print("Vocab: {} | H={} L={} heads={}".format(vs, HIDDEN, LAYERS, HEADS))

    novel_path = os.path.join(r"F:\小说\女生小说", "傲世九重天-风凌天下.txt")
    with open(novel_path, encoding="utf-8", errors="ignore") as f:
        novel_text = f.read(2000000)
    novel_chunks = ["".join(list(novel_text)[i:i + 200]) for i in range(0, len(novel_text) - 200, 200)][:500]

    # Shared all_ids for PPL test
    all_ids = voc.encode(novel_text)

    results = {}

    for arch in ["v1", "v2"]:
        print("\n" + "=" * 70)
        print("  Training FRSMASH {} (H={} L={} K={})".format(arch, HIDDEN, LAYERS, K_SLOW))
        print("=" * 70)

        if arch == "v1":
            from frsmash_cd import FRSMASH_CD
            model = FRSMASH_CD(vs, HIDDEN, HEADS, LAYERS, K=K_SLOW, state_cap=None, state_decay=None).to(DEV)
        else:
            from frsmash_v2 import FRSMASH
            model = FRSMASH(vs, HIDDEN, HEADS, LAYERS, K=K_SLOW).to(DEV)

        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("  Params: {:,}".format(n))

        # Phase 1: Pretrain
        if not args.skip_pt:
            print("\n  --- Pretrain ({} samples, {} epoch) ---".format(PT_SAMPLES, PT_EP))
            pt_ds = CachedDataset('{}/pretrain_t2t_mini.jsonl'.format(DATA_DIR), voc, PT_SEQ,
                                  'pretrain', 'pt_cache_{}'.format(PT_SAMPLES), PT_SAMPLES)
            model = train_phase(model, pt_ds, DEV, vs, 'pretrain', PT_EP, PT_SEQ, 'frsmash_{}_pt'.format(arch))
            gc.collect(); torch.cuda.empty_cache()

        # Phase 2: SFT
        if not args.skip_sft:
            print("\n  --- SFT ({} samples, {} epoch) ---".format(SFT_SAMPLES, SFT_EP))
            sft_ds = CachedDataset('{}/sft_t2t_mini.jsonl'.format(DATA_DIR), voc, SFT_SEQ,
                                   'sft', 'sft_cache_{}'.format(SFT_SAMPLES), SFT_SAMPLES)
            model = train_phase(model, sft_ds, DEV, vs, 'sft', SFT_EP, SFT_SEQ, 'frsmash_{}_sft'.format(arch))
            gc.collect(); torch.cuda.empty_cache()

        # Phase 3: Needle SFT
        if not args.skip_ndl:
            print("\n  --- Needle SFT ({} samples, {} epochs) ---".format(NDL_SAMPLES, NDL_EP))
            ndl_data = needle_data(voc, sp, NDL_SAMPLES, novel_chunks)
            ndl_ds = ND(ndl_data)
            model = train_phase(model, ndl_ds, DEV, vs, 'needle', NDL_EP, SFT_SEQ, 'frsmash_{}_ndl'.format(arch))
            gc.collect(); torch.cuda.empty_cache()

        # Phase 4: Evaluation
        print("\n  --- Evaluation ---")
        test_ppl(model, DEV, vs, all_ids)
        print("\n  --- Needle Depth Scan ---")
        r = eval_needle(model, voc, sp, novel_chunks, DEV, vs)
        results[arch] = r

    # Final comparison
    print("\n" + "=" * 70)
    print("  FRSMASH v1 (cummax) vs v2 (F-layer) FINAL COMPARISON")
    print("=" * 70)
    for arch in ["v1", "v2"]:
        r = results.get(arch, {})
        accs = [r.get(cl, {}).get(d, 0) for cl in [512, 768, 1024, 2048] for d in [100, 95, 90, 85, 80]]
        avg_acc = sum(accs) / max(len(accs), 1)
        mx = max(accs) if accs else 0
        print("  {}: max_acc={:.0%} avg_top5={:.0%}".format(arch.upper(), mx, avg_acc))
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_pt', action='store_true')
    parser.add_argument('--skip_sft', action='store_true')
    parser.add_argument('--skip_ndl', action='store_true')
    parser.add_argument('--arch', type=str, default='both', choices=['v1', 'v2', 'both'])
    args = parser.parse_args()
    main(args)
