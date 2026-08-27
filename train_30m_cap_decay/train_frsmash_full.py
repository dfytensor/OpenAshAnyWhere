"""
FRSMASH v1/v2 全量训练脚本
用法: python train_frsmash_full.py --arch v1   # v1 (cummax)
      python train_frsmash_full.py --arch v2   # v2 (F-layer)
      python train_frsmash_full.py --arch v1 --skip_pt --skip_sft --test_only
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

# ── 模型配置 ──────────────────────────────────────
HIDDEN = 384
HEADS = 8
LAYERS = 6
K_SLOW = 8

# ── 训练配置 ──────────────────────────────────────
PT_SEQ = 512; SFT_SEQ = 768; NDL_SEQ = 768
BATCH = 32
LR = 6e-4
WD = 0.01
PT_EP = 3; SFT_EP = 2; NDL_EP = 3
SAVE_EVERY = 500; LOG_EVERY = 20
DEV = "cuda"

OUT_DIR = "./train_30m_cap_decay"
DATA_DIR = "./minimind_data"
CACHE_DIR = "./train_30m_cap_decay/cache"


# ═══════════════════════════════════════════════════
# 数据
# ═══════════════════════════════════════════════════
class CachedDataset(Dataset):
    def __init__(self, path, tok, seq_len, data_type, cache_name):
        self.data = []
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = "{}/{}_{}.pt".format(CACHE_DIR, cache_name, seq_len)
        if os.path.exists(cache_path):
            self.data = torch.load(cache_path, weights_only=False)
            return
        is_ = tok.token_to_id.get('<|im_start|>'); ie_ = tok.token_to_id.get('<|im_end|>')
        uid_ = tok.token_to_id.get('_eval'); aid_ = tok.token_to_id.get('<|agent|>')
        ts_ = tok.token_to_id.get('<|think|>'); te_ = tok.token_to_id.get('<|end_think|>')
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
        buffer = []
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
                            if msg.get('reasoning_content'):
                                ids += [ts_] + tok.encode(msg['reasoning_content']) + [te_]
                            ids += tok.encode(ct) + [ie_]
                if len(ids) >= 4: buffer.append(torch.tensor(ids[:seq_len+1], dtype=torch.long))
            except: pass
        torch.save(buffer, cache_path)
        self.data = buffer

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]
    @staticmethod
    def collate(items):
        p = pad_sequence(items, batch_first=True, padding_value=0)
        return p[:, :-1], p[:, 1:]


class NeedleDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]
    @staticmethod
    def collate(items):
        p = pad_sequence(items, batch_first=True, padding_value=0)
        return p[:, :-1], p[:, 1:]


# ═══════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════
def safe_save(obj, path):
    fd, tmp = tempfile.mkstemp(suffix='.tmp', dir=os.path.dirname(path))
    try:
        os.close(fd); torch.save(obj, tmp)
        if os.path.exists(path): os.remove(path)
        os.rename(tmp, path)
    except:
        if os.path.exists(tmp): os.remove(tmp); raise


# ═══════════════════════════════════════════════════
# 训练
# ═══════════════════════════════════════════════════
def train_phase(model, ds, dev, vs, phases, epochs, seq_len, prefix):
    """通用训练阶段 (支持 resume)"""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler()
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0,
                        collate_fn=ds.collate, drop_last=True, pin_memory=True)
    sp_epoch = len(loader); total_steps = epochs * sp_epoch
    ckp_path = "{}/{}.pth".format(OUT_DIR, prefix)
    global_step = 0; best_loss = float('inf')

    if os.path.exists(ckp_path):
        ckp = torch.load(ckp_path, map_location=dev)
        model.load_state_dict(ckp['model']); opt.load_state_dict(ckp['optimizer'])
        scaler.load_state_dict(ckp['scaler']); global_step = ckp.get('step', 0)
        best_loss = ckp.get('best_loss', float('inf')); del ckp
        print("  [Resume] step {} best_loss={:.4f}".format(global_step, best_loss))

    opt.zero_grad(set_to_none=True)
    t0 = time.time(); running_loss = 0.0; start_ep = global_step // sp_epoch
    print("  [{}] {} epochs, {} steps/epoch, total {} steps, seq={}".format(
        prefix, epochs, sp_epoch, total_steps, seq_len))

    for ep in range(start_ep, epochs):
        it = iter(loader)
        for _ in range(sp_epoch):
            if global_step >= total_steps: break
            try: x, t = next(it)
            except StopIteration: it = iter(loader); x, t = next(it)
            x = x[:, :seq_len].to(dev, non_blocking=True).clamp(0, vs - 1)
            t = t[:, :seq_len].to(dev, non_blocking=True).clamp(0, vs - 1)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vs), t.reshape(-1), ignore_index=0)
            scaler.scale(loss).backward(); running_loss += loss.item()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            global_step += 1

            lr = LR * (0.1 + 0.45 * (1 + math.cos(math.pi * global_step / total_steps)))
            for pg in opt.param_groups: pg['lr'] = lr

            if global_step % LOG_EVERY == 0:
                avg = running_loss / LOG_EVERY
                elapsed = time.time() - t0
                print("    s{:>7d}/{} loss={:.4f} lr={:.2e} ({:.0f}s)".format(
                    global_step, total_steps, avg, lr, elapsed), flush=True)
                running_loss = 0.0
                if avg < best_loss: best_loss = avg
            if global_step % SAVE_EVERY == 0:
                safe_save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                           'scaler': scaler.state_dict(), 'step': global_step,
                           'best_loss': best_loss}, ckp_path)

        safe_save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                   'scaler': scaler.state_dict(), 'step': global_step,
                   'best_loss': best_loss}, ckp_path)
        print("  [{}] EPOCH {}/{} done, best_loss={:.4f}".format(prefix, ep+1, epochs, best_loss))

    final_path = "{}/{}_final.pth".format(OUT_DIR, prefix)
    safe_save({'model': model.state_dict(), 'step': global_step, 'best_loss': best_loss}, final_path)
    return model


# ═══════════════════════════════════════════════════
# Needle 数据生成
# ═══════════════════════════════════════════════════
def gen_needle_data(voc, sp, chunks, n):
    TM = [
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
        ("车牌号是{val}，是一辆黑色轿车。", "车牌号是什么？", "plate"),
        ("电脑密码是{val}，不要告诉别人。", "电脑密码是什么？", "digit4"),
        ("WiFi密码是{val}，连上了吗？", "WiFi密码是什么？", "wifi"),
        ("公司名字是{val}，在科技园。", "公司名字是什么？", "company"),
    ]
    DIGITS = [str(random.randint(1000, 9999)) for _ in range(100)]
    PO = {
        'digit': DIGITS, 'digit3': [str(random.randint(100,599))],
        'digit4': [str(random.randint(1000,9999))],
        'date': ["{}年6月12日".format(random.randint(2020,2026))],
        'card': ["6225880" + str(random.randint(10000000,99999999))],
        'name_person': ["李华","王明","张伟","刘芳","陈静"],
        'name_cat': ["橘子","小白","花花","咪咪","大黄"],
        'city': ["深圳","杭州","成都","武汉","南京"],
        'food': ["苹果","牛奶","鸡蛋","西瓜","草莓"],
        'plate': ["京A"+str(random.randint(10000,99999)) for _ in range(20)],
        'wifi': [str(random.randint(10000000,99999999)) for _ in range(30)],
        'company': ["星辰科技","蓝海信息","云端数据"],
        'location': ["花盆","鞋柜","书架","地毯","沙发垫","窗台"],
    }
    data = []
    for _ in range(n):
        t = random.choice(TM); val = random.choice(PO.get(t[2], DIGITS))
        stmt = t[0].format(val=val); ans = voc.encode(val)
        nids = voc.encode(stmt); qids = voc.encode(t[1])
        ctx = voc.encode(random.choice(chunks))
        max_ctx = NDL_SEQ - len(nids) - len(qids) - 10
        if max_ctx < 20: max_ctx = 20
        ctx = ctx[:max_ctx]
        pos = random.randint(0, max(1, len(ctx) - 1))
        qa = [sp["im_start"], sp["user"]] + qids + [sp["im_end"], sp["im_start"], sp["agent"]] + ans + [sp["im_end"]]
        full = ctx[:pos] + nids + ctx[pos:] + qa
        if len(full) > NDL_SEQ + 1: full = full[:NDL_SEQ + 1]
        while len(full) < NDL_SEQ + 1: full.append(0)
        data.append(torch.tensor(full[:NDL_SEQ + 1], dtype=torch.long))
    return data


# ═══════════════════════════════════════════════════
# 评估
# ═══════════════════════════════════════════════════
@torch.no_grad()
def test_ppl(model, dev, vs, all_ids):
    model.eval()
    print("\n  PPL Stability:")
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


@torch.no_grad()
def test_needle(model, voc, sp, chunks, dev, vs, n_trials=15):
    model.eval()
    ND = [
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
    D = HIDDEN; results = {}

    print("\n  Needle Depth Scan:")
    for cl in CTX_LENS:
        results[cl] = {}
        for d in DEPTHS:
            hits = 0
            for _ in range(n_trials):
                ns, q, ans = random.choice(ND)
                nids = voc.encode(ns); qids = voc.encode(q)
                max_ctx = cl - len(nids) - len(qids) - 20
                if max_ctx < 20: max_ctx = 20
                cids = voc.encode(random.choice(chunks))[:max_ctx]
                pos = int(len(cids) * d / 100); pos = max(0, min(pos, len(cids) - 1))
                qa = [sp["im_start"], sp["user"]] + qids + [sp["im_end"], sp["im_start"], sp["agent"]]
                full = cids[:pos] + nids + cids[pos:] + qa
                if len(full) > cl: full = full[:cl]
                while len(full) < 4: full.append(0)
                x = torch.tensor([full], dtype=torch.long, device=dev).clamp(0, vs - 1)
                ash_st = [None] * LAYERS; h_slow = torch.zeros(1, D, device=dev)
                gen = x
                for _ in range(60):
                    logits, ash_st, h_slow = model.generate_step(
                        gen[:, -1:], ash_st, h_slow)
                    v, _ = torch.topk(logits, 40)
                    logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
                    nt = torch.multinomial(F.softmax(logits / 0.7, dim=-1), 1)
                    gen = torch.cat([gen, nt], dim=1)
                    if nt.item() == sp["im_end"]: break
                resp = voc.decode(gen[0].tolist()[len(full):]).strip()[:200]
                hits += 1 if ans in resp else 0
            acc = hits / n_trials; results[cl][d] = acc
            lb = "{}K".format(cl // 1024) if cl >= 1024 else str(cl)
            print("  {:>5} @{:>3}%  ACC={:>5.0%}".format(lb, d, acc))
        print()

    print("\n  Summary:")
    hdr = "{:>5}".format("Depth")
    for cl in CTX_LENS: hdr += "  {:>6}".format("{}K".format(cl // 1024) if cl >= 1024 else str(cl))
    print("  " + hdr); print("  " + "-" * (5 + 8 * len(CTX_LENS)))
    for d in DEPTHS:
        row = "  {:>4}%".format(d)
        for cl in CTX_LENS: row += "  {:>5.0%}".format(results[cl].get(d, 0))
        print(row)

    model.train()
    return results


# ═══════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True, choices=['v1', 'v2'])
    parser.add_argument('--skip_pt', action='store_true')
    parser.add_argument('--skip_sft', action='store_true')
    parser.add_argument('--skip_ndl', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    vs = len(voc.token_to_id) + 1; sp = _sp(voc)
    print("FRSMASH {}  |  H={} L={} heads={} K={} vocab={}".format(
        args.arch.upper(), HIDDEN, LAYERS, HEADS, K_SLOW, vs))

    # 模型
    if args.arch == 'v1':
        from frsmash_cd import FRSMASH_CD
        model = FRSMASH_CD(vs, HIDDEN, HEADS, LAYERS, K=K_SLOW,
                           state_cap=None, state_decay=None).to(DEV)
    else:
        from frsmash_v2 import FRSMASH
        model = FRSMASH(vs, HIDDEN, HEADS, LAYERS, K=K_SLOW).to(DEV)

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Params: {:,}".format(n))

    # PPL 测试用文本
    novel_path = os.path.join(r"F:\小说\女生小说", "傲世九重天-风凌天下.txt")
    with open(novel_path, encoding="utf-8", errors="ignore") as f:
        novel_text = f.read(2000000)
    all_ids = voc.encode(novel_text)
    novel_chunks = ["".join(list(novel_text)[i:i + 200])
                    for i in range(0, len(novel_text) - 200, 200)][:500]

    if args.test_only:
        ckp_path = "{}/frsmash_{}_sft_final.pth".format(OUT_DIR, args.arch)
        if os.path.exists(ckp_path):
            model.load_state_dict(torch.load(ckp_path, map_location=DEV)['model'])
        test_ppl(model, DEV, vs, all_ids)
        test_needle(model, voc, sp, novel_chunks, DEV, vs)
        return

    prefix_base = "frsmash_{}".format(args.arch)

    # ── 阶段 1: Pretrain ──
    if not args.skip_pt:
        print("\n" + "=" * 60)
        print("  Phase 1: Pretrain ({} epochs, seq={})".format(PT_EP, PT_SEQ))
        print("=" * 60)
        pt_ds = CachedDataset("{}/pretrain_t2t_mini.jsonl".format(DATA_DIR), voc,
                              PT_SEQ, 'pretrain', 'pt_cache_full')
        model = train_phase(model, pt_ds, DEV, vs, 'pretrain', PT_EP, PT_SEQ,
                            "{}_pt".format(prefix_base))
        gc.collect(); torch.cuda.empty_cache()

    # ── 阶段 2: SFT ──
    if not args.skip_sft:
        print("\n" + "=" * 60)
        print("  Phase 2: SFT ({} epochs, seq={})".format(SFT_EP, SFT_SEQ))
        print("=" * 60)
        if args.skip_pt:
            ckp_path = "{}/{}_pt_final.pth".format(OUT_DIR, prefix_base)
            if os.path.exists(ckp_path):
                model.load_state_dict(torch.load(ckp_path, map_location=DEV)['model'])
                print("  Loaded pretrain weights from {}".format(ckp_path))
        sft_ds = CachedDataset("{}/sft_t2t_mini.jsonl".format(DATA_DIR), voc,
                               SFT_SEQ, 'sft', 'sft_cache_full')
        model = train_phase(model, sft_ds, DEV, vs, 'sft', SFT_EP, SFT_SEQ,
                            "{}_sft".format(prefix_base))
        gc.collect(); torch.cuda.empty_cache()

    # ── 阶段 3: Needle SFT ──
    if not args.skip_ndl:
        print("\n" + "=" * 60)
        print("  Phase 3: Needle SFT ({} epochs, seq={})".format(NDL_EP, NDL_SEQ))
        print("=" * 60)
        if args.skip_pt and args.skip_sft:
            ckp_path = "{}/{}_sft_final.pth".format(OUT_DIR, prefix_base)
            if os.path.exists(ckp_path):
                model.load_state_dict(torch.load(ckp_path, map_location=DEV)['model'])
                print("  Loaded SFT weights from {}".format(ckp_path))
        print("  Generating 20000 needle samples...")
        ndl_data = gen_needle_data(voc, sp, novel_chunks, 20000)
        ndl_ds = NeedleDataset(ndl_data)
        model = train_phase(model, ndl_ds, DEV, vs, 'needle', NDL_EP, NDL_SEQ,
                            "{}_ndl".format(prefix_base))
        gc.collect(); torch.cuda.empty_cache()

    # ── 阶段 4: 评估 ──
    print("\n" + "=" * 60)
    print("  Evaluation")
    print("=" * 60)
    test_ppl(model, DEV, vs, all_ids)
    test_needle(model, voc, sp, novel_chunks, DEV, vs)

    print("\nDone!")


if __name__ == "__main__":
    main()
