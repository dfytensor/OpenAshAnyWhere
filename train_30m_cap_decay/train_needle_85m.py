"""
Needle-in-a-Haystack fine-tuning for 85M base model
"""
import torch, sys, os, json, math, time, random, tempfile, gc
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

ROOT = r"F:\OpenASH2605"
BENCH = os.path.join(ROOT, "experiment_openash_vs_wdlm", "bench")
sys.path.insert(0, ROOT); sys.path.insert(0, BENCH); os.chdir(ROOT)
from open_ash_voc import OpenASHVoc
from config import agent_voc_path
from open_ash import OpenASH
from open_ash_infer import _sp

OUT_DIR = "./train_30m_cap_decay"
DEV = "cuda"
CHUNK = 64
STATE_CAP = 150
STATE_DECAY = 0.97
HIDDEN_SIZE = 768
NUM_LAYERS = 12
NUM_HEADS = 8

SFT_SEQ = 768
BATCH_SIZE = 8
GRAD_ACCUM = 4
LR = 1e-5
WEIGHT_DECAY = 0.01
N_EPOCHS = 3
SAVE_EVERY = 200
LOG_EVERY = 20

N_TRAIN = 20000
RANDOM_SEED = 42

NEEDLE_TEMPLATES = [
    ("我的手机密码是{val}，请记住这个密码。", "我的手机密码是什么？", "digit"),
    ("密码箱的密码是{val}，千万不要忘记。", "密码箱的密码是多少？", "digit"),
    ("会议室在三楼{val}房间，下午两点开会。", "会议室在哪个房间？", "digit3"),
    ("今天是{val}，天气晴朗。", "今天的日期是哪天？", "date"),
    ("张三的银行卡号是{val}。", "张三的银行卡号后四位是多少？", "card"),
    ("钥匙藏在门口{val}下面第三个位置。", "钥匙藏在哪里？", "location"),
    ("小明家的猫叫{val}，是一只橘色的胖猫。", "小明家的猫叫什么名字？", "name_cat"),
    ("这本书的作者是{val}，出版社是清华大学出版社。", "这本书的作者是谁？", "name_person"),
    ("我最喜欢的颜色是{val}。", "我最喜欢的颜色是什么？", "color"),
    ("那个城市叫{val}，在南方。", "那个城市叫什么？", "city"),
    ("冰箱里有{val}，记得吃。", "冰箱里有什么？", "food"),
    ("车牌号是{val}，是一辆黑色轿车。", "车牌号是什么？", "plate"),
    ("我的英文名是{val}，朋友们都这么叫我。", "我的英文名是什么？", "english_name"),
    ("电脑密码是{val}，不要告诉别人。", "电脑密码是什么？", "digit4"),
    ("门牌号是{val}，在二楼。", "门牌号是多少？", "door"),
    ("那个学校叫{val}，在市中心。", "那个学校叫什么？", "school"),
    ("我的生日是{val}。", "我的生日是哪天？", "birthday"),
    ("WiFi密码是{val}，连上了吗？", "WiFi密码是什么？", "wifi"),
    ("那个水果叫{val}，很甜。", "那个水果叫什么？", "fruit"),
    ("公司名字是{val}，在科技园。", "公司名字是什么？", "company"),
]

DIGITS_POOL = [str(random.randint(1000, 9999)) for _ in range(50)]
CAT_NAMES = ["橘子", "小白", "花花", "咪咪", "大黄", "豆豆", "团子", "雪球", "小黑", "布丁"]
PERSON_NAMES = ["李华", "王明", "张伟", "刘芳", "陈静", "赵强", "周杰", "吴婷", "孙磊", "黄丽"]
COLORS = ["红色", "蓝色", "绿色", "紫色", "橙色", "粉色", "金色", "银色"]
CITIES = ["深圳", "杭州", "成都", "武汉", "南京", "重庆", "苏州", "长沙", "西安", "青岛"]
FOODS = ["苹果", "牛奶", "鸡蛋", "西瓜", "草莓", "蛋糕", "面包", "橙子", "葡萄", "芒果"]
PLATES = ["京A" + str(random.randint(10000, 99999)) for _ in range(20)]
ENGLISH_NAMES = ["Tom", "Jack", "Alice", "Bob", "Lucy", "David", "Mary", "Peter", "Lily", "Kevin"]
DOORS = [str(random.randint(101, 599)) for _ in range(30)]
SCHOOLS = ["希望小学", "育才中学", "实验学校", "阳光学校", "明德书院", "启航学校"]
BIRTHDAYS = ["{}月{}日".format(random.randint(1, 12), random.randint(1, 28)) for _ in range(30)]
WIFIS = [str(random.randint(10000000, 99999999)) for _ in range(30)]
FRUITS = ["火龙果", "榴莲", "猕猴桃", "百香果", "山竹", "荔枝", "龙眼", "杨梅"]
COMPANIES = ["星辰科技", "蓝海信息", "云端数据", "智联创新", "锐思软件"]
LOCATIONS = ["花盆", "鞋柜", "书架", "地毯", "沙发垫", "冰箱顶", "窗台", "门垫"]


def gen_val(val_type):
    pool = {
        "digit": DIGITS_POOL, "digit3": [str(random.randint(100, 599))],
        "digit4": [str(random.randint(1000, 9999))],
        "date": ["{}年{}月{}日".format(random.randint(2020, 2026), random.randint(1, 12), random.randint(1, 28))],
        "card": ["6225880" + str(random.randint(10000000, 99999999))],
        "location": LOCATIONS, "name_cat": CAT_NAMES, "name_person": PERSON_NAMES,
        "color": COLORS, "city": CITIES, "food": FOODS, "plate": PLATES,
        "english_name": ENGLISH_NAMES, "door": DOORS, "school": SCHOOLS,
        "birthday": BIRTHDAYS, "wifi": WIFIS, "fruit": FRUITS, "company": COMPANIES,
    }
    return random.choice(pool.get(val_type, [str(random.randint(1000, 9999))]))


def load_novel_chunks(path, n_chunks=500, chunk_size=200):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read(2000000)
    return ["".join(list(text)[i:i + chunk_size]) for i in range(0, len(text) - chunk_size, chunk_size)][:n_chunks]


def generate_needle_sample(voc, sp, novel_chunks, seq_len):
    template = random.choice(NEEDLE_TEMPLATES)
    stmt_tmpl, question, val_type = template
    val = gen_val(val_type)
    needle_stmt = stmt_tmpl.format(val=val)
    context_chunk = random.choice(novel_chunks)
    needle_ids = voc.encode(needle_stmt)
    question_ids = voc.encode(question)
    context_ids = voc.encode(context_chunk)
    max_ctx = seq_len - len(needle_ids) - len(question_ids) - 10
    if max_ctx < 20: max_ctx = 20
    context_ids = context_ids[:max_ctx]
    insert_pos = random.randint(0, max(1, len(context_ids) - 1))
    before = context_ids[:insert_pos]
    after = context_ids[insert_pos:]
    answer_ids = voc.encode(val)
    qa_ids = ([sp["im_start"], sp["user"]] + question_ids +
              [sp["im_end"], sp["im_start"], sp["agent"]] + answer_ids + [sp["im_end"]])
    full_ids = before + needle_ids + after + qa_ids
    if len(full_ids) > seq_len + 1: full_ids = full_ids[:seq_len + 1]
    while len(full_ids) < seq_len + 1: full_ids.append(0)
    return torch.tensor(full_ids[:seq_len + 1], dtype=torch.long)


class NeedleDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]
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


def apply_cap_decay(state, n_layers):
    for i in range(n_layers):
        if state[i] is not None:
            s = state[i]; sn = s.norm()
            if sn > STATE_CAP: s = s * (STATE_CAP / sn)
            state[i] = s * STATE_DECAY


def train_needle(model, train_data, dev, vs):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler()
    n_layers = len(model.decoder_layers)
    ds = NeedleDataset(train_data)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                        collate_fn=NeedleDataset.collate, drop_last=True, pin_memory=True)
    steps_per_epoch = len(loader)
    total_steps = N_EPOCHS * steps_per_epoch
    ckp_path = f"{OUT_DIR}/openash85m_base_needle_latest.pth"
    global_step = 0; best_loss = float('inf')

    if os.path.exists(ckp_path):
        ckp = torch.load(ckp_path, map_location=dev)
        model.load_state_dict(ckp['model']); opt.load_state_dict(ckp['optimizer'])
        scaler.load_state_dict(ckp['scaler']); global_step = ckp.get('step', 0)
        best_loss = ckp.get('best_loss', float('inf')); del ckp
        print(f"[Resume] from step {global_step}, best_loss={best_loss:.4f}")

    opt.zero_grad(set_to_none=True)
    t0 = time.time(); running_loss = 0.0; start_epoch = global_step // steps_per_epoch
    print(f"[85M Needle SFT] {N_EPOCHS} epochs, {steps_per_epoch} steps/epoch, {total_steps} total")

    for epoch in range(start_epoch, N_EPOCHS):
        it = iter(loader)
        for step_in_epoch in range(steps_per_epoch):
            if global_step >= total_steps: break
            for micro in range(GRAD_ACCUM):
                try: x, t = next(it)
                except StopIteration: it = iter(loader); x, t = next(it)
                x = x[:, :SFT_SEQ].to(dev, non_blocking=True).clamp(0, vs - 1)
                t = t[:, :SFT_SEQ].to(dev, non_blocking=True).clamp(0, vs - 1)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    states = [None] * n_layers; chunk_logits = []
                    for c0 in range(0, x.size(1), CHUNK):
                        c = x[:, c0:c0 + CHUNK]; h = model.em(c)
                        for i, layer in enumerate(model.decoder_layers):
                            h2, s = layer(h, states[i]); h = h2 + h; states[i] = s
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
                print(f"  [85m-needle] e{epoch+1}/{N_EPOCHS} s{global_step:>6d}/{total_steps} "
                      f"loss={avg:.4f} lr={lr:.2e} {global_step*GRAD_ACCUM*BATCH_SIZE*SFT_SEQ/elapsed:.0f}tok/s", flush=True)
                running_loss = 0.0
                if avg < best_loss: best_loss = avg
            if global_step % SAVE_EVERY == 0:
                safe_save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                           'scaler': scaler.state_dict(), 'step': global_step, 'best_loss': best_loss}, ckp_path)
                print(f"  [Save] step {global_step}", flush=True)

        safe_save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                   'scaler': scaler.state_dict(), 'step': global_step, 'best_loss': best_loss}, ckp_path)
        print(f"  EPOCH {epoch+1}/{N_EPOCHS} complete", flush=True)

    final_path = f"{OUT_DIR}/openash85m_base_needle_final.pth"
    safe_save({'model': model.state_dict(), 'step': global_step, 'best_loss': best_loss}, final_path)
    print(f"[85M Needle SFT] Final: {final_path}")
    return model


@torch.no_grad()
def needle_eval(model, voc, sp, novel_chunks, dev, vs, n_trials=15):
    model.eval()
    n_layers = len(model.decoder_layers)
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
    results = {}

    print("\n" + "=" * 70)
    print("  85M Needle-in-a-Haystack Evaluation")
    print("=" * 70)

    for cl in CTX_LENS:
        results[cl] = {}
        for d in DEPTHS:
            hits = 0
            for trial in range(n_trials):
                needle_stmt, question, answer = random.choice(NEEDLES)
                needle_ids = voc.encode(needle_stmt)
                question_ids = voc.encode(question)
                max_ctx = cl - len(needle_ids) - len(question_ids) - 20
                if max_ctx < 20: max_ctx = 20
                context_ids = voc.encode(random.choice(novel_chunks))[:max_ctx]
                insert_pos = int(len(context_ids) * d / 100)
                insert_pos = max(0, min(insert_pos, len(context_ids) - 1))
                before = context_ids[:insert_pos]; after = context_ids[insert_pos:]
                qa_prefix = [sp["im_start"], sp["user"]] + question_ids + [sp["im_end"], sp["im_start"], sp["agent"]]
                full_ids = before + needle_ids + after + qa_prefix
                if len(full_ids) > cl: full_ids = full_ids[:cl]
                while len(full_ids) < 64: full_ids.append(0)
                dist = len(full_ids) - len(before)

                x = torch.tensor([full_ids], dtype=torch.long, device=dev).clamp(0, vs - 1)
                states = [None] * n_layers; generated = x
                for _ in range(60):
                    ctx = generated[:, -768:]
                    c = ctx[:, -CHUNK:] if ctx.size(1) > CHUNK else ctx
                    h = model.em(c)
                    for i, la in enumerate(model.decoder_layers):
                        h2, s = la(h, states[i]); h = h2 + h; states[i] = s
                    logits = model.head_score(h)[:, -1, :] / 0.7
                    v, _ = torch.topk(logits, 40)
                    logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
                    nt = torch.multinomial(F.softmax(logits, dim=-1), 1)
                    generated = torch.cat([generated, nt], dim=1)
                    if nt.item() == sp["im_end"]: break
                resp_text = voc.decode(generated[0].tolist()[len(full_ids):]).strip()[:200]
                hit = 1 if answer in resp_text else 0
                hits += hit

            acc = hits / n_trials
            results[cl][d] = acc
            lb = "{}K".format(cl // 1024) if cl >= 1024 else str(cl)
            print("  {:>5} @{:>3}%  ACC={:>5.0%}  ({}/{})".format(lb, d, acc, hits, n_trials))
            sys.stdout.flush()
        print()

    print("\n" + "=" * 60)
    print("  85M Depth Scan Summary")
    print("=" * 60)
    header = "{:>5}".format("Depth")
    for cl in CTX_LENS:
        header += "  {:>6}".format("{}K".format(cl // 1024) if cl >= 1024 else str(cl))
    print(header)
    print("-" * (5 + 8 * len(CTX_LENS)))
    for d in DEPTHS:
        row = "{:>4}%".format(d)
        for cl in CTX_LENS:
            row += "  {:>5.0%}".format(results[cl].get(d, 0))
        print(row)
    model.train()


if __name__ == "__main__":
    random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    vs = len(voc.token_to_id) + 1; sp = _sp(voc)

    novel_path = os.path.join(r"F:\小说\女生小说", "傲世九重天-风凌天下.txt")
    novel_chunks = load_novel_chunks(novel_path, 500, 200)
    print(f"{len(novel_chunks)} context chunks loaded")

    print(f"Generating {N_TRAIN} needle training samples...")
    train_data = [generate_needle_sample(voc, sp, novel_chunks, SFT_SEQ) for _ in range(N_TRAIN)]
    print(f"  Done: {len(train_data)} samples")

    dev = torch.device('cuda:0')
    model = OpenASH(voc_size=vs, hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS,
                    num_layers=NUM_LAYERS, model_flag="train").to(dev)
    model.load_state_dict(torch.load(os.path.join(BENCH, "full_sft_768_12.pth"), map_location=dev))
    print(f"85M base loaded: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")

    print(f"\n{'='*60}\n  85M Needle SFT: {N_TRAIN} samples, {N_EPOCHS} epochs\n  CAP={STATE_CAP}, DECAY={STATE_DECAY}, SEQ={SFT_SEQ}\n{'='*60}")
    model = train_needle(model, train_data, dev, vs)

    print("\nRunning needle evaluation...")
    needle_eval(model, voc, sp, novel_chunks, dev, vs, n_trials=15)
    print("\nDone!")
