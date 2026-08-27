"""
FRSMASH v3 vs OpenASH 85M — 公平对比 (修复版)
"""
import sys, os, torch, torch.nn.functional as F, math, time, random, tempfile
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

DEV = torch.device("cuda:0")
torch.manual_seed(42); random.seed(42)

SEQ = 768; BATCH = 8; LR = 2e-5; STEPS = 2000; N_NDL = 15000; N_TRIALS = 50

# ── 词表与路径 ──
sys.path.insert(0, r"F:\OpenASH2605")  # main project first
sys.path.insert(1, r"F:\OpenASH2605\FRSMASH")  # FRSMASH submodule second
os.chdir(r"F:\OpenASH2605")
from open_ash_voc import OpenASHVoc; from config import agent_voc_path
voc = OpenASHVoc(agent_voc_path=agent_voc_path)
vs = len(voc.token_to_id) + 1
uem, ist, uir, agr = [voc.token_to_id[k] for k in ['<|im_end|>','<|im_start|>','<|user|>','<|agent|>']]

novel_path = os.path.join(r"F:\小说\女生小说", "傲世九重天-风凌天下.txt")
with open(novel_path, encoding="utf-8", errors="ignore") as f:
    novel_text = f.read(2000000)
chunks = ["".join(list(novel_text)[i:i + 200]) for i in range(0, len(novel_text) - 200, 200)][:500]

# ── 20+ 训练模板 ──
TM = [
    ("我的手机密码是{val}，请记住。", "我的手机密码是什么？", "d4"),
    ("密码箱的密码是{val}，不要忘。", "密码箱的密码是多少？", "d4"),
    ("电脑密码是{val}。", "电脑密码是什么？", "d4"),
    ("银行卡密码是{val}，记好。", "银行卡密码是多少？", "d4"),
    ("WiFi密码是{val}。", "WiFi密码是什么？", "d8"),
    ("今天是{val}，天气晴。", "今天的日期是哪天？", "date"),
    ("会议定在{val}举行。", "会议定在哪天？", "date"),
    ("我的生日是{val}。", "我的生日是哪天？", "date"),
    ("作者是{val}，出版社清华。", "作者是谁？", "name"),
    ("医生叫{val}，很专业。", "医生叫什么？", "name"),
    ("老师是{val}，教数学。", "数学老师是谁？", "name"),
    ("钥匙藏在{val}下面。", "钥匙藏在哪里？", "loc"),
    ("猫叫{val}，是只橘猫。", "猫叫什么？", "cat"),
    ("那个城市叫{val}。", "那个城市叫什么？", "city"),
    ("冰箱有{val}，记得吃。", "冰箱里有什么？", "food"),
    ("会议室在{val}房间。", "会议室在哪个房间？", "room"),
    ("车牌是{val}，黑色车。", "车牌号是什么？", "plate"),
    ("公司叫{val}，在科技园。", "公司叫什么？", "company"),
    ("水果叫{val}，很甜。", "水果叫什么？", "fruit"),
    ("英文名叫{val}。", "英文名是什么？", "en_name"),
    ("门牌是{val}，在二楼。", "门牌号是多少？", "door"),
    ("学校叫{val}，市中心。", "学校叫什么？", "school"),
]
D4=[str(random.randint(1000,9999)) for _ in range(50)]
D8=[str(random.randint(10000000,99999999)) for _ in range(30)]
P={
    'd4':D4,'d8':D8,
    'date':["{}年6月{}日".format(random.randint(2020,2026),random.randint(1,28)) for _ in range(20)],
    'name':["李华","王明","张伟","刘芳","陈静","赵强","周杰","吴婷"],
    'cat':["橘子","小白","花花","咪咪","大黄","豆豆","团子","雪球"],
    'city':["深圳","杭州","成都","武汉","南京","重庆","苏州","长沙"],
    'food':["苹果","牛奶","鸡蛋","西瓜","草莓","蛋糕","面包","橙子"],
    'room':[str(random.randint(101,599)) for _ in range(20)],
    'plate':["京A"+str(random.randint(10000,99999)) for _ in range(20)],
    'company':["星辰科技","蓝海信息","云端数据","智联创新","锐思软件"],
    'fruit':["火龙果","榴莲","猕猴桃","百香果","山竹","荔枝"],
    'en_name':["Tom","Jack","Alice","Bob","Lucy","David","Mary"],
    'door':[str(random.randint(101,599)) for _ in range(20)],
    'school':["希望小学","育才中学","实验学校","阳光学校","明德书院"],
    'loc':["花盆","鞋柜","书架","地毯","沙发垫","冰箱顶","窗台","门垫"],
}

def mk_sample():
    t=random.choice(TM); val=random.choice(P.get(t[2],D4))
    stmt=t[0].format(val=val); ans=voc.encode(val); nids=voc.encode(stmt); qids=voc.encode(t[1])
    ctx=voc.encode(random.choice(chunks))[:max(20,SEQ-len(nids)-len(qids)-10)]
    pos=random.randint(0,max(1,len(ctx)-1))
    qa=[ist,uir]+qids+[uem,ist,agr]+ans+[uem]
    full=ctx[:pos]+nids+ctx[pos:]+qa
    if len(full)>SEQ+1:full=full[:SEQ+1]
    while len(full)<SEQ+1:full.append(0)
    return torch.tensor(full[:SEQ+1],dtype=torch.long)
data=[mk_sample() for _ in range(N_NDL)]

class DS(Dataset):
    def __init__(self,d):self.d=d
    def __len__(self):return len(self.d)
    def __getitem__(self,i):return self.d[i]
    @staticmethod
    def collate(i):
        p=pad_sequence(i,batch_first=True,padding_value=0)
        return p[:,:-1],p[:,1:]

# ── 评估用 Needle (20道, 与训练不重复) ──
EN = [
    ("我的银行卡号是6225880137123456。","银行卡后四位？","3456"),
    ("手机解锁密码是8473。","手机解锁密码是多少？","8473"),
    ("保险箱密码是9527，切记。","保险箱密码是多少？","9527"),
    ("今天是2026年8月15日，晴。","今天是几号？","2026"),
    ("生日是2025年3月20日。","生日是哪天？","2025"),
    ("作者是赵强，出版社北京。","作者是谁？","赵强"),
    ("邻居叫吴婷，很友善。","邻居叫什么？","吴婷"),
    ("猫叫豆豆，花猫。","猫叫什么名字？","豆豆"),
    ("藏在地毯下面第二个抽屉。","藏在哪里？","地毯"),
    ("在重庆买了房子。","哪个城市？","重庆"),
    ("买了草莓和葡萄。","买了什么水果？","草莓"),
    ("会议室在512房间。","哪个房间？","512"),
    ("车牌是沪B88345。","车牌多少？","沪B"),
    ("公司叫锐思软件。","公司叫什么？","锐思"),
    ("吃了荔枝很甜。","吃了什么？","荔枝"),
    ("英文名叫Lucy。","英文名是什么？","Lucy"),
    ("住302号房。","门牌号是多少？","302"),
    ("在明德书院上学。","哪个学校？","明德"),
    ("花盆下面有钥匙。","花盆下面有什么？","花盆"),
    ("密码是4321无误。","密码是多少？","4321"),
]
CTX_L=[512,768,1024,2048,4096]
DEPT=[100,95,90,85,80,75,70,60,50,40,30,20,10]

# ═══════════════════════════════════════════════════
# FRSMASH v3
# ═══════════════════════════════════════════════════
def build_v3():
    import importlib.util
    spec = importlib.util.spec_from_file_location("fs_config", r"F:\OpenASH2605\FRSMASH\config\__init__.py")
    fs_cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fs_cfg)
    spec2 = importlib.util.spec_from_file_location("fs_model", r"F:\OpenASH2605\FRSMASH\model\frsmash_v3.py")
    fs_model = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(fs_model)
    model = fs_model.FRSMASHv3(fs_cfg.V, fs_cfg.H, fs_cfg.NUM_HEADS, fs_cfg.NUM_LAYERS,
                                fs_cfg.K, fs_cfg.N_SLOTS)
    ck = torch.load(r"F:\OpenASH2605\FRSMASH\checkpoints\sft\frsmash_v3_sft_step_17000.pt",
                    map_location="cpu", weights_only=True)
    model.load_state_dict(ck.get('model_state_dict', ck), strict=False)
    return model.to(DEV), fs_cfg.H

def fwd_v3(model, x_in, t):
    B = x_in.size(0)
    states = [torch.zeros((B, model.n_slots, model.D//model.n_slots), device=DEV, dtype=torch.bfloat16)
              for _ in model.ash_layers]
    h_slow = torch.zeros((B, model.D), device=DEV, dtype=torch.bfloat16)
    logits = model(x_in, states, h_slow)
    loss = F.cross_entropy(logits.reshape(-1, vs), t.reshape(-1), ignore_index=0)
    return logits, loss

@torch.no_grad()
def gen_v3(model, x, full_len):
    D = model.D; NS = model.n_slots; NL = model.ash_layers
    states = [torch.zeros((1, NS, D//NS), device=DEV, dtype=torch.bfloat16) for _ in NL]
    h_slow = torch.zeros((1, D), device=DEV, dtype=torch.bfloat16)
    logits, states, h_slow = model(x, states, h_slow, return_state=True)
    nt = _sample(logits[:, -1, :])
    gen = torch.cat([x, nt], dim=1)
    for _ in range(59):
        logits, states, h_slow = model.generate_step(gen[:, -1:], states, h_slow)
        nt = _sample(logits)
        gen = torch.cat([gen, nt], dim=1)
        if nt.item() == uem: break
    return voc.decode(gen[0].tolist()[full_len:]).strip()[:200]

# ═══════════════════════════════════════════════════
# OpenASH 85M (cd)
# ═══════════════════════════════════════════════════
CHUNK = 64; CAP = 150; DECAY = 0.97

def build_oa85():
    from open_ash import OpenASH
    BENCH = r"F:\OpenASH2605\experiment_openash_vs_wdlm\bench"
    model = OpenASH(vs, hidden_size=768, num_heads=8, num_layers=12, model_flag="train")
    ck = torch.load(os.path.join(BENCH, "full_sft_768_12.pth"), map_location="cpu")
    model.load_state_dict(ck)
    return model.to(DEV), 768

def fwd_oa85(model, x_in, t):
    from open_ash import OpenASH
    NL = len(model.decoder_layers)
    states = [None] * NL; cl = []
    for c0 in range(0, x_in.size(1), CHUNK):
        c = x_in[:, c0:c0+CHUNK]
        h = model.em(c)
        for i, layer in enumerate(model.decoder_layers):
            h2, s = layer(h, states[i]); h = h2 + h; states[i] = s
        for i in range(NL):
            if states[i] is not None:
                sn = states[i].norm()
                if sn > CAP: states[i] = states[i] * (CAP / sn)
                states[i] = states[i] * DECAY
        cl.append(model.head_score(h))
    logits = torch.cat(cl, dim=1)
    loss = F.cross_entropy(logits.reshape(-1, vs), t.reshape(-1), ignore_index=0)
    return logits, loss

@torch.no_grad()
def gen_oa85(model, x, full_len):
    NL = len(model.decoder_layers)
    states = [None] * NL
    # Prefill
    for c0 in range(0, x.size(1), CHUNK):
        c = x[:, c0:c0+CHUNK]
        h = model.em(c)
        for i, layer in enumerate(model.decoder_layers):
            h2, s = layer(h, states[i]); h = h2 + h; states[i] = s
        for i in range(NL):
            if states[i] is not None:
                sn = states[i].norm()
                if sn > CAP: states[i] = states[i] * (CAP / sn)
                states[i] = states[i] * DECAY
    nt = _sample(model.head_score(h[:, -1:])[:, -1, :])
    gen = torch.cat([x, nt], dim=1)
    for _ in range(59):
        h = model.em(gen[:, -1:])
        for i, layer in enumerate(model.decoder_layers):
            h2, s = layer(h, states[i]); h = h2 + h; states[i] = s
        for i in range(NL):
            if states[i] is not None:
                sn = states[i].norm()
                if sn > CAP: states[i] = states[i] * (CAP / sn)
                states[i] = states[i] * DECAY
        nt = _sample(model.head_score(h)[:, -1, :])
        gen = torch.cat([gen, nt], dim=1)
        if nt.item() == uem: break
    return voc.decode(gen[0].tolist()[full_len:]).strip()[:200]

# ═══════════════════════════════════════════════════
# 通用
# ═══════════════════════════════════════════════════
def _sample(logits):
    logits = logits / 0.7
    v, _ = torch.topk(logits, 40)
    logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
    return torch.multinomial(F.softmax(logits, dim=-1), 1)

def safe_save(obj, path):
    fd, tmp = tempfile.mkstemp(suffix='.tmp', dir=os.path.dirname(path))
    os.close(fd); torch.save(obj, tmp)
    if os.path.exists(path): os.remove(path)
    os.rename(tmp, path)

@torch.no_grad()
def run_eval(model, gen_fn):
    results = {}
    for cl in CTX_L:
        for d in DEPT:
            hits = 0
            for _ in range(N_TRIALS):
                ns, q, ans = random.choice(EN)
                nids = voc.encode(ns); qids = voc.encode(q)
                cids = voc.encode(random.choice(chunks))[:max(20, cl - len(nids) - len(qids) - 20)]
                pos = int(len(cids) * d / 100); pos = max(0, min(pos, len(cids) - 1))
                qa = [ist, uir] + qids + [uem, ist, agr]
                full = cids[:pos] + nids + cids[pos:] + qa
                if len(full) > cl: full = full[:cl]
                while len(full) < 4: full.append(0)
                x = torch.tensor([full], dtype=torch.long, device=DEV).clamp(0, vs - 1)
                resp = gen_fn(model, x, len(full))
                hits += 1 if ans in resp else 0
            results[(cl, d)] = hits / N_TRIALS
        lb = "{}K".format(cl // 1024) if cl >= 1024 else str(cl)
        avg = sum(results[(cl, d)] for d in DEPT) / len(DEPT)
        print("  {} {:>5}  avg={:.0%}".format(label, lb, avg))
        sys.stdout.flush()
    return results

# ═══════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════
all_res = {}
print("Training data: {} samples, {} needle templates".format(N_NDL, len(TM)))
print("Eval: {} trials, {} needles, {} context x {} depths".format(N_TRIALS, len(EN), len(CTX_L), len(DEPT)))
print()

for label, build_fn, fwd_fn, gen_fn in [
    ("FRSMASH_v3", build_v3, fwd_v3, gen_v3),
    ("OpenASH_85M_cd", build_oa85, fwd_oa85, gen_oa85),
]:
    print("=" * 70)
    print("  {}".format(label))
    print("=" * 70)

    model, H = build_fn()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("  {} params | H={} L={}".format("{:,}".format(n), H, 
        len(model.ash_layers) if hasattr(model, 'ash_layers') else len(model.decoder_layers)))

    # Needle SFT
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler()
    loader = DataLoader(DS(data), batch_size=BATCH, shuffle=True, num_workers=0,
                        collate_fn=DS.collate, drop_last=True, pin_memory=True)
    it = iter(loader); t0 = time.time(); rl = 0

    for step in range(1, STEPS + 1):
        try: x_in, t = next(it)
        except: it = iter(loader); x_in, t = next(it)
        x_in = x_in[:, :SEQ].to(DEV).clamp(0, vs - 1)
        t = t[:, :SEQ].to(DEV).clamp(0, vs - 1)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, loss = fwd_fn(model, x_in, t)
        scaler.scale(loss).backward(); rl += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if step % 200 == 0:
            print("  s{:>5d} loss={:.4f} ({:.0f}s)".format(step, rl/200, time.time()-t0)); rl = 0

    # Eval
    print("\n  Needle Evaluation...")
    res = run_eval(model, gen_fn)
    all_res[label] = res
    del model; torch.cuda.empty_cache()

# ── 对比 ──
print("\n" + "=" * 90)
print("  FINAL: FRSMASH v3 vs OpenASH 85M (cd)")
print("  {} steps, {} trials/cell, {} templates".format(STEPS, N_TRIALS, len(EN)))
print("=" * 90)
print("  {:>16}  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}".format(
    "Model", "512", "768", "1K", "2K", "4K", "avg"))
print("  " + "-" * 55)
for label in all_res:
    r = all_res[label]; all_a = []
    row = "  {:>16}".format(label)
    for cl in CTX_L:
        avg_c = sum(r[(cl,d)] for d in DEPT) / len(DEPT)
        row += "  {:>5.0%}".format(avg_c); all_a.extend([r[(cl,d)] for d in DEPT])
    row += "  {:>5.0%}".format(sum(all_a)/len(all_a))
    print(row)

print("\n  Depth distribution (averaged over all context lengths):")
print("  {:>6}".format("Depth"), end="")
for label in all_res: print("  {:>12}".format(label), end="")
print("\n  " + "-" * 35)
for d in DEPT:
    print("  {:>5}%".format(d), end="")
    for label in all_res:
        a = [all_res[label].get((cl,d),0) for cl in CTX_L]
        print("  {:>11.0%}".format(sum(a)/len(a)), end="")
    print()

print("\n  Best cell:")
for label in all_res:
    r = all_res[label]; best_k, best_v = max(r.items(), key=lambda x:x[1])
    print("  {}: {:.0%} @ {}K depth={}%".format(label, best_v, best_k[0]//1024 if best_k[0]>=1024 else best_k[0], best_k[1]))

print("\nDone.")
