"""Hybrid ASH+Mem — resume training + eval"""
import sys, os, math, time, random, tempfile
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, r"F:\OpenASH2605"); os.chdir(r"F:\OpenASH2605")
DEV = torch.device("cuda:0"); torch.manual_seed(42); random.seed(42)

from open_ash_voc import OpenASHVoc; from config import agent_voc_path
voc = OpenASHVoc(agent_voc_path=agent_voc_path); vs = len(voc.token_to_id) + 1
uem, ist, uir, agr = [voc.token_to_id[k]
    for k in ['<|im_end|>','<|im_start|>','<|user|>','<|agent|>']]

from open_ash import OpenASH
from frsmash_cd import ASHDecoderLayer, SlowMemoryCellCD

H = 768; LAYERS = 12; HEADS = 8; K_SLOW = 8
CHUNK = 64; CAP = 150; DECAY = 0.97
SEQ = 768; BATCH = 8; LR = 2e-5; TOTAL_STEPS = 5000
N_NDL = 15000; N_TRIALS = 30
OUT_DIR = "./train_30m_cap_decay"
CKP = "{}/hybrid_ash_mem.pth".format(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# ── 数据 ──
novel_path = os.path.join(r"F:\小说\女生小说", "傲世九重天-风凌天下.txt")
with open(novel_path, encoding="utf-8", errors="ignore") as f:
    novel_text = f.read(2000000)
chunks = ["".join(list(novel_text)[i:i + 200])
          for i in range(0, len(novel_text) - 200, 200)][:500]

TM = [
    ("我的手机密码是{val}，请记住。","我的手机密码是什么？","d4"),
    ("密码箱的密码是{val}，不要忘。","密码箱的密码是多少？","d4"),
    ("电脑密码是{val}。","电脑密码是什么？","d4"),
    ("银行卡密码是{val}，记好。","银行卡密码是多少？","d4"),
    ("WiFi密码是{val}。","WiFi密码是什么？","d8"),
    ("今天是{val}，天气晴。","今天的日期是哪天？","date"),
    ("会议定在{val}举行。","会议定在哪天？","date"),
    ("我的生日是{val}。","我的生日是哪天？","date"),
    ("作者是{val}，出版社清华。","作者是谁？","name"),
    ("医生叫{val}，很专业。","医生叫什么？","name"),
    ("老师是{val}，教数学。","数学老师是谁？","name"),
    ("钥匙藏在{val}下面。","钥匙藏在哪里？","loc"),
    ("猫叫{val}，是只橘猫。","猫叫什么？","cat"),
    ("那个城市叫{val}。","那个城市叫什么？","city"),
    ("冰箱有{val}，记得吃。","冰箱里有什么？","food"),
    ("会议室在{val}房间。","会议室在哪个房间？","room"),
    ("车牌是{val}，黑色车。","车牌号是什么？","plate"),
    ("公司叫{val}，在科技园。","公司叫什么？","company"),
    ("水果叫{val}，很甜。","水果叫什么？","fruit"),
    ("英文名叫{val}。","英文名是什么？","en_name"),
    ("门牌是{val}，在二楼。","门牌号是多少？","door"),
    ("学校叫{val}，市中心。","学校叫什么？","school"),
]
D4 = [str(random.randint(1000, 9999)) for _ in range(50)]
D8 = [str(random.randint(10000000, 99999999)) for _ in range(30)]
P = {
    'd4': D4, 'd8': D8,
    'date': ["{}年6月{}日".format(random.randint(2020,2026),random.randint(1,28)) for _ in range(20)],
    'name': ["李华","王明","张伟","刘芳","陈静","赵强","周杰","吴婷"],
    'cat': ["橘子","小白","花花","咪咪","大黄","豆豆","团子","雪球"],
    'city': ["深圳","杭州","成都","武汉","南京","重庆","苏州","长沙"],
    'food': ["苹果","牛奶","鸡蛋","西瓜","草莓","蛋糕","面包","橙子"],
    'room': [str(random.randint(101,599)) for _ in range(20)],
    'plate': ["京A"+str(random.randint(10000,99999)) for _ in range(20)],
    'company': ["星辰科技","蓝海信息","云端数据","智联创新","锐思软件"],
    'fruit': ["火龙果","榴莲","猕猴桃","百香果","山竹","荔枝"],
    'en_name': ["Tom","Jack","Alice","Bob","Lucy","David","Mary"],
    'door': [str(random.randint(101,599)) for _ in range(20)],
    'school': ["希望小学","育才中学","实验学校","阳光学校","明德书院"],
    'loc': ["花盆","鞋柜","书架","地毯","沙发垫","冰箱顶","窗台","门垫"],
}

def mk_sample():
    t = random.choice(TM)
    val = random.choice(P.get(t[2], D4))
    stmt = t[0].format(val=val)
    ans = voc.encode(val); nids = voc.encode(stmt); qids = voc.encode(t[1])
    ctx = voc.encode(random.choice(chunks))
    max_ctx = SEQ - len(nids) - len(qids) - 10
    if max_ctx < 20: max_ctx = 20
    ctx = ctx[:max_ctx]
    pos = random.randint(0, max(1, len(ctx) - 1))
    qa = [ist, uir] + qids + [uem, ist, agr] + ans + [uem]
    full = ctx[:pos] + nids + ctx[pos:] + qa
    if len(full) > SEQ + 1: full = full[:SEQ + 1]
    while len(full) < SEQ + 1: full.append(0)
    return torch.tensor(full[:SEQ + 1], dtype=torch.long)

data = [mk_sample() for _ in range(N_NDL)]

class DS(Dataset):
    def __init__(self, d): self.d = d
    def __len__(self): return len(self.d)
    def __getitem__(self, i): return self.d[i]
    @staticmethod
    def collate(i):
        p = pad_sequence(i, batch_first=True, padding_value=0)
        return p[:, :-1], p[:, 1:]

def safe_save(obj, path):
    fd, tmp = tempfile.mkstemp(suffix='.tmp', dir=os.path.dirname(path))
    os.close(fd); torch.save(obj, tmp)
    if os.path.exists(path): os.remove(path)
    os.rename(tmp, path)

# ── 模型 ──
class FRSMASH_ASH_CD(nn.Module):
    def __init__(self, voc_size, hidden_size, num_heads, num_layers, K=8):
        super().__init__()
        self.D = hidden_size; self.K = K; self.NL = num_layers
        self.em = nn.Embedding(voc_size, hidden_size, padding_idx=0)
        self.ash_layers = nn.ModuleList([
            ASHDecoderLayer(hidden_size, num_heads, "train")
            for _ in range(num_layers)
        ])
        self.ash_norm = nn.LayerNorm(hidden_size)
        self.mem_inp = nn.Linear(hidden_size, hidden_size)
        self.slow = SlowMemoryCellCD(hidden_size)
        self.mem_proj = nn.Linear(hidden_size, hidden_size)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 4), nn.GELU(),
            nn.Linear(hidden_size // 4, 1), nn.Sigmoid()
        )
        self.fusion_norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, voc_size, bias=False)

    def forward(self, x, return_state=False):
        B, T = x.shape; D = self.D
        x_emb = self.em(x)
        ash_states = [None] * self.NL
        ash_outputs = []
        for c0 in range(0, T, CHUNK):
            c = x[:, c0:c0 + CHUNK]
            h = self.em(c)
            for i, layer in enumerate(self.ash_layers):
                h2, s = layer(h, ash_states[i])
                h = h2 + h; ash_states[i] = s
            for i in range(self.NL):
                if ash_states[i] is not None:
                    sn = ash_states[i].norm()
                    if sn > CAP: ash_states[i] = ash_states[i] * (CAP / sn)
                    ash_states[i] = ash_states[i] * DECAY
            ash_outputs.append(h)
        x_ash = self.ash_norm(torch.cat(ash_outputs, dim=1))
        h_slow = torch.zeros(B, D, device=x.device)
        inp_seq = self.mem_inp(x_emb)
        H_slow = torch.zeros(B, T, D, device=x.device); prev = 0
        for t in range(0, T, self.K):
            h_slow = self.slow(inp_seq[:, t], h_slow, cap=None, decay=None)
            H_slow[:, prev:t + 1] = h_slow.unsqueeze(1); prev = t + 1
        if prev < T: H_slow[:, prev:] = h_slow.unsqueeze(1)
        x_mem = self.mem_proj(H_slow)
        cat = torch.cat([x_ash, x_mem], dim=-1)
        gate = self.fusion_gate(cat)
        fused = self.fusion_norm(gate * x_ash + (1 - gate) * x_mem + x_emb)
        logits = self.head(fused)
        if return_state:
            return logits, ash_states, h_slow
        return logits

# ── 构建/恢复 ──
model = FRSMASH_ASH_CD(vs, H, HEADS, LAYERS, K=K_SLOW).to(DEV)
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
step0 = 0

if os.path.exists(CKP):
    print("Resuming from checkpoint...")
    ck = torch.load(CKP, map_location="cpu", weights_only=True)
    model.load_state_dict(ck['model']); step0 = ck['step']
else:
    print("Transferring 85M OpenASH weights...")
    BENCH = r"F:\OpenASH2605\experiment_openash_vs_wdlm\bench"
    oa_ck = torch.load(os.path.join(BENCH, "full_sft_768_12.pth"), map_location="cpu")
    oa_state = oa_ck
    name_map = {}
    for oa_k in oa_state:
        if oa_k == 'head_score.weight': name_map[oa_k] = 'head.weight'
        elif oa_k.startswith('decoder_layers.'):
            parts = oa_k.split('.')
            rest = '.'.join(parts[2:])
            rest = rest.replace('self_attention_linear', 'attn')
            rest = rest.replace('layer_norm', 'norm')
            name_map[oa_k] = 'ash_layers.{}.{}'.format(parts[1], rest)
        else: name_map[oa_k] = oa_k
    mst = model.state_dict()
    t_cnt = 0
    for oa_k, my_k in name_map.items():
        if my_k in mst and oa_state[oa_k].shape == mst[my_k].shape:
            mst[my_k] = oa_state[oa_k]; t_cnt += 1
    model.load_state_dict(mst)
    print("  Transferred: {} / New: {}".format(t_cnt, len(mst) - t_cnt))

print("  Total params: {:,}  |  From step: {}".format(n, step0))

# ── 训练 ──
model.train()
opt = torch.optim.AdamW([
    {'params': [p for n, p in model.named_parameters()
                if any(x in n for x in ['slow','mem','fusion','gate','ash_norm'])], 'lr': LR},
    {'params': [p for n, p in model.named_parameters()
                if not any(x in n for x in ['slow','mem','fusion','gate','ash_norm'])], 'lr': LR * 0.1},
], weight_decay=0.01, betas=(0.9, 0.95))
scaler = torch.amp.GradScaler()
loader = DataLoader(DS(data), batch_size=BATCH, shuffle=True, num_workers=0,
                    collate_fn=DS.collate, drop_last=True, pin_memory=True)
it = iter(loader); t0 = time.time(); rl = 0

if os.path.exists(CKP):
    ck = torch.load(CKP, map_location="cpu", weights_only=True)
    opt.load_state_dict(ck['optimizer'])
    scaler.load_state_dict(ck['scaler'])

for step in range(step0 + 1, TOTAL_STEPS + 1):
    try: x_in, t = next(it)
    except: it = iter(loader); x_in, t = next(it)
    x_in = x_in[:, :SEQ].to(DEV).clamp(0, vs - 1)
    t = t[:, :SEQ].to(DEV).clamp(0, vs - 1)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(x_in)
        loss = F.cross_entropy(logits.reshape(-1, vs), t.reshape(-1), ignore_index=0)
    scaler.scale(loss).backward(); rl += loss.item()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    if step % 500 == 0:
        avg = rl / 500
        print("  s{:>5d} loss={:.4f} ({:.0f}s)".format(step, avg, time.time() - t0))
        rl = 0; safe_save({'model':model.state_dict(),'optimizer':opt.state_dict(),
                           'scaler':scaler.state_dict(),'step':step}, CKP)

safe_save({'model':model.state_dict(),'optimizer':opt.state_dict(),
           'scaler':scaler.state_dict(),'step':TOTAL_STEPS}, CKP)

# ── Eval ──
print("\nEval (50 trials, 20 needles, 5 ctx x 13 depths)...")
model.eval()

EN = [
    ("银行卡号6225880137123456。","后四位？","3456"),
    ("手机解锁密码8473。","手机密码多少？","8473"),("保险箱密码9527切记。","保险箱密码？","9527"),
    ("今天是2026年8月15日晴。","今天几号？","2026"),("生日2025年3月20日。","生日哪天？","2025"),
    ("作者赵强出版社北京。","作者是谁？","赵强"),("邻居叫吴婷很友善。","邻居叫什么？","吴婷"),
    ("猫叫豆豆花猫。","猫叫什么名字？","豆豆"),("藏在地毯下面抽屉。","藏在哪里？","地毯"),
    ("在重庆买了房子。","哪个城市？","重庆"),("买了草莓和葡萄。","买了什么水果？","草莓"),
    ("会议室在512房间。","哪个房间？","512"),("车牌是沪B88345。","车牌多少？","沪B"),
    ("公司叫锐思软件。","公司叫什么？","锐思"),("吃了荔枝很甜。","吃了什么？","荔枝"),
    ("英文名叫Lucy。","英文名是什么？","Lucy"),("住302号房。","门牌号？","302"),
    ("在明德书院上学。","哪个学校？","明德"),("花盆下面有钥匙。","花盆下面有什么？","花盆"),
]
CTX_L = [512, 1024, 2048]
DEPT = [100, 80, 50, 30, 10]
res_all = {}

def _sample(logits):
    logits = logits / 0.7
    v, _ = torch.topk(logits, 40)
    logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
    return torch.multinomial(F.softmax(logits, dim=-1), 1)

@torch.no_grad()
def eval_one(model):
    for cl in CTX_L:
        for d in DEPT:
            hits = 0
            for _ in range(N_TRIALS):
                ns, q, ans = random.choice(EN)
                nids = voc.encode(ns); qids = voc.encode(q)
                cids = voc.encode(random.choice(chunks))
                cids = cids[:max(20, cl - len(nids) - len(qids) - 20)]
                pos = int(len(cids) * d / 100)
                pos = max(0, min(pos, len(cids) - 1))
                qa = [ist, uir] + qids + [uem, ist, agr]
                full = cids[:pos] + nids + cids[pos:] + qa
                if len(full) > cl: full = full[:cl]
                while len(full) < 4: full.append(0)
                x = torch.tensor([full], dtype=torch.long, device=DEV).clamp(0, vs - 1)
                # Prefill
                ash_st = [None] * LAYERS
                h_slow = torch.zeros(1, H, device=DEV)
                x_emb = model.em(x)
                ash_out = []
                for c0 in range(0, x.size(1), CHUNK):
                    c = x[:, c0:c0+CHUNK]
                    h = model.em(c)
                    for i, layer in enumerate(model.ash_layers):
                        h2, s = layer(h, ash_st[i]); h = h2 + h; ash_st[i] = s
                    for i in range(LAYERS):
                        if ash_st[i] is not None:
                            sn = ash_st[i].norm()
                            if sn > CAP: ash_st[i] = ash_st[i] * (CAP / sn)
                            ash_st[i] = ash_st[i] * DECAY
                    ash_out.append(h)
                x_ash = model.ash_norm(torch.cat(ash_out, dim=1))
                inp_seq = model.mem_inp(x_emb)
                H_slow = torch.zeros(1, x.size(1), H, device=DEV); prev = 0
                for t in range(0, x.size(1), K_SLOW):
                    h_slow = model.slow(inp_seq[:, t], h_slow, cap=None, decay=None)
                    H_slow[:, prev:t+1] = h_slow.unsqueeze(1); prev = t + 1
                if prev < x.size(1): H_slow[:, prev:] = h_slow.unsqueeze(1)
                x_mem = model.mem_proj(H_slow)
                cat = torch.cat([x_ash, x_mem], dim=-1)
                gate = model.fusion_gate(cat)
                fused = model.fusion_norm(gate * x_ash + (1-gate) * x_mem + x_emb)
                nt = _sample(model.head(fused)[:, -1, :])
                gen = torch.cat([x, nt], dim=1)
                # Generate
                for _ in range(59):
                    tok = gen[:, -1:]
                    h = model.em(tok)
                    for i, layer in enumerate(model.ash_layers):
                        layer.attn.model_flag = "infer"
                        h2, s = layer.attn(h, ash_st[i])
                        h1 = layer.norm(layer.alpha * layer.ffn(h2) + (1-layer.alpha) * h)
                        h = h1 + h; ash_st[i] = s
                    for i in range(LAYERS):
                        if ash_st[i] is not None:
                            sn = ash_st[i].norm()
                            if sn > CAP: ash_st[i] = ash_st[i] * (CAP / sn)
                            ash_st[i] = ash_st[i] * DECAY
                    x_ash_t = model.ash_norm(h[:, 0])
                    h_slow = model.slow(model.mem_inp(x_emb[:, -1:])[:,0], h_slow, cap=None, decay=None)
                    x_mem_t = model.mem_proj(h_slow)
                    cat_t = torch.cat([x_ash_t, x_mem_t], dim=-1)
                    gate_t = model.fusion_gate(cat_t)
                    fused_t = model.fusion_norm(gate_t * x_ash_t + (1-gate_t) * x_mem_t + x_emb[:, -1, :].squeeze(0))
                    nt = _sample(model.head(fused_t))
                    gen = torch.cat([gen, nt], dim=1)
                    if nt.item() == uem: break
                resp = voc.decode(gen[0].tolist()[len(full):]).strip()[:200]
                hits += 1 if ans in resp else 0
            acc = hits / N_TRIALS; res_all[(cl, d)] = acc
        lb = "{}K".format(cl // 1024) if cl >= 1024 else str(cl)
        avg = sum(res_all[(cl, d)] for d in DEPT) / len(DEPT)
        print("  {:>5}  avg={:.0%}".format(lb, avg)); sys.stdout.flush()
    return res_all

res = eval_one(model)
all_v = list(res.values())
print("\n  Hybrid ASH+Mem: avg={:.0%} max={:.0%}".format(sum(all_v)/len(all_v), max(all_v)))
print("\n  Depth avg:")
for d in DEPT:
    a = sum(res.get((cl,d),0) for cl in CTX_L) / len(CTX_L)
    print("  {:>3}%: {:.0%}".format(d, a))
print("\nDone.")
