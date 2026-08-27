"""
ParallelMind —— 并行推理模型 (Non-Autoregressive Iterative Decoder)
================================================================
目标: 真正实现"并行推理" —— 一次性并行预测 L 个 token, 用 K 步迭代精炼,
      每步条件于上一步的预测 (位置并行 / 迭代串行). 跳出 autoregressive 的 O(L) 串行.

设计动机 (来自 ParticleMind 实验的证伪结论):
  - 粒子云 + 朗之万动力学的"思考精炼"被证明无效 (能量梯度方向 ⊥ 读出方向).
  - 并行多 token 读出会塌缩到众数 —— 根因是各位置预测彼此独立、缺乏协调.
  - 本模型用两个已被证明有效的机制解决塌缩并实现真精炼:
      (1) 目标位置间 双向自注意力 —— 每个位置看到其它位置的当前预测, 协调去歧义;
      (2) 迭代条件精炼 —— step k+1 把 step k 的预测 (软嵌入) 作为输入, 每步真正改善;
      (3) 掩码增强训练 —— 随机 blank 部分位置, 让模型学会从部分信息填补 (Mask-Predict 式).

结构: prompt -> 交叉注意上下文; L 个目标位置 (初始=[MASK]) -> K 步精炼 -> 并行 L 个 logits.
"""
import os, sys, json, math, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, r'F:\OpenASH2605')
os.chdir(r'F:\OpenASH2605')
from open_ash_voc import OpenASHVoc
from config import agent_voc_path

# ============================================================
# Config
# ============================================================
DIM         = 224
N_LAYERS    = 5
N_HEADS     = 4
L_PROMPT    = 16
L_TARGET    = 8
REFINE_K    = 3
MASK_PROB   = 0.2
MAX_LEN     = 40

MAX_SAMPLES = 15000
EPOCHS      = 6
BATCH_SIZE  = 32
LR          = 3e-4
SEED        = 42
LOG_EVERY   = 30

DATA_DIR     = './minimind_data'
PRETRAIN_FILE = 'pretrain_t2t_mini.jsonl'
OUT_DIR      = './ParticleMind'


# ============================================================
# 1. Decoder Layer: 目标位置双向自注意 + 对 prompt 交叉注意 + FFN
# ============================================================
class DecoderLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)   # 目标位置间 (双向)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)  # 目标 -> prompt
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, ctx):
        a, _ = self.self_attn(x, x, x)
        x = self.norm1(x + a)
        b, _ = self.cross_attn(x, ctx, ctx)
        x = self.norm2(x + b)
        return self.norm3(x + self.ffn(x))


# ============================================================
# 2. ParallelMind 主体
# ============================================================
class ParallelMind(nn.Module):
    def __init__(self, vocab_size, dim=128, n_layers=2, n_heads=4, max_len=40):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.mask = nn.Parameter(torch.randn(dim) * 0.02)        # [MASK] 空白向量
        self.layers = nn.ModuleList([DecoderLayer(dim, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def decode(self, state, ctx):
        h = state
        for layer in self.layers:
            h = layer(h, ctx)
        return self.norm(h)

    def forward(self, prompt_ids, target_len=L_TARGET, steps=REFINE_K,
                target_ids=None, return_trace=False):
        B, Lp = prompt_ids.shape
        dev = prompt_ids.device
        pos = lambda a: self.pos(torch.arange(a[0], a[1], device=dev))
        ctx = self.embed(prompt_ids) + pos((0, Lp))
        pos_t = pos((Lp, Lp + target_len))
        mask_state = (self.mask + pos_t).unsqueeze(0).expand(B, -1, -1)

        state = mask_state
        logits_list = []
        for k in range(steps):
            h = self.decode(state, ctx)
            lg = self.head(h)
            logits_list.append(lg)
            if k < steps - 1:
                soft = lg.detach().softmax(-1) @ self.embed.weight
                state = soft + pos_t
                if self.training:
                    blank = torch.rand(B, target_len, device=dev) < MASK_PROB
                    state = torch.where(blank.unsqueeze(-1), mask_state, state)
                if self.training and target_ids is not None:
                    gt = self.embed(target_ids) + pos_t
                    use_gt = (torch.rand(B, target_len, device=dev) < 0.15).unsqueeze(-1)
                    state = torch.where(use_gt, gt, state)
        if return_trace:
            return logits_list[-1], logits_list
        return logits_list[-1]


# ============================================================
# 2b. AR 教师 (小因果 Transformer) —— 序列级知识蒸馏用
#     给每个前缀贪心续写 Lt 个 token, 得到比真实数据更平滑(少多模态)的教师序列.
# ============================================================
class ARTeacher(nn.Module):
    def __init__(self, vocab_size, dim=128, n_layers=4, n_heads=4, max_len=40):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, n_heads, dim * 4,
                                       batch_first=True, activation='gelu')
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        causal = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        for L in self.layers:
            h = L(h, src_mask=causal)
        return self.head(self.norm(h))                       # [B,T,V]

    @torch.no_grad()
    def generate(self, prefix, n):
        out = prefix
        for _ in range(n):
            nxt = self.forward(out)[:, -1, :].argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
        return out[:, prefix.size(1):]                       # [B,n]


def train_teacher(ds, dev, vs, epochs=2, bs=64, lr=3e-4, n_layers=6):
    """在窗口(prefix+target)上做 next-token 训练 (因果)."""
    teacher = ARTeacher(vs, dim=DIM, n_layers=n_layers, n_heads=N_HEADS, max_len=MAX_LEN).to(dev)
    # 每个样本的窗口(prompt+target)整段做 LM
    seqs = [torch.tensor(p + t, dtype=torch.long) for p, t in ds.pairs]
    seqs = torch.stack(seqs)
    opt = torch.optim.AdamW(teacher.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    teacher.train()
    n = len(seqs)
    print(f'\n--- 训练 AR 教师: {n} 序列, {epochs} epochs ---')
    for ep in range(epochs):
        idx = torch.randperm(n)
        tot = 0.0; cnt = 0
        for i in range(0, n - bs + 1, bs):
            b = seqs[idx[i:i + bs]].to(dev)
            lg = teacher(b[:, :-1])
            loss = F.cross_entropy(lg.reshape(-1, vs), b[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            opt.step()
            tot += loss.item(); cnt += 1
        print(f'  teacher epoch {ep+1}/{epochs} | loss {tot/cnt:.4f}')
    return teacher


@torch.no_grad()
def make_teacher_targets(teacher, ds, dev, bs=256):
    """对每个样本的前缀贪心续写 Lt 个 token -> 教师目标序列."""
    teacher.eval()
    out = []
    prompts = [ds[i][0] for i in range(len(ds))]
    for i in range(0, len(prompts), bs):
        b = torch.stack(prompts[i:i + bs]).to(dev)
        out.append(teacher.generate(b, ds.lt))
    return torch.cat(out, dim=0).cpu()                       # [N, Lt]


# ============================================================
# 2c. 预训练 FRSM V6 教师 (高质量, 已在完整 MiniMind 上训练至 26500 步)
# ============================================================
FRSM_TEACHER_PATH = r'C:\Users\Administrator\Downloads\frsm_v6_fast_final.pt'

def load_frsm_teacher(ckpt_path, vs, dev):
    from frsm_v6a_fast import FRSM_V6_Fast
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    d_model = ck.get('config_d_model', 830)
    ns = ck.get('config_num_scales', 4)
    teacher = FRSM_V6_Fast(vocab_size=vs, d_model=d_model, num_scales=ns)
    teacher.load_state_dict(ck['model_state_dict'])
    teacher.to(dev).eval()
    n = sum(p.numel() for p in teacher.parameters())
    print(f'  加载 FRSM 教师: d_model={d_model}, scales={ns}, {n:,} params')
    return teacher

class FRSMTeacherWrapper(nn.Module):
    """统一教师接口: generate(prefix, n) -> [B,n]."""
    def __init__(self, frsm):
        super().__init__()
        self.frsm = frsm
    @torch.no_grad()
    def generate(self, prefix, n):
        logits, H = self.frsm(prefix, return_state=True)
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        out = [nxt]
        for _ in range(n - 1):
            lg, H = self.frsm.generate_step(nxt, H)
            nxt = lg.argmax(-1, keepdim=True)
            out.append(nxt)
        return torch.cat(out, dim=1)                          # [B,n]


# ============================================================
# 3. Dataset (MiniMind 真实文本 -> prompt/target 窗口)
# ============================================================
class PMDataset(torch.utils.data.Dataset):
    def __init__(self, path, voc, l_prompt, l_target, max_samples=6000):
        self.voc = voc; self.lp = l_prompt; self.lt = l_target
        need = l_prompt + l_target
        self.samples = []
        with open(os.path.join(DATA_DIR, path), encoding='utf-8') as f:
            for line in f:
                if len(self.samples) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line).get('text', '')
                ids = voc.encode(text)
                if len(ids) >= need + 4:
                    self.samples.append(ids)
        # 固定窗口 (确定性): 知识蒸馏需要 prompt 与教师目标一一对应
        self.pairs = []
        for ids in self.samples:
            start = (len(ids) - need) // 2
            w = ids[start:start + need]
            self.pairs.append((w[:l_prompt], w[l_prompt:]))
        self.kd_target = None    # 若设置, __getitem__ 返回教师目标替代真实目标
        print(f'PMDataset: {len(self.samples)} usable samples (need>={need})')

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        p, t = self.pairs[i]
        p = torch.tensor(p, dtype=torch.long)
        t = self.kd_target[i] if self.kd_target is not None else torch.tensor(t, dtype=torch.long)
        return p, t


    @staticmethod
    def collate(items):
        ps, ts = zip(*items)
        return torch.stack(ps), torch.stack(ts)


# ============================================================
# 4. 训练 (深监督: 每个 refine step 都算 CE)
# ============================================================
def train_loop(model, loader, opt, dev, vs):
    model.train()
    hist = []; t0 = time.time()
    step = 0; total = len(loader) * EPOCHS
    for ep in range(EPOCHS):
        for prompt, target in loader:
            prompt = prompt.to(dev); target = target.to(dev)
            lg_final, lg_list = model(prompt, target_len=L_TARGET, steps=REFINE_K,
                                      target_ids=target, return_trace=True)
            K = len(lg_list)
            loss = 0.0
            for k, lg in enumerate(lg_list):
                w = (k + 1) / K
                loss = loss + w * F.cross_entropy(lg.reshape(-1, vs), target.reshape(-1))
            loss = loss / K
            prob = lg_final.softmax(-1)
            avg_p = prob.mean(dim=1)
            ent = -(avg_p * (avg_p + 1e-8).log()).sum(-1)
            loss = loss - ent.mean() * 0.05

            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            with torch.no_grad():
                acc = (lg_final.argmax(-1) == target).float().mean().item()
            hist.append((loss.item(), acc))
            step += 1
            if step % LOG_EVERY == 0 or step == total:
                recent = hist[-LOG_EVERY:]
                avg_l = sum(r[0] for r in recent) / len(recent)
                avg_a = sum(r[1] for r in recent) / len(recent)
                print(f'  step {step:4d}/{total} | loss {avg_l:.4f} '
                      f'| gnorm {float(gnorm):.2f} | tok_acc {avg_a:.3f} '
                      f'| {time.time()-t0:.0f}s', flush=True)
    return hist


# ============================================================
# 5. 验证 (并行推理的严格探针)
# ============================================================
@torch.no_grad()
def evaluate(model, ds, voc, vs, dev, kd_targets=None, teacher=None):
    model.eval()
    print('\n' + '=' * 64)
    print('  ParallelMind 并行推理验证 (序列级知识蒸馏后)')
    print('=' * 64)

    # 真实目标 (评估基准)
    real_t = lambda i: torch.tensor(ds.pairs[i][1])
    # --- 示例预测 ---
    p = torch.tensor(ds.pairs[0][0]).unsqueeze(0).to(dev)
    t = real_t(0).to(dev)
    lg_final, lg_list = model(p, target_len=L_TARGET, steps=REFINE_K, return_trace=True)
    pred = lg_final.argmax(-1)[0]
    print(f'\n[示例] 并行预测 {L_TARGET} 个 token (K={REFINE_K} 步精炼):')
    print(f'    real target : {t.tolist()}')
    print(f'    pred        : {pred.tolist()}')
    if kd_targets is not None:
        print(f'    teacher seq : {kd_targets[0].tolist()}')
    print(f'    real txt: {voc.decode(t.tolist())!r}')
    print(f'    pred txt: {voc.decode(pred.tolist())!r}')

    # --- 探针 A: 精炼步数标度 (并行推理的核心: 更多精炼步是否更好) ---
    print('\n[探针A 精炼步数标度] 推理时改变 K:')
    print(f'    {"K":>3} | {"CE":>7} | {"token_acc":>9} | {"exact(L位全对)":>12}')
    def _acc_at(k, n=128):
        ce = 0; corr = 0; tot = 0; exact = 0
        for i in range(min(n, len(ds))):
            pp, tt = ds[i]; pp = pp.unsqueeze(0).to(dev); tt = tt.to(dev)
            lg = model(pp, target_len=L_TARGET, steps=k)
            ce += F.cross_entropy(lg.reshape(-1, vs), tt.reshape(-1)).item()
            pr = lg.argmax(-1)[0]
            corr += (pr == tt).sum().item(); tot += tt.numel()
            exact += (pr == tt).all().float().item()
        return ce/min(n,len(ds)), corr/tot, exact/min(n,len(ds))
    for k in [1, 2, 4, 8, 12]:
        ce, a, ex = _acc_at(k)
        print(f'    {k:>3} | {ce:>7.3f} | {a:>9.3f} | {ex:>12.3f}')

    # --- 探针 D: 迭代精炼曲线 (单次运行内 CE 应随步数下降) ---
    print('\n[探针D 迭代精炼曲线] K=12 一次运行, 各步读出 CE (应单调下降):')
    print(f'    {"step":>4} | {"CE":>7} | {"token_acc":>9}')
    K_D = 12; n_D = 128
    sc = [0.0]*K_D; sa = [0]*K_D; st = [0]*K_D
    for i in range(min(n_D, len(ds))):
        pp, tt = ds[i]; pp = pp.unsqueeze(0).to(dev); tt = tt.to(dev)
        _, lg_list = model(pp, target_len=L_TARGET, steps=K_D, return_trace=True)
        for k, lg in enumerate(lg_list):
            sc[k] += F.cross_entropy(lg.reshape(-1, vs), tt.reshape(-1)).item()
            sa[k] += (lg.argmax(-1)[0] == tt).sum().item()
            st[k] += tt.numel()
    n = min(n_D, len(ds))
    for k in range(K_D):
        print(f'    {k+1:>4} | {sc[k]/n:>7.3f} | {sa[k]/st[k]:>9.3f}')

    # --- 探针 E: 上下文消融 (打乱 prompt 应使预测变差) ---
    print('\n[探针E 上下文消融] 正确 prompt vs 打乱 prompt:')
    n_E = 128; perm = list(range(n_E)); random.shuffle(perm)
    cer = cew = cr = cw = 0
    for i in range(min(n_E, len(ds))):
        pp, tt = ds[i]; tt = tt.to(dev)
        for p_in, tag in [(pp, 'r'), (ds[perm[i]][0], 'w')]:
            lg = model(p_in.unsqueeze(0).to(dev), target_len=L_TARGET, steps=REFINE_K)
            ce = F.cross_entropy(lg.reshape(-1, vs), tt.reshape(-1)).item()
            ca = (lg.argmax(-1)[0] == tt).sum().item()
            if tag == 'r': cer += ce; cr += ca
            else:         cew += ce; cw += ca
    ntok = n * L_TARGET; n = min(n_E, len(ds))
    print(f'    正确 prompt: CE {cer/n:.3f} | acc {cr/(n*L_TARGET):.3f}')
    print(f'    打乱 prompt: CE {cew/n:.3f} | acc {cw/(n*L_TARGET):.3f}')

    # --- 探针 F: 并行推理速度优势 (核心卖点) ---
    print('\n[探针F 并行 vs 自回归 延迟] (batch=32, L_target=8)')
    pb = torch.randint(0, vs, (32, L_PROMPT), device=dev)
    model.eval()
    # 并行(K=REFINE_K)
    torch.cuda.synchronize() if dev.type == 'cuda' else None
    t0 = time.time()
    for _ in range(20):
        _ = model(pb, target_len=L_TARGET, steps=REFINE_K)
    torch.cuda.synchronize() if dev.type == 'cuda' else None
    t_parK = (time.time() - t0) / 20
    # 并行(K=1 最快模式)
    t0 = time.time()
    for _ in range(20):
        _ = model(pb, target_len=L_TARGET, steps=1)
    torch.cuda.synchronize() if dev.type == 'cuda' else None
    t_par1 = (time.time() - t0) / 20
    # 自回归(逐 token 串行)
    t0 = time.time()
    for _ in range(20):
        cur = pb
        for _ in range(L_TARGET):
            _ = model(cur, target_len=1, steps=1)
    torch.cuda.synchronize() if dev.type == 'cuda' else None
    t_ar = (time.time() - t0) / 20
    print(f'    并行 K=1 (最快) : {t_par1*1000:.1f} ms')
    print(f'    并行 K={REFINE_K}       : {t_parK*1000:.1f} ms')
    print(f'    自回归 (串行)    : {t_ar*1000:.1f} ms')
    print(f'    加速比 K=1/AR   : {t_ar/t_par1:.1f}x')

    # --- 整体: 对真实目标 vs 教师目标的 token/exact 命中 ---
    print('\n[整体命中] NAR 预测 vs 真实续写 / vs 教师续写:')
    n = min(256, len(ds))
    cr = ctr = ex_r = ex_t = 0; tot = 0
    for i in range(n):
        pp = torch.tensor(ds.pairs[i][0]).unsqueeze(0).to(dev)
        pr = model(pp, target_len=L_TARGET, steps=REFINE_K).argmax(-1)[0]
        rt = real_t(i).to(dev)
        cr += (pr == rt).sum().item(); tot += rt.numel()
        ex_r += (pr == rt).all().float().item()
        if kd_targets is not None:
            tt = kd_targets[i].to(dev)
            ctr += (pr == tt).sum().item()
            ex_t += (pr == tt).all().float().item()
    print(f'    vs 真实: token_acc {cr/tot:.3f} | exact {ex_r/n:.3f}')
    if kd_targets is not None:
        print(f'    vs 教师: token_acc {ctr/tot:.3f} | exact {ex_t/n:.3f}')

    # ===== 补充指标: PPL / BLEU / Teacher Fluency =====
    n_eval = min(256, len(ds))
    reals_txt = [voc.decode(ds.pairs[i][1]) for i in range(n_eval)]
    with torch.no_grad():
        # 收集 NAR 预测
        preds_ids = [model(torch.tensor(ds.pairs[i][0]).unsqueeze(0).to(dev),
                           target_len=L_TARGET, steps=REFINE_K).argmax(-1)[0].cpu()
                     for i in range(n_eval)]
    preds_txt = [voc.decode(p.tolist()) for p in preds_ids]

    # Perplexity (使用探针 D 中 K=1 的 CE)
    ce_k1 = 0.0
    for i in range(min(n_eval, len(ds))):
        pp, tt = ds[i]; pp = pp.unsqueeze(0).to(dev); tt = tt.to(dev)
        lg = model(pp, target_len=L_TARGET, steps=1)
        ce_k1 += F.cross_entropy(lg.reshape(-1, vs), tt.reshape(-1)).item()
    ce_k1 /= min(n_eval, len(ds))
    ppl = math.exp(ce_k1)

    # BLEU-1 / BLEU-2 (字符级, 中文适用)
    def _bleu(references, candidates, ngram):
        matched = total = 0
        for ref, cand in zip(references, candidates):
            r_grams = set(ref[i:i+ngram] for i in range(len(ref)-ngram+1))
            c_grams = [cand[i:i+ngram] for i in range(len(cand)-ngram+1)]
            matched += sum(1 for g in c_grams if g in r_grams)
            total += max(1, len(c_grams))
        return matched / total if total > 0 else 0.0

    bleu1 = _bleu(reals_txt, preds_txt, 1)  # 字符 uni-gram
    bleu2 = _bleu(reals_txt, preds_txt, 2)  # 字符 bi-gram

    # Teacher Fluency (用教师给 NAR 输出打分)
    tf_nar = tf_real = 0.0
    if teacher is not None:
        for i in range(min(n_eval // 2, len(ds) // 2)):
            p_ids = torch.tensor(ds.pairs[i][0], dtype=torch.long, device=dev)
            # NAR 输出续写被教师认可的程度
            nar_out = preds_ids[i].to(dev)
            seq_nar = torch.cat([p_ids, nar_out]).unsqueeze(0)
            tf_nar += F.cross_entropy(
                teacher(seq_nar)[0, -L_TARGET:, :].reshape(-1, vs),
                nar_out.unsqueeze(0).reshape(-1)).item()
            # 真实续写被教师认可的程度 (baseline)
            rt = torch.tensor(ds.pairs[i][1], device=dev)
            seq_real = torch.cat([p_ids, rt]).unsqueeze(0)
            tf_real += F.cross_entropy(
                teacher(seq_real)[0, -L_TARGET:, :].reshape(-1, vs),
                rt.unsqueeze(0).reshape(-1)).item()
        nf = min(n_eval // 2, len(ds) // 2)
        tf_nar /= nf; tf_real /= nf

    print(f'\n[补充指标] 自动化质量评估 (n={n_eval}):')
    print(f'    Perplexity (K=1)         = {ppl:.1f}')
    print(f'    BLEU-1 (char unigram)    = {bleu1:.3f}')
    print(f'    BLEU-2 (char bigram)     = {bleu2:.3f}')
    if teacher is not None:
        print(f'    教师流畅度 CE (NAR续写)   = {tf_nar:.2f}  (越低越流畅)')
        print(f'    教师流畅度 CE (真实续写)   = {tf_real:.2f}  (oracle基线)')
        print(f'    NAR vs 真实 流畅度比率     = {tf_nar/tf_real:.2f}  (接近1.0=接近人类水平)')

    print('=' * 64)


# ============================================================
# 6. Accept/Reject 验证 (补救 1: 半无损并行推理)
# ============================================================
@torch.no_grad()
def accept_reject_analysis(model, teacher, ds, dev, voc, vs, n=384):
    """用 AR 教师逐 token 验证 NAR 输出, 统计可接受率。
    对比三种验证策略:
      - strict: AR argmax == NAR token (最严)
      - soft:   AR prob(NAR token) >= 0.5 * AR prob(top1) (宽松)
      - conf:   NAR 置信度 > 0.7 (纯粹 NAR 自评, 不依赖 AR)
    """
    model.eval(); teacher.eval()

    def run_strategy(name, accept_fn, threshold_desc):
        ppos_ok = [0]*L_TARGET; block_dist = [0]*(L_TARGET+1); total = 0
        for i in range(min(n, len(ds))):
            p_ids = torch.tensor(ds.pairs[i][0], dtype=torch.long, device=dev).unsqueeze(0)
            lg = model(p_ids, target_len=L_TARGET, steps=1)
            nar_tok = lg.argmax(-1)[0]; nar_prob = lg.softmax(-1)[0]
            nar_conf = nar_prob.gather(-1, nar_tok.unsqueeze(-1)).squeeze(-1)
            cur = p_ids; accepted = 0
            for j in range(L_TARGET):
                ar_lg = teacher(cur)[0, -1, :]
                ar_prob = ar_lg.softmax(-1)
                ar_top1 = ar_lg.argmax().item()
                nt = nar_tok[j].item(); nc = nar_conf[j].item()
                if accept_fn(ar_top1, ar_prob, nt, nc):
                    accepted += 1; cur = torch.cat([cur, nar_tok[j:j+1].unsqueeze(0)], dim=1)
                    ppos_ok[j] += 1
                else:
                    break
            block_dist[accepted] += 1; total += 1
        saved = sum(k * block_dist[k] for k in range(L_TARGET+1))
        total_pos = total * L_TARGET
        ar_steps = total_pos - saved + total
        spd = total_pos / ar_steps if ar_steps > 0 else float('inf')
        return ppos_ok, block_dist, saved/total, spd, block_dist[0]/total

    strategies = [
        ('Strict(argmax==)', lambda t1,p,nt,nc: t1==nt, 'AR top1 完全一致'),
        ('Soft(p>=0.5*top1)', lambda t1,p,nt,nc: (p[nt].item() >= p[t1].item()*0.5),
         'AR 认为 NAR token 至少还不错'),
        ('Conf(>0.7) ',    lambda t1,p,nt,nc: nc>0.7, 'NAR 自评高置信'),
    ]

    print('\n' + '=' * 64)
    print('  补救1: Accept/Reject 验证实验 (三种策略对比)')
    print('=' * 64)
    print(f'  验证样本数: {n}')

    for name, fn, desc in strategies:
        ppos, bdist, mean_acc, speedup, full_rej = run_strategy(name, fn, desc)
        print(f'\n  --- {name} ---  [{desc}]')
        print(f'    逐位置: {", ".join(f"{ppos[j]/max(1,n):.2f}" for j in range(L_TARGET))}')
        print(f'    全拒率: {full_rej:.1%}  |  平均 Accept: {mean_acc:.2f}  |  '
              f'有效加速: {speedup:.2f}×')
    print('=' * 64)
if __name__ == '__main__':
    random.seed(SEED); torch.manual_seed(SEED)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {dev}')
    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    vs = len(voc.token_to_id) + 1

    ds = PMDataset(PRETRAIN_FILE, voc, L_PROMPT, L_TARGET, MAX_SAMPLES)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=PMDataset.collate, drop_last=True)

    # ===== 阶段 1: 训练 AR 教师 (足够的 epoch 以获得合理续写质量) =====
    teacher = train_teacher(ds, dev, vs, epochs=25, bs=64, lr=3e-4, n_layers=6)

    # ===== 阶段 2: 生成教师序列 (序列级知识蒸馏目标) =====
    kd_targets = make_teacher_targets(teacher, ds, dev)
    # 展示一条教师续写样本
    print(f'  教师续写示例: prefix...{ds.pairs[0][0][-3:]} -> '
          f'teacher{kd_targets[0].tolist()} | real{ds.pairs[0][1]}')

    # ===== 阶段 3: 用教师序列训练 ParallelMind (序列级 KD) =====
    ds.kd_target = kd_targets
    model = ParallelMind(vocab_size=vs, dim=DIM, n_layers=N_LAYERS,
                         n_heads=N_HEADS, max_len=MAX_LEN).to(dev)
    print(f'ParallelMind: {sum(p.numel() for p in model.parameters()):,} params')
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.01)
    print(f'\n训练(NAR, 序列级KD): {len(ds)} samples, {EPOCHS} epochs, bs={BATCH_SIZE}, K={REFINE_K}')
    print('-' * 64)
    train_loop(model, loader, opt, dev, vs)

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(),
                'config': (vs, DIM, N_LAYERS, N_HEADS, L_TARGET, REFINE_K)},
               os.path.join(OUT_DIR, 'parallel_mind.pt'))
    print(f'\n已保存: {OUT_DIR}/parallel_mind.pt')

    # 评估时用真实目标 (关闭 KD 覆盖)
    ds.kd_target = None
    evaluate(model, ds, voc, vs, dev, kd_targets=kd_targets, teacher=teacher)

    # ===== 补救 1 验证: Accept/Reject 半无损加速 =====
    accept_reject_analysis(model, teacher, ds, dev, voc, vs)
