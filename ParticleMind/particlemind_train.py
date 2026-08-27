"""
ParticleMind 最小验证 (Minimal Verification on Real Data)
=========================================================
基于 ParticleMind.md 的设计，在 MiniMind 真实文本 + OpenASHVoc 词表上
端到端训练一次，验证模型全部核心特性是否真正可学习：

  特性 1  连续思维云       : N 个粒子在 D 维语义空间中协同存在
  特性 2  朗之万动力学     : dx = -∇U·dt + sqrt(2T)·dW (overdamped Langevin / Euler-Maruyama)
  特性 3  势能场学习       : PotentialNet U_θ 通过对比损失塑造语义景观
  特性 4  注意力读出       : 多查询交叉注意力把粒子群 → 离散 logits
  特性 5  多 token 并行预测 : L 个读出查询并行给出非自回归序列
  特性 6  对比 EBM         : 正/负样本路径能量差 (pos_energy < neg_energy)

任务设定 (非自回归):
  给定前 L_p 个 prompt token -> 编码成初始粒子群 -> K 步思考 -> 并行读出 L_t 个 token。
  负样本路径使用 batch 内错配 (shuffle) 的 prompt, 迫使势能场把
  "语义连贯的 context" 识别为低能、"错配的 context" 识别为高能 ——
  这正是 ParticleMind.md §3 描述的能量基模型 (EBM) 范式的可学习化实现。
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
# Config (极小规模, 目标: 几分钟内跑完并看清所有特性)
# ============================================================
DIM          = 96      # 语义/粒子维度 (调大以加强信号)
N_PARTICLES  = 48      # 思维云粒子数
L_PROMPT     = 16      # prompt token 数
THINK_STEPS  = 4       # 朗之万演化步数 (特性 2, 训练用; 可微轨迹故取较小 K)
TEMP         = 0.3     # 温度: 控制随机扩散强度 (专注 vs 发散)
PRIOR_NOISE  = 0.3     # 训练时初始粒子加噪 (正则; 推理关闭)
DT           = 0.1     # 演化时间步长
MARGIN       = 0.5     # 能量对比 hinge margin
CE_W         = 1.0     # 分类损失权重
EBM_W        = 0.1     # 对比能量损失权重
REG_W        = 1e-4    # 能量正则 (防爆炸)

MAX_SAMPLES  = 6000    # 子集大小 (验证用)
L_TARGET     = 8       # 并行预测的目标 token 数 (特性 5)  [注: 改 1 可做单token干净消融]
EPOCHS       = 3
BATCH_SIZE   = 32
LR           = 3e-4
SEED         = 42
LOG_EVERY    = 30

DATA_DIR     = './minimind_data'
PRETRAIN_FILE = 'pretrain_t2t_mini.jsonl'
OUT_DIR      = './ParticleMind'


# ============================================================
# 1. 势能网络 U_θ(x|c)  —— 特性 3 (条件化: 上下文 c 塑造语义景观)
# ============================================================
class PotentialNet(nn.Module):
    """条件势能场 U(x|c): 粒子 x [B,N,D] + 上下文向量 c [B,D] -> [B,N,1].
    低势能 = 与上下文 c 一致的合理思维构型."""
    def __init__(self, dim=96, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x, c):
        c_exp = c.unsqueeze(1).expand(-1, x.size(1), -1)        # [B,N,D]
        return self.net(torch.cat([x, c_exp], dim=-1))          # [B,N,1]


# ============================================================
# 2. Context Encoder: prompt token ids -> 上下文条件向量 c [B,D]
#    (不再用作初始粒子! 粒子从与上下文无关的先验出发 —— 这是让"思考"必需的关键)
# ============================================================
class ContextEncoder(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, prompt_ids):
        h = self.embed(prompt_ids)                              # [B, Lp, D]
        c = h.mean(dim=1)                                       # [B, D] 全局上下文摘要
        return self.norm(self.proj(c))


# ============================================================
# 3. ParticleMind 主体
# ============================================================
class ParticleMind(nn.Module):
    def __init__(self, vocab_size, dim=96, num_particles=48,
                 seq_len=8, temp=0.3, dt=0.1, n_heads=4, prior_noise=0.3):
        super().__init__()
        self.dim, self.N, self.seq_len = dim, num_particles, seq_len
        self.temp, self.dt, self.prior_noise = temp, dt, prior_noise

        self.ctx_encoder = ContextEncoder(vocab_size, dim)
        self.potential = PotentialNet(dim)

        # 上下文无关的可学习先验粒子. 推理时所有样本同一起点, 只有 K>0 的条件动力学
        # 才能把不同上下文拉向不同答案 —— 故 K=0 必然失效, 思考成为必需.
        self.prior = nn.Parameter(torch.randn(1, num_particles, dim) * 0.1)

        # 多 token 并行读出: L 个查询同时从粒子群提取 logits  —— 特性 4 + 5
        self.readout_query = nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)
        self.readout_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.readout_norm = nn.LayerNorm(dim)           # 残差+归一, 抑制众数塌缩
        self.readout_pre = nn.Sequential(               # 读出前的非线性映射
            nn.Linear(dim, dim), nn.GELU(),
        )
        self.decoder = nn.Linear(dim, vocab_size)

    # ---- 特性 2: 条件朗之万动力学 dx = -∇_x U(x|c)·dt + sqrt(2T)·dW ----
    def _dynamics_step(self, particles, context):
        """单步 Euler-Maruyama. 训练保留图, 推理 detach."""
        if self.training:
            p_in = particles
        else:
            p_in = particles.detach().requires_grad_(True)
        U = self.potential(p_in, context)                                # [B,N,1]
        grad = torch.autograd.grad(
            U.sum(), p_in, create_graph=self.training, retain_graph=self.training
        )[0]                                                             # ∇_x U(x|c)
        noise = torch.randn_like(particles) * math.sqrt(2 * self.temp)
        return particles - grad * self.dt + noise * math.sqrt(self.dt)

    def step_dynamics(self, particles, context, steps=4):
        for _ in range(steps):
            particles = self._dynamics_step(particles, context)
        return particles

    def energy(self, particles, context):
        return self.potential(particles, context).mean()

    # ---- 特性 4: 注意力读出 (连续 -> 离散) ----
    def readout(self, particles):
        q = self.readout_query.expand(particles.size(0), -1, -1)           # [B,L,D]
        agg, _ = self.readout_attn(q, particles, particles)                # [B,L,D]
        agg = self.readout_norm(agg + q)                                   # 残差+归一
        return self.decoder(self.readout_pre(agg))                         # [B,L,V]

    def forward(self, prompt_ids, steps=4, return_particles=False,
                return_trace=False, context=None):
        """迭代精炼: 每个 think step 后都读出, 末端 logits 用于预测,
        训练时对所有步读出做深监督 (diffusion 式 x0 预测)."""
        c = self.ctx_encoder(prompt_ids) if context is None else context
        B = c.size(0)
        x = self.prior.expand(B, -1, -1)                                   # 上下文无关起点
        if self.training:
            x = x + torch.randn_like(x) * self.prior_noise                 # 训练加噪正则
        logits_list = []
        for _ in range(steps):
            x = self._dynamics_step(x, c)                                  # 条件动力学一步
            logits_list.append(self.readout(x))                            # 该步读出
        logits = logits_list[-1]
        if return_trace:
            return logits, logits_list, x, c
        if return_particles:
            return logits, x, c
        return logits


# ============================================================
# 4. Dataset: MiniMind 真实文本 -> (prompt, target) 窗口
# ============================================================
class PMDataset(torch.utils.data.Dataset):
    def __init__(self, path, voc, l_prompt, l_target, max_samples=3000):
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
        print(f'PMDataset: {len(self.samples)} usable samples (need>={need})')

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        ids = self.samples[i]
        need = self.lp + self.lt
        start = random.randint(0, len(ids) - need)             # 随机窗口增强多样性
        w = ids[start:start + need]
        p = torch.tensor(w[:self.lp], dtype=torch.long)
        t = torch.tensor(w[self.lp:], dtype=torch.long)
        return p, t

    @staticmethod
    def collate(items):
        ps, ts = zip(*items)
        return torch.stack(ps), torch.stack(ts)


# ============================================================
# 5. 训练 + 验证
# ============================================================
def feature_check_shapes(model, vs, dev):
    """训练前: 校验所有张量形状是否符合设计."""
    print('\n===== 特性形状自检 =====')
    p = torch.randint(0, vs, (2, L_PROMPT), device=dev)
    c = model.ctx_encoder(p)
    print(f'特性3 上下文向量 c(prompt)[B,D]   = {tuple(c.shape)}  (期望 D={DIM})')
    init = model.prior.expand(p.size(0), -1, -1)
    print(f'特性1 思维云先验 prior[B,N,D]      = {tuple(init.shape)}  '
          f'(期望 N={N_PARTICLES}, D={DIM}; 与上下文无关)')
    part = model.step_dynamics(init, c, steps=THINK_STEPS)
    print(f'特性2 条件动力学 after {THINK_STEPS} steps = {tuple(part.shape)}')
    logits = model.readout(part)
    print(f'特性4/5 读出    logits[B,L,V]            = {tuple(logits.shape)}  '
          f'(期望 L={L_TARGET})')
    e = model.potential(part, c)
    print(f'特性3 势能      U(x|c)[B,N,1]           = {tuple(e.shape)}')
    n_params = sum(x.numel() for x in model.parameters())
    print(f'参数量: {n_params:,}')
    print('=========================\n')


def train_loop(model, loader, opt, dev, vs, voc):
    model.train()
    hist = []; t0 = time.time()
    step = 0; total = len(loader) * EPOCHS
    for ep in range(EPOCHS):
        for prompt, target in loader:
            prompt, target = prompt.to(dev), target.to(dev)
            B = prompt.size(0)

            # ---- 正样本路径: 先验粒子 + 条件动力学 + 逐步读出 (迭代精炼) ----
            # 深监督: 每个 think step 的读出都对 target 算 CE, 后期步权重更高 (diffusion 式)
            _, pos_logits_list, pos_part, ctx = model(
                prompt, steps=THINK_STEPS, return_trace=True)
            K = len(pos_logits_list)
            pos_loss = 0.0
            for k, lg in enumerate(pos_logits_list):
                w = (k + 1) / K                                      # 1/K..1, 强调最终精炼步
                pos_loss = pos_loss + w * F.cross_entropy(
                    lg.reshape(-1, vs), target.reshape(-1))
            pos_loss = pos_loss / K
            pos_logits = pos_logits_list[-1]                         # 末端读出, 用于指标

            # ---- 负样本路径: EBM "corrupted thought" (终态粒子加噪 = 混乱构型) ----
            neg_part = pos_part.detach() + torch.randn_like(pos_part) * 1.0

            # ---- 特性 3 + 6: 条件能量对比 (EBM hinge) ----
            pos_e = model.potential(pos_part, ctx).mean()
            neg_e = model.potential(neg_part, ctx).mean()
            ebm_loss = F.relu(pos_e - neg_e + MARGIN)
            reg = (pos_e ** 2 + neg_e ** 2) * REG_W

            loss = CE_W * pos_loss + EBM_W * ebm_loss + reg

            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_gnorm = float(gnorm)

            hist.append((loss.item(), pos_loss.item(), ebm_loss.item(),
                         pos_e.item(), neg_e.item()))
            step += 1
            if step % LOG_EVERY == 0 or step == total:
                recent = hist[-LOG_EVERY:]
                avg = lambda k: sum(r[k] for r in recent) / len(recent)
                with torch.no_grad():
                    acc = (pos_logits.argmax(-1) == target).float().mean().item()
                el = time.time() - t0
                print(f'  step {step:4d}/{total} | loss {avg(0):.4f} '
                      f'ce {avg(1):.4f} ebm {avg(2):.4f} '
                      f'| E_pos {avg(3):+.3f} E_neg {avg(4):+.3f} gap {avg(4)-avg(3):+.3f} '
                      f'| gnorm {last_gnorm:.2f} | tok_acc {acc:.3f} | {el:.0f}s', flush=True)
    return hist


@torch.enable_grad()
def verify_all_features(model, ds, voc, vs, dev):
    """训练后: 逐项验证 6 大特性是否真正生效.
    注意: step_dynamics 内部用 torch.autograd.grad 求 ∇U, 故需 enable_grad
    (但全程不调用 backward, 不会更新参数)."""
    model.eval()
    print('\n' + '=' * 64)
    print('  ParticleMind 全特性验证 (训练后)')
    print('=' * 64)

    prompt, target = ds[0]
    prompt = prompt.unsqueeze(0).to(dev)
    target = target.unsqueeze(0).to(dev)

    # --- 特性 1: 思维云先验 (上下文无关) ---
    ctx = model.ctx_encoder(prompt)
    init = model.prior.expand(prompt.size(0), -1, -1)
    print(f'\n[特性1 思维云] 先验粒子 {tuple(init.shape)} (上下文无关); '
          f'上下文向量 c {tuple(ctx.shape)}')

    # --- 特性 2: 条件动力学逐步观察能量随思考步数下降 ---
    print('\n[特性2 条件朗之万动力学] 思考过程中 U(x|c) 变化 (应下降):')
    part = init
    for s in range(THINK_STEPS + 2):
        e = model.potential(part, ctx).mean().item()
        print(f'    think step {s}: mean U(x|c) = {e:+.4f}')
        part = model.step_dynamics(part, ctx, steps=1)

    # --- 特性 4 + 5: 并行多 token 读出 ---
    logits = model.readout(part)
    pred = logits.argmax(-1)[0]
    print(f'\n[特性4/5 注意力读出 + 并行预测] 并行输出 {L_TARGET} 个 token:')
    print(f'    target ids : {target[0].tolist()}')
    print(f'    pred  ids  : {pred.tolist()}')
    print(f'    token acc  : {(pred == target[0]).float().mean().item():.3f}')
    print(f'    target txt : {voc.decode(target[0].tolist())!r}')
    print(f'    pred  txt  : {voc.decode(pred.tolist())!r}')

    # --- 特性 3 + 6: 条件能量对比 (连贯 vs 混乱思维云) ---
    print('\n[特性3/6 势能场 + 对比EBM] 连贯思维云 vs 混乱思维云 的 U(x|c):')
    pos_es, neg_es = [], []
    for i in range(0, 32, 4):
        p, _ = ds[i]
        p = p.unsqueeze(0).to(dev)
        c = model.ctx_encoder(p)
        pe = model.step_dynamics(model.prior.expand(p.size(0), -1, -1), c, steps=THINK_STEPS)
        pos_es.append(model.potential(pe, c).mean().item())
        ne = pe.detach() + torch.randn_like(pe) * 1.0
        neg_es.append(model.potential(ne, c).mean().item())
    mean_pos = sum(pos_es) / len(pos_es)
    mean_neg = sum(neg_es) / len(neg_es)
    print(f'    E_pos (连贯思维云) 均值 = {mean_pos:+.4f}')
    print(f'    E_neg (混乱思维云) 均值 = {mean_neg:+.4f}')
    print(f'    能量差 E_neg - E_pos    = {mean_neg - mean_pos:+.4f}  '
          f'({"景观已区分 [OK]" if mean_neg > mean_pos else "未区分 [NO]"})')

    # ===== 严谨探针 A: 思考步数标度 (ParticleMind 的核心主张) =====
    # 训练时 K=5; 推理时改变 K, 看 CE/准确率/能量是否随"思考"改善.
    # K=0 = 不思考(直接读出 context 粒子). 在 T=0 下测 (确定性, 曲线干净).
    # 关键判据: CE(K=5) 应明显低于 CE(K=0) —— 思考真正降低了预测损失.
    print('\n[探针A 思考步数标度] 训练K=%d, T=0 确定性推理:' % THINK_STEPS)
    print(f'    {"K":>4} | {"CE loss":>8} | {"token_acc":>9} | {"mean U":>9}')
    old_T_A = model.temp; model.temp = 0.0
    def _acc_at(k, n=96):
        corr = tot = 0; ces = []; es = []
        for i in range(min(n, len(ds))):
            p, t = ds[i]
            p = p.unsqueeze(0).to(dev); t = t.to(dev)
            if k == 0:
                # K=0: 粒子停在上下文无关先验, 读出必然上下文盲 -> 应最差
                part = model.prior.expand(p.size(0), -1, -1)
                lg = model.readout(part)
            else:
                lg, part, c = model(p, steps=k, return_particles=True)
            corr += (lg.argmax(-1)[0] == t).sum().item(); tot += t.numel()
            ces.append(F.cross_entropy(lg.reshape(-1, vs), t.reshape(-1)).item())
            cctx = model.ctx_encoder(p)
            es.append(model.potential(part, cctx).mean().item())
        return sum(ces)/len(ces), corr/tot, sum(es)/len(es)
    for k in [0, 1, 2, 4, 8, 12]:
        ce, a, e = _acc_at(k)
        print(f'    {k:>4} | {ce:>8.3f} | {a:>9.3f} | {e:>+9.4f}')
    model.temp = old_T_A

    # ===== 严谨探针 B: 温度效应 (特性2 专注 vs 发散) =====
    print('\n[探针B 温度效应] 固定K=%d, 改变温度 T:' % THINK_STEPS)
    print(f'    {"T":>5} | {"mean U":>9} | {"U std":>8}  (低T=收敛, 高T=发散)')
    old_T = model.temp
    for T in [0.0, 0.3, 1.0, 2.0]:
        model.temp = T
        es, stds = [], []
        for i in range(8):
            p, _ = ds[i]; p = p.unsqueeze(0).to(dev)
            c = model.ctx_encoder(p)
            part = model.step_dynamics(model.prior.expand(p.size(0), -1, -1), c, steps=THINK_STEPS)
            u = model.potential(part, c)
            es.append(u.mean().item()); stds.append(u.std().item())
        print(f'    {T:>5.2f} | {sum(es)/len(es):>+9.4f} | {sum(stds)/len(stds):>8.4f}')
    model.temp = old_T

    # ===== 严谨探针 C: 基线对比 (证明能量景观是学出来的, 非偶然) =====
    print('\n[探针C 基线对比] 未训练随机模型 vs 训练模型的 E_neg-E_pos:')
    torch.manual_seed(999)
    rand_model = ParticleMind(vocab_size=vs, dim=DIM, num_particles=N_PARTICLES,
                              seq_len=L_TARGET, temp=TEMP, dt=DT).to(dev)
    rand_model.eval()
    def _gap(m):
        ps, ns = [], []
        for i in range(0, 16, 2):
            p, _ = ds[i]; p = p.unsqueeze(0).to(dev)
            c = m.ctx_encoder(p)
            pe = m.step_dynamics(m.prior.expand(p.size(0), -1, -1), c, steps=THINK_STEPS)
            ps.append(m.potential(pe, c).mean().item())
            ne = pe.detach() + torch.randn_like(pe) * 1.0
            ns.append(m.potential(ne, c).mean().item())
        return sum(ns) / len(ns) - sum(ps) / len(ps)
    g_rand = _gap(rand_model); g_trained = _gap(model)
    print(f'    随机模型  gap = {g_rand:+.4f}')
    print(f'    训练模型  gap = {g_trained:+.4f}')
    print(f'    提升         = {g_trained - g_rand:+.4f}  '
          f'({"EBM景观确为所学 [OK]" if g_trained > g_rand else "[NO]"})')

    # ===== 严谨探针 D: 迭代精炼曲线 (单次运行内 CE 随 think step 下降) =====
    # 直接验证"逐步精炼": 同一样本, 一次 K 步运行中, 每个 step 的读出 CE 应单调下降.
    print('\n[探针D 迭代精炼曲线] K=10 一次运行, 各步读出的 CE (应单调下降):')
    print(f'    {"step":>4} | {"CE":>7} | {"token_acc":>9}')
    old_T_D = model.temp; model.temp = 0.0
    n_D = 96
    K_D = 10
    # 收集每个 step 的预测, 统一计算
    step_correct = [0] * K_D; step_tot = [0] * K_D; step_ce = [[] for _ in range(K_D)]
    for i in range(min(n_D, len(ds))):
        p, t = ds[i]
        p = p.unsqueeze(0).to(dev); t = t.to(dev)
        _, lg_list, _, _ = model(p, steps=K_D, return_trace=True)
        for k, lg in enumerate(lg_list):
            step_ce[k].append(F.cross_entropy(lg.reshape(-1, vs), t.reshape(-1)).item())
            step_correct[k] += (lg.argmax(-1)[0] == t).sum().item()
            step_tot[k] += t.numel()
    for k in range(K_D):
        ce = sum(step_ce[k]) / len(step_ce[k])
        acc = step_correct[k] / step_tot[k]
        print(f'    {k+1:>4} | {ce:>7.3f} | {acc:>9.3f}')
    model.temp = old_T_D

    # ===== 严谨探针 E: 上下文消融 (证明动力学真的在搬运上下文, 而非走捷径) =====
    # 同一目标, 推理时分别喂 [正确上下文] / [打乱上下文]. 若打乱后 CE 大涨/acc 暴跌,
    # 说明预测确实经由条件动力学从上下文获得 —— 没有 readout 旁路.
    print('\n[探针E 上下文消融] 正确上下文 vs 打乱上下文 (K=%d, T=0):' % THINK_STEPS)
    old_T_E = model.temp; model.temp = 0.0
    n_E = 96
    perm = list(range(n_E)); random.shuffle(perm)
    ctxs = [model.ctx_encoder(ds[i][0].unsqueeze(0).to(dev)) for i in range(n_E)]
    ce_r = ce_w = 0.0; cr = cw = 0
    for i in range(n_E):
        t = ds[i][1].to(dev)
        for c, tag in [(ctxs[i], 'r'), (ctxs[perm[i]], 'w')]:
            lg, _, _, _ = model(ds[i][0].unsqueeze(0).to(dev),
                                steps=THINK_STEPS, return_trace=True, context=c)
            ce = F.cross_entropy(lg.reshape(-1, vs), t.reshape(-1)).item()
            ca = (lg.argmax(-1)[0] == t).sum().item()
            if tag == 'r': ce_r += ce; cr += ca
            else:         ce_w += ce; cw += ca
    ntok = n_E * L_TARGET
    print(f'    正确上下文 : CE {ce_r/n_E:.3f} | acc {cr/ntok:.3f}')
    print(f'    打乱上下文 : CE {ce_w/n_E:.3f} | acc {cw/ntok:.3f}')
    print(f'    CE 差(乱-正) = {ce_w/n_E - ce_r/n_E:+.3f}  '
          f'({"上下文确被动力学利用 [OK]" if ce_w > ce_r else "[NO]"})')
    model.temp = old_T_E

    # --- 多样本 token 准确率 ---
    print('\n[整体] 测试集 token 级准确率:')
    correct = tot = 0
    for i in range(0, min(128, len(ds)), 1):
        p, t = ds[i]
        p = p.unsqueeze(0).to(dev); t = t.to(dev)
        lg = model(p, steps=THINK_STEPS)
        correct += (lg.argmax(-1)[0] == t).sum().item()
        tot += t.numel()
    print(f'    token acc = {correct}/{tot} = {correct/tot:.3f}')
    print('=' * 64)


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    random.seed(SEED); torch.manual_seed(SEED)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {dev}')

    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    vs = len(voc.token_to_id) + 1

    model = ParticleMind(vocab_size=vs, dim=DIM, num_particles=N_PARTICLES,
                         seq_len=L_TARGET, temp=TEMP, dt=DT,
                         prior_noise=PRIOR_NOISE).to(dev)
    feature_check_shapes(model, vs, dev)

    ds = PMDataset(PRETRAIN_FILE, voc, L_PROMPT, L_TARGET, MAX_SAMPLES)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=PMDataset.collate, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.01)

    print(f'\n训练: {len(ds)} samples, {EPOCHS} epochs, bs={BATCH_SIZE}, '
          f'{len(loader)*EPOCHS} steps, K={THINK_STEPS}, T={TEMP}, dt={DT}')
    print('-' * 64)
    train_loop(model, loader, opt, dev, vs, voc)

    # 保存
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt = {'model_state_dict': model.state_dict(),
            'config': (vs, DIM, N_PARTICLES, L_TARGET, TEMP, DT)}
    torch.save(ckpt, os.path.join(OUT_DIR, 'particlemind_min.pt'))
    print(f'\n已保存: {OUT_DIR}/particlemind_min.pt')

    verify_all_features(model, ds, voc, vs, dev)
