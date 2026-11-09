# -*- coding: utf-8 -*-
"""v5 core: 在 minimind 真实语料上复现并扩展弹道外推 (ballistic extrapolation) 管线.

模型族 (承接 v3/v4):
  A^(k)_{wv} = Σ_j γ_kj(R_j,σ_j) φ_{wj}(v),   γ_kj ∝ exp(-(k-R_j)²/(2σ_j²))
  预测: log P(v|h) ∝ log π_v + Σ_k ω_k · log(A^(k)_{x_{t+1-k},v}/π_v)   (ω 在 dev 标定)
新增评估:
  - 表级外推恢复率 ρ(k) = cos(Â^(k)-π, A^(k)-π)   (对未见过的 k 直接量测外推质量)
"""
import json, zipfile, pickle, io, os
import numpy as np

DATA_JSONL = r"F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl"
PT_SMALL   = r"F:\OpenASH2605\minimind_data\pretrain_cached_30000_384.pt"

# ---------------- 数据 ----------------

def load_docs(path=DATA_JSONL, max_bytes=200_000_000, target_chars=25_000_000):
    """逐行读 minimind 预训练 jsonl, 每行是一个文档; 读到 target_chars 或文件尾为止."""
    with open(path, 'rb') as f:
        buf = f.read(max_bytes)
    buf = buf[:buf.rfind(b'\n')]
    docs, total = [], 0
    for line in buf.split(b'\n'):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line.decode('utf-8'))
        except Exception:
            continue
        t = d.get('text', '')
        if t:
            docs.append(t); total += len(t)
        if total >= target_chars:
            break
    return docs, total

def build_vocab(docs, V):
    from collections import Counter
    cnt = Counter()
    for t in docs:
        cnt.update(t)
    chars = [c for c, _ in cnt.most_common(V - 1)]
    stoi = {c: i + 1 for i, c in enumerate(chars)}   # 0 = UNK
    return chars, stoi

def encode_docs(docs, stoi):
    parts, bounds, pos = [], [], 0
    for t in docs:
        bounds.append(pos)
        parts.append(np.fromiter((stoi.get(c, 0) for c in t), dtype=np.int32, count=len(t)))
        pos += len(t)
    return np.concatenate(parts), np.asarray(bounds, dtype=np.int64)

def doc_ids_of(total_len, bounds):
    """切段时第一个文档可能从段中间开始 (bounds[0]>0): 把段首尾文档也各自编号"""
    bounds = np.asarray(bounds, dtype=np.int64)
    if len(bounds) == 0 or bounds[0] > 0:
        bounds = np.concatenate([[0], bounds])
    lens = np.diff(np.append(bounds, total_len))
    return np.repeat(np.arange(len(lens), dtype=np.int64), lens)

# ---------------- 统计量 ----------------

def unigram(ids, V):
    c = np.bincount(ids, minlength=V).astype(np.float64) + 1.0
    return (c / c.sum()).astype(np.float64)

def build_A(ids, did, K, V, pi, alpha=1.0):
    """A[k][w,v] = P(x_{t+k}=v | x_t=w), 行内平滑 (cnt + α·π)/(rowsum + α). 只统计文档内部配对."""
    n = len(ids)
    A = np.empty((K + 1, V, V), dtype=np.float32)
    a64 = ids.astype(np.int64)
    for k in range(1, K + 1):
        a, b = a64[:n - k], a64[k:]
        same = did[:n - k] == did[k:]
        idx = a[:len(a)] * V + b
        cnt = np.bincount(idx[same], minlength=V * V).astype(np.float64).reshape(V, V)
        rows = cnt.sum(1, keepdims=True)
        A[k] = ((cnt + alpha * pi[None, :]) / (rows + alpha)).astype(np.float32)
    return A

def signal_curve(A, pi):
    """P1: 每个 lag 携带的结构强度 ||A^(k)-π||_F"""
    X = A[1:].astype(np.float64) - pi[None, None, :]
    return np.linalg.norm(X.reshape(len(X), -1), axis=1)

def adjacent_dist(A):
    """P2: 相邻 lag 表差异 ||A^(k)-A^(k+1)||_F"""
    D = np.diff(A[1:].astype(np.float64), axis=0)
    return np.linalg.norm(D.reshape(len(D), -1), axis=1)

def joint_spectrum(A, pi, ncomp=8):
    """P3: 跨 k 联合谱 (把 K 张表居中后展平成 K×V² 矩阵做 SVD)"""
    X = (A[1:].astype(np.float64) - pi[None, None, :]).reshape(len(A) - 1, -1)
    s = np.linalg.svd(X, compute_uv=False)
    cum = np.cumsum(s ** 2) / np.sum(s ** 2)
    return s, cum[:ncomp]

# ---------------- 标定与评估 ----------------

def sample_positions(ids, did, K, n_pos, rng):
    """采样预测位置 t (预测 ids[t+1]); 要求窗口 [t+1-K, t+1] 都在同一文档内."""
    n = len(ids)
    cand = np.arange(K, n - 1, dtype=np.int64)
    ok = did[cand - (K - 1)] == did[cand + 1]
    cand = cand[ok]
    if len(cand) > n_pos:
        cand = np.sort(rng.choice(cand, size=n_pos, replace=False))
    return cand

def lag_features(ids, pos, A, pi, Kuse):
    """F_k[t,v] = log(A^(k)[x_{t+1-k}, v]/π_v), 返回 Kuse 个 (T,V) 数组"""
    F = []
    for k in range(1, Kuse + 1):
        w = ids[pos + 1 - k]
        F.append(np.log(A[k][w].astype(np.float32) / pi[None, :].astype(np.float32)))
    return F

def calib_weights(F_list, y, pi, maxiter=200):
    """dev 上标定 ω: 最小化 softmax NLL (解析梯度 + L-BFGS)"""
    from scipy.optimize import minimize
    T, V = F_list[0].shape
    lpi = np.log(pi).astype(np.float32)
    base = np.tile(lpi, (T, 1))
    yv = np.arange(V) == y[:, None]
    yidx = y.astype(np.int64)
    Kuse = len(F_list)

    def nll_grad(om):
        s = base.copy()
        for k in range(Kuse):
            s += om[k] * F_list[k]
        s -= s.max(1, keepdims=True)
        p = np.exp(s); Z = p.sum(1, keepdims=True); p /= Z
        nll = -np.log(np.maximum(p[np.arange(T), yidx], 1e-12)).mean()
        R = p.copy(); R[yv] -= 1.0; R /= T
        g = np.array([(R * F_list[k]).sum(dtype=np.float64) for k in range(Kuse)])
        return nll, g

    om0 = np.full(Kuse, 1.0 / Kuse)
    r = minimize(nll_grad, om0, jac=True, method='L-BFGS-B',
                 options=dict(maxiter=maxiter))
    return r.x, float(r.fun)

def eval_ppl(ids, pos, A, pi, om):
    """测试位置上的困惑度 (分块流式)"""
    lpi = np.log(pi).astype(np.float32)
    Kuse = len(om)
    tot, n = 0.0, 0
    for s in range(0, len(pos), 4096):
        p = pos[s:s + 4096]
        y = ids[p + 1].astype(np.int64)
        cur = np.tile(lpi, (len(p), 1))
        for k in range(1, Kuse + 1):
            cur += om[k - 1] * np.log(A[k][ids[p + 1 - k]].astype(np.float32) / pi[None, :])
        cur -= cur.max(1, keepdims=True)
        lp = cur[np.arange(len(p)), y] - np.log(np.exp(cur).sum(1))
        tot += -lp.sum(); n += len(p)
    return np.exp(tot / n), tot / n

# ---------------- 弹道模型 ----------------

def gamma_mat(R, sig, Kmax):
    k = np.arange(1, Kmax + 1, dtype=np.float64)[:, None]
    g = np.exp(-(k - R[None, :]) ** 2 / (2 * sig[None, :] ** 2))
    return g / np.maximum(g.sum(1, keepdims=True), 1e-300)

def fit_R_sigma(A, Kfit, J, n_rows=400, restarts=4, seed=1, maxiter=600):
    """在频率最高的 n_rows 个源 token 行上拟合 (R_j, σ_j), 加速且无偏.
    v6.3 鲁棒性: R 越界惩罚 + σ 下限 + 发散自动换种子重启"""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    Ak = A[1:Kfit + 1].astype(np.float64)          # (Kfit, V, V)
    rowmass = Ak.sum(axis=(0, 2))
    rows = np.argsort(-rowmass)[:n_rows]
    Asub = Ak[:, rows, :]                          # (Kfit, n_rows, V)

    def obj(p):
        R, sig = p[:J], np.exp(p[J:])
        pen = 0.0
        if np.any(R < -2.0) or np.any(R > Kfit + 6):
            pen += 1e4 * float(np.sum(np.clip(-2.0 - R, 0, None) + np.clip(R - (Kfit + 6), 0, None)) + 1)
        if np.any(sig < 0.15):
            pen += 1e4 * float(np.sum(np.clip(0.15 - sig, 0, None)) + 1)
        g = gamma_mat(R, sig, Kfit)
        phi = np.einsum('kj,kwv->jwv', g, Asub) / np.maximum(g.sum(0), 1e-12)[:, None, None]
        Ah = np.einsum('kj,jwv->kwv', g, phi)
        d = Asub - Ah
        return float((d * d).sum()) + pen

    best = None
    seed_try = 0
    while seed_try < restarts + 3:
        t = seed_try
        R0 = np.linspace(1, Kfit, J) + (rng.normal(0, 0.5, J) if t else 0.0)
        p0 = np.concatenate([np.clip(R0, 0.5, Kfit + 2), np.log(np.maximum(np.full(J, 1.0), 0.2))])
        r = minimize(obj, p0, method='Nelder-Mead',
                     options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-8))
        Rr, sgr = r.x[:J], np.exp(r.x[J:])
        ok = bool(np.all(Rr >= -2.0) and np.all(Rr <= Kfit + 6) and np.all(sgr >= 0.15))
        if ok and (best is None or r.fun < best.fun):
            best = r
        seed_try += 1
        if best is not None and seed_try >= restarts:
            break
    if best is None:                               # 全部发散: 回退网格初始化解
        Rr = np.linspace(1, Kfit, J); sgr = np.full(J, 1.0)
        return Rr, sgr
    return best.x[:J], np.exp(best.x[J:])

def make_phi(A, Kfit, R, sig):
    """固定 γ, 在全表上闭式解 φ_j"""
    g = gamma_mat(R, sig, Kfit)
    Ak = A[1:Kfit + 1].astype(np.float64)
    phi = np.einsum('kj,kwv->jwv', g, Ak) / np.maximum(g.sum(0)[:, None, None], 1e-12)
    return g, phi.astype(np.float32)

def make_tables(R, sig, phi, K):
    """γ 在 k=1..K 上取值 (含外推区), 混合 φ 后行归一化. 返回 (K+1,V,V), 索引 j 对应 lag j, [0] 空置"""
    Vv, Vv2 = phi.shape[1], phi.shape[2]
    Ah = np.empty((K + 1, Vv, Vv2), dtype=np.float32)
    g = gamma_mat(R, sig, K)
    for k in range(1, K + 1):
        m = np.tensordot(g[k - 1], phi, axes=(0, 0))   # (V,V) float64
        m /= m.sum(1, keepdims=True)
        Ah[k] = m.astype(np.float32)
    return Ah

# ---------------- 表级指标 ----------------

def rho_table(Ahat_k, Atrue_k, pi):
    """外推恢复率: cos(Â-π, A-π), 并返回幅度比 ||Â-π||/||A-π||"""
    x = Ahat_k.astype(np.float64) - pi[None, :]
    y = Atrue_k.astype(np.float64) - pi[None, :]
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    return float((x * y).sum() / (nx * ny + 1e-12)), float(nx / (ny + 1e-12))

# ---------------- 免 torch 解析 .pt (子词对照用) ----------------

_STORAGE_DTYPES = {'FloatStorage': np.float32, 'DoubleStorage': np.float64,
                   'HalfStorage': np.float16, 'LongStorage': np.int64,
                   'IntStorage': np.int32, 'ShortStorage': np.int16,
                   'CharStorage': np.int8, 'ByteStorage': np.uint8,
                   'BoolStorage': np.bool_}

def _rebuild_tensor_v2(storage, offset, size, stride, *a, **kw):
    item = storage.dtype.itemsize
    base = np.asarray(storage)[offset:]
    return np.lib.stride_tricks.as_strided(base, shape=tuple(size),
                                           strides=tuple(s * item for s in stride)).copy()

_rebuild_tensor = _rebuild_tensor_v2

def load_pt(path):
    """不依赖 torch 的 torch.save zip 解析 (仅张量数据)."""
    zf = zipfile.ZipFile(path)
    prefix = [n for n in zf.namelist() if n.endswith('/data.pkl')][0].rsplit('/', 1)[0]
    cache = {}

    def rec(key):
        if key not in cache:
            cache[key] = zf.read(prefix + '/data/' + key)
        return cache[key]

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == 'torch._utils' and name.startswith('_rebuild_tensor'):
                return globals()[name]
            if mod.startswith('torch'):
                return type(name, (object,), {})
            return super().find_class(mod, name)

        def persistent_load(self, pid):
            typ, key, numel = pid[1], pid[2], (pid[4] if len(pid) > 4 else None)
            raw = rec(key)
            dt = _STORAGE_DTYPES.get(getattr(typ, '__name__', ''), None)
            if dt is None:   # UntypedStorage: 用字节数推断
                item = len(raw) // max(int(numel), 1)
                dt = {1: np.uint8, 2: np.float16, 4: np.int32, 8: np.int64}[item]
            return np.frombuffer(raw, dtype=dt)

    with zf.open(prefix + '/data.pkl') as f:
        return U(io.BytesIO(f.read())).load()

# ---------------- 绘图 ----------------

def setup_font():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    return plt
