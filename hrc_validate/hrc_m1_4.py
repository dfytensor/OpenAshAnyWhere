#!/usr/bin/env python3
"""
HRC-LLM v0.17 机制验证 · 阶段1: 纯算法恒等式 (零训练, 合成+真实分数)

验证 HRC 文档声称的 4 条可精确验证机制:
  M1. 已见段最小堆 == 全局 top-B, 零误差, 与到达顺序无关          (v0.5 §1.2)
  M2. 块内 top-K_seg 预筛: 均匀分布无损; 聚集场景 K_seg=4 是拐点  (v0.5 §1.5)
  M3. 禁止块内归一化: 归一化使召回崩塌 (38.9% -> 6.1%)            (v0.5 §1.4)
  M4. 预算语义: top-k 是预算锚点 (decode 零选择)                  (v0.5 §1.9)

分数源两种:
  S-A 合成: 分数带位置趋势 + 块内聚集的"关键段"
  S-B 真实: 用 30M 模型对文本段的 surprise 分数 (见 stage2, 此处先不依赖)
"""
import numpy as np

# ---------------- M1: 流式最小堆 vs 全局 top-B ----------------
def m1_stream_heap(n_seg, B, trend, seed=0):
    rng = np.random.RandomState(seed)
    # 关键段占比 5% (集中在少数块 -> 聚集)
    scores = rng.rand(n_seg) + trend * np.arange(n_seg) / n_seg
    heap = []
    import heapq
    for i, s in enumerate(scores):
        if len(heap) < B:
            heapq.heappush(heap, (s, i))
        elif s > heap[0][0]:
            heapq.heappop(heap)
            heapq.heappush(heap, (s, i))
    heap_sel = sorted(i for _, i in heap)
    global_sel = sorted(np.argsort(scores)[::-1][:B])
    return heap_sel == global_sel, len(set(heap_sel) & set(global_sel))


# ---------------- M2: K_seg 预筛拐点 ----------------
def m2_kseg(k_seg, n_seg=16, n_key_total=8, seed=0, clustered=True):
    """每块 n_seg 段; 关键段总 n_key_total 个. 返回预筛后关键段保留比例."""
    rng = np.random.RandomState(seed)
    n_block = 16
    key_blocks = [0, 1, 2, 3] if clustered else list(range(16))
    key_pos = {}
    for b in key_blocks:
        k = rng.randint(0, n_seg, size=1)
        key_pos[(b, int(k[0]))] = True
    # 只有 n_key_total 个关键段
    total_key = min(n_key_total, len(key_pos))
    # 预筛: 每块保留分数最高 k_seg 个段; 关键段分数设为次高 (难识别)
    kept = 0
    for b in range(n_block):
        scores = rng.rand(n_seg)
        for k in key_pos:
            if k[0] == b:
                scores[k[1]] = 0.9  # 关键段高但不最高 -> 测试 k_seg 是否覆盖
                top = np.argsort(scores)[::-1][:k_seg]
                if k[1] in top:
                    kept += 1
    return kept / total_key


# ---------------- M3: 块内归一化危害 ----------------
def m3_normalization(n_block=64, n_seg=16, seed=0):
    """5% 预算. 关键段全局高分且聚集在少数块. 归一化使每块都出顶分 -> 稀释."""
    N = n_block * n_seg
    B = max(int(N * 0.1), 1)           # 10% 预算(需覆盖5%关键段)
    rng = np.random.RandomState(seed)
    all_s = rng.rand(N) * 0.5          # 普通段分数 0~0.5 (绝对低于关键段)
    # 关键段: 5% 数量, 分数 1.0 (绝对最高, 聚集在 20% 块)
    n_key = int(N * 0.05)
    key_blocks = rng.choice(n_block, int(n_block * 0.2), replace=False)
    key = [int(b) * n_seg + int(j) for b in key_blocks for j in rng.choice(n_seg, 1)]
    key = key[:n_key]
    for k in key:
        all_s[k] = 1.0
    def recall(sel):
        return len(set(sel) & set(key)) / len(key)
    g_sel = np.argsort(all_s)[::-1][:B]
    nrm = all_s.reshape(n_block, n_seg)
    nrm = (nrm - nrm.min(1, keepdims=True)) / (nrm.max(1, keepdims=True) - nrm.min(1, keepdims=True) + 1e-9)
    n_sel = np.argsort(nrm.ravel())[::-1][:B]
    return recall(g_sel), recall(n_sel)


# ---------------- M4: 概率累积 top-p 不是预算锚点 ----------------
def m4_topp(n_seg=4096, B=204):
    """top-p 的动态 k 不能当固定预算; top-k 才行 (文档 §1.9)."""
    rng = np.random.RandomState(0)
    s = np.exp(rng.rand(n_seg) * 3)  # 指数分布分数 -> softmax 概率
    p = s / s.sum()
    sorted_p = np.sort(p)[::-1]
    cum = np.cumsum(sorted_p)
    k_p90 = int((cum < 0.9).sum()) + 1
    return abs(k_p90 - B) > B * 0.3  # top-p 的 k 偏离固定 B 超过 30%


if __name__ == "__main__":
    print("===== HRC-LLM 机制验证 · 阶段1: 纯算法 =====", flush=True)
    # M1
    for trend in (0, 0.5, 1.0):
        for B in (64, 204):
            ok, inter = m1_stream_heap(4096, B, trend, seed=1)
            print("M1 堆==全局top-B: trend=%.1f B=%d -> %s (重合 %d/%d)"
                  % (trend, B, "通过" if ok else "失败", inter, B), flush=True)
    # M2 均匀 vs 聚集
    for clustered in (False, True):
        row = []
        for k in (1, 2, 3, 4, 6):
            row.append("k=%d:%.2f" % (k, m2_kseg(k, clustered=clustered)))
        print("M2 K_seg 关键段保留率 (%s): %s" % ("聚集" if clustered else "均匀", "  ".join(row)),
              flush=True)
    # M3
    r_g, r_n = m3_normalization()
    print("M3 关键段召回: 全局可比=%.3f  块内归一化=%.3f  -> %s" %
          (r_g, r_n, "归一化危害证实" if r_n < r_g * 0.5 else "未复现"), flush=True)
    # M4
    print("M4 top-p k 偏离固定预算>30%%: %s" % ("证实" if m4_topp() else "未复现"), flush=True)
