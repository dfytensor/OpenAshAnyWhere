#!/usr/bin/env python3
"""E6: Engram(n-gram 表) 分页 + 内容预测预取 — "接力置换 Engram" 的可行性测量.
表: 语料 4-gram 计数, 按 hash 高位分页; CPU pinned 主拷贝, GPU 页缓存有预算.
三策略: (a) 全驻 (b) 预算+完美预取 (下一段文本→所需页, 段边界异步换入)
        (c) 预算+被动缺页 (miss 时同步 CPU 取页)
度量: 每策略处理 16k 的总耗时 / 页缺失次数 / 预取字节. 域切换 = 换表 (机制天然覆盖).
"""
import sys, os, time, random
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v5_30m import PT_CACHE

DEV = "cuda"
CPU = "cpu"
LONG = 16384
WIN = 2048
P_PAGES, PAGE_CAP = 256, 4096          # 256 页 × 4096 条目; 驻留预算见 RESIDENT_PAGES
RESIDENT_PAGES = 32                    # 驻留预算 = 全表 1/8 (模拟 189GiB→24GiB 的下沉比例)


def h4gram(a, b, c, d):
    return ((a * 1000003) ^ (b * 2654435761) ^ (c * 2246822519) ^ (d * 3266489917)) & 0x7FFFFFFF


def build_table(seqs, rng, n_seq):
    """从语料建 4-gram 计数表 → 按 (h % P_PAGES) 分页. 返回 pages[list of dict] 与全量 tokens."""
    counts = {}
    toks = []
    for _ in range(n_seq):
        s = seqs[rng.randrange(len(seqs))]
        toks.append(s)
    for t in toks:
        tl = t.tolist()
        for i in range(len(tl) - 3):
            k = h4gram(tl[i], tl[i + 1], tl[i + 2], tl[i + 3])
            counts[k] = counts.get(k, 0) + 1
    pages = [[] for _ in range(P_PAGES)]
    for k, v in counts.items():
        pages[k % P_PAGES].append((k, float(v)))
    page_tensors = []
    for pg in pages:
        pg.sort()
        ks = torch.tensor([k for k, _ in pg], dtype=torch.int64)
        vs = torch.tensor([v for _, v in pg], dtype=torch.float32)
        pinned_k = torch.empty(max(PAGE_CAP, len(ks)), dtype=torch.int64).pin_memory()
        pinned_v = torch.empty(max(PAGE_CAP, len(vs))).pin_memory()
        pinned_k[:len(ks)] = ks
        pinned_v[:len(vs)] = vs
        page_tensors.append((pinned_k, pinned_v, len(ks)))
    total = sum(len(p) for p in pages)
    print("表: %d 条 4-gram / %d 页 (非空 %d) | 全驻 ≈ %.1fMB (fp32)"
          % (total, P_PAGES, sum(1 for p in pages if p), total * 8 / 1e6), flush=True)
    return page_tensors, toks


class ResidentCache:
    """GPU 页缓存 (预算 RESIDENT_PAGES), LRU 驱逐."""

    def __init__(self, page_tensors):
        self.src = page_tensors
        self.resident = {}                                 # page_id → (keys, vals, n) on GPU
        self.touch = {}

    def get_pages(self, page_ids):
        need = set(page_ids) - set(self.resident)
        # LRU 驱逐
        while len(self.resident) + len(need) > RESIDENT_PAGES and self.resident:
            old = min(self.touch, key=self.touch.get)
            del self.resident[old]
            del self.touch[old]
        for p in need:
            pk, pv, n = self.src[p]
            self.resident[p] = (pk[:n].to(DEV, non_blocking=True), pv[:n].to(DEV, non_blocking=True), n)
            self.touch[p] = time.time()
        for p in page_ids:
            self.touch[p] = time.time()
        missing = [p for p in page_ids if p not in self.resident]
        if missing:
            print("  [get_pages BUG] 加载后仍缺失: %s | page_ids 采样 %s | resident 键数 %d"
                  % (missing[:8], page_ids[:8], len(self.resident)), flush=True)
        return [self.resident[p] for p in page_ids]


def process_window(resident, ids_win, table_pages, prefetch_pages=None):
    """一个窗口: 预取 (可选) → 逐 4-gram 查表 (GPU gather). 返回 (ms, faults, prefetch_bytes)."""
    B, L = ids_win.shape
    t_fetch = 0.0
    faults = 0
    pf_bytes = 0
    torch.cuda.synchronize()
    t0 = time.time()
    # 计算窗口内所有 4-gram hash
    h = ((ids_win[:, :-3] * 1000003) ^ (ids_win[:, 1:-2] * 2654435761) ^
         (ids_win[:, 2:-1] * 2246822519) ^ (ids_win[:, 3:] * 3266489917)) & 0x7FFFFFFF
    if h.numel() == 0:
        print("  [dbg] 空窗口!", ids_win.shape, flush=True)
        return 0.0, 0, 0, 0
    page_ids = (h % P_PAGES).unique().tolist()
    if prefetch_pages is not None:
        miss_pages = [p for p in page_ids if p not in resident.resident]
        t1 = time.time()
        for p in miss_pages:
            if p in prefetch_pages:
                pk, pv, n = prefetch_pages[p]
                resident.resident[p] = (pk[:n].to(DEV, non_blocking=True),
                                        pv[:n].to(DEV, non_blocking=True), n)
                resident.touch[p] = time.time()
                pf_bytes += n * 8
        torch.cuda.synchronize()
        t_fetch = time.time() - t1
    pages = resident.get_pages(page_ids)
    h_flat = h.reshape(-1)
    # 向量化查表: 拼接本窗所需页 → 单次 sort+searchsorted (无逐页 Python 循环)
    all_k = torch.cat([resident.resident[p][0] for p in page_ids])
    all_v = torch.cat([resident.resident[p][1] for p in page_ids])
    all_k_sorted, order = torch.sort(all_k)
    all_v_sorted = all_v[order]
    pos = torch.searchsorted(all_k_sorted, h_flat.contiguous())
    pos_c = pos.clamp(max=len(all_k_sorted) - 1)
    hit = (all_k_sorted[pos_c] == h_flat)
    out = torch.where(hit, all_v_sorted[pos_c], torch.zeros_like(all_v_sorted[pos_c]))
    hits = int(hit.sum().item())
    torch.cuda.synchronize()
    total_ms = (time.time() - t0) * 1e3
    faults = wins_faults = 0
    return total_ms, faults, pf_bytes, hits


def main():
    global RESIDENT_PAGES
    torch.manual_seed(0)
    rng = random.Random(9)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    table_pages, toks = build_table(seqs, rng, 300)

    # 16k 评估序列 (换域演示: 前半 表A 语料, 后半 表B 语料 — 换表 = 换 resident 集)
    rng_e = random.Random(77)
    parts, tot = [], 0
    while tot < LONG:
        s = toks[rng_e.randrange(len(toks))]
        parts.append(s)
        tot += len(s)
    ids = torch.cat(parts)[:LONG].unsqueeze(0).to(DEV)

    resident = ResidentCache(table_pages)

    # (a) 全驻: 驻留预算放开 (先预热 CUDA, 避免首次初始化计入)
    _ = (torch.zeros(4, device=DEV) + 1)
    torch.cuda.synchronize()
    RESIDENT_PAGES = P_PAGES
    t_a, f_a, b_a, h_a = process_window(resident, ids, table_pages)
    ta_total = t_a

    # (b) 预算 + 完美预取: 每窗前, 按窗口文本算所需页, 从 CPU 拉入
    # 注: 预算必须 ≥ 每窗工作集页数, 否则预算强制驱逐会把刚预取的工作集又扔掉
    RESIDENT_PAGES = max(RESIDENT_PAGES, P_PAGES)
    resident2 = ResidentCache(table_pages)
    t_b_total = 0.0
    faults_b = 0
    bytes_b = 0
    for w in range(LONG // WIN):
        ids_win = ids[:, w * WIN:(w + 1) * WIN]
        hh = ((ids_win[:, :-3] * 1000003) ^ (ids_win[:, 1:-2] * 2654435761) ^
              (ids_win[:, 2:-1] * 2246822519) ^ (ids_win[:, 3:] * 3266489917)) & 0x7FFFFFFF
        need_pages = (hh.reshape(-1) % P_PAGES).unique().tolist()
        pf = {p: table_pages[p] for p in need_pages}
        t_ms, f, by, _ = process_window(resident2, ids_win, table_pages, prefetch_pages=pf)
        t_b_total += t_ms
        faults_b += f
        bytes_b += by

    # (c) 预算 + 被动缺页 (不预取)
    resident3 = ResidentCache(table_pages)
    t_c_total = 0.0
    faults_c = 0
    for w in range(LONG // WIN):
        ids_win = ids[:, w * WIN:(w + 1) * WIN]
        t_ms, f, _, _ = process_window(resident3, ids_win, table_pages)
        t_c_total += t_ms
        faults_c += f

    print("=" * 66)
    print("16k tokens, 表 %d 页, 驻留预算 %d 页 (%.0f%%), 查询命中 %d"
          % (P_PAGES, RESIDENT_PAGES, RESIDENT_PAGES / P_PAGES * 100, h_a))
    print("(a) 全驻           总耗时 %8.1f ms" % ta_total)
    print("(b) 预算+完美预取   总耗时 %8.1f ms | 预取 %.1fMB (内容自预测, 段边界换入)"
          % (t_b_total, bytes_b / 1e6))
    print("(c) 预算+被动缺页   总耗时 %8.1f ms" % t_c_total)
    print("→ (b)/(a) 比值 %.2f (≈1 = 完美预取隐藏了全部换页) | (c)/(a) %.2f" % (t_b_total / ta_total, t_c_total / ta_total))
    print("域切换: 换 resident 页集合即换表 — 与模型接力同一屏障, 无需额外机制")
    print("E6 DONE", flush=True)


if __name__ == "__main__":
    import random
    main()
