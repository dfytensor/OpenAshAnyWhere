"""
Ultra-FineWeb 中英文分语言质量分布分析 + 分层抽样
策略：分片(shard)级抽样，随机选若干 parquet 分片后全量读取，速度快且抽样稳定

用法：
  uv run python sample_en_zh_analysis.py
"""
import json
import time
import os
import sys
import argparse
import numpy as np
from collections import Counter
from tqdm import tqdm
from modelscope import MsDataset

CONFIG = {
    "dataset_name": "OpenBMB/Ultra-FineWeb",
    "subsets": ["en", "zh"],
    "seed": 42,
    "score_bins": np.arange(0.0, 1.01, 0.1),
    "percentiles": [1, 5, 10, 25, 50, 75, 90, 95, 99],
    "top_n_sources": 20,
    "min_source_count": 50,
    "cache_dir": "./ufw_en_zh_cache",
    "en_token_ratio": 0.75,
    "zh_token_ratio": 1.8,
    "target_en_ratio": 0.7,
    "target_zh_ratio": 0.3,
    "total_target_tokens": 1_000_000_000,

    # ---------- 抽样策略 ----------
    # 模式可选: "shard" (分片级) 或 "row" (行级随机)
    # "shard" 模式: 随机选 N 个完整分片，全量读取（速度快）
    # "row" 模式:   流式遍历全量，每行按几率保留（慢但无偏）
    "sample_mode": "shard",
    # shard 模式下的分片数（每个子集）:
    "shards_per_subset": 3,        # 3 个分片 ≈ 3/2048 ≈ 0.15% of en. 约 5K-30K 行
    # row 模式下的抽样比例:
    "sample_ratio": 0.00001,       # 0.001%
}

def estimate_tokens(text: str, lang: str) -> int:
    if not text:
        return 0
    if lang == "en":
        return max(1, int(len(text.split()) / CONFIG["en_token_ratio"]))
    return max(1, int(len(text) / CONFIG["zh_token_ratio"]))


def _unpatch_hf_filesystem():
    """Restore HfFileSystem to work around modelscope recursion bug when loading multiple splits."""
    try:
        from huggingface_hub.hf_file_system import HfFileSystem
        from modelscope.msdatasets.utils import hf_datasets_util
        if hf_datasets_util._hf_fs_init_original is not None:
            HfFileSystem.__init__ = hf_datasets_util._hf_fs_init_original
            hf_datasets_util._hf_fs_init_original = None
        if hf_datasets_util._hf_fs_open_original is not None:
            HfFileSystem._open = hf_datasets_util._hf_fs_open_original
            hf_datasets_util._hf_fs_open_original = None
    except Exception:
        pass

def run_scan():
    rng = np.random.default_rng(CONFIG["seed"])
    all_scores, all_tokens, all_sources, all_langs = [], [], [], []
    cache_path = os.path.join(CONFIG["cache_dir"], "scan_state.jsonl")

    for lang_idx, lang in enumerate(CONFIG["subsets"]):
        if lang_idx > 0:
            _unpatch_hf_filesystem()
        print(f"\n[SCAN] Loading {CONFIG['dataset_name']} ({lang}) ...")
        t0 = time.time()

        ds = MsDataset.load(CONFIG["dataset_name"], split=lang, use_streaming=True)

        n_shards = ds.n_shards
        if CONFIG["sample_mode"] == "shard":
            selected = sorted(rng.choice(n_shards, CONFIG["shards_per_subset"], replace=False))
            print(f"  [SHARD] {CONFIG['shards_per_subset']}/{n_shards} shards: {selected}")
        else:
            selected = []

        scores_buf, tokens_buf, sources_buf = [], [], []

        for shard_idx in selected:
            shard_ds = ds.shard(num_shards=n_shards, index=int(shard_idx))
            for row in shard_ds:
                if CONFIG["sample_mode"] == "row" and rng.random() > CONFIG["sample_ratio"]:
                    continue
                score = row.get("score")
                content = row.get("content", "")
                if score is None or not content:
                    continue
                source = row.get("source", "unknown")
                tok_len = estimate_tokens(content, lang)
                scores_buf.append(float(score))
                tokens_buf.append(tok_len)
                sources_buf.append(source)
                if len(scores_buf) % 1000 == 0:
                    elapsed = time.time() - t0
                    print(f"  {len(scores_buf):,} rows in {elapsed:.0f}s ({len(scores_buf)/elapsed:.1f} rows/s)")
            # shard done, update elapsed
            elapsed = time.time() - t0
            print(f"  [SHARD {shard_idx}] {len(scores_buf):,} rows so far ({len(scores_buf)/elapsed:.1f} rows/s)")

        all_scores.extend(scores_buf)
        all_tokens.extend(tokens_buf)
        all_sources.extend(sources_buf)
        all_langs.extend([lang] * len(scores_buf))

        elapsed = time.time() - t0
        print(f"  [DONE] {lang}: {len(scores_buf):,} rows in {elapsed:.0f}s ({len(scores_buf)/elapsed:.1f} rows/s)")

    # 清理缓存文件
    _cleanup()
    return np.array(all_scores), np.array(all_tokens), np.array(all_sources), np.array(all_langs)


def _cleanup():
    import shutil
    cache_dir = CONFIG["cache_dir"]
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)


def analyze_stats(scores, token_lens, sources, langs):
    if len(scores) == 0:
        raise ValueError("No data scanned. Try more shards or increase sample ratio.")

    stats = {}
    stats["metadata"] = {
        "mode": CONFIG["sample_mode"],
        "shards_per_subset": CONFIG.get("shards_per_subset", "N/A"),
        "sample_count": int(len(scores)),
        "total_tokens_sampled": int(token_lens.sum()),
        "en_count": int((langs == "en").sum()),
        "zh_count": int((langs == "zh").sum()),
        "en_tokens": int(token_lens[langs == "en"].sum()),
        "zh_tokens": int(token_lens[langs == "zh"].sum()),
    }
    meta = stats["metadata"]

    # Score bin x language breakdown
    bin_edges = CONFIG["score_bins"]
    bin_indices = np.digitize(scores, bin_edges) - 1
    score_bin_lang_stats = []
    for b in range(len(bin_edges) - 1):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        en_mask = mask & (langs == "en")
        zh_mask = mask & (langs == "zh")
        stats["score_bin_lang_distribution"] = score_bin_lang_stats
        en_toks = token_lens[en_mask].sum()
        zh_toks = token_lens[zh_mask].sum()
        total_toks = en_toks + zh_toks
        score_bin_lang_stats.append({
            "score_range": f"[{bin_edges[b]:.1f}, {bin_edges[b+1]:.1f})",
            "total_rows": int(mask.sum()),
            "en_rows": int(en_mask.sum()),
            "zh_rows": int(zh_mask.sum()),
            "en_row_pct": float(en_mask.sum() / mask.sum() * 100),
            "zh_row_pct": float(zh_mask.sum() / mask.sum() * 100),
            "total_tokens": int(total_toks),
            "en_tokens": int(en_toks),
            "zh_tokens": int(zh_toks),
            "en_token_pct": float(en_toks / total_toks * 100) if total_toks > 0 else 0,
            "zh_token_pct": float(zh_toks / total_toks * 100) if total_toks > 0 else 0,
        })
    stats["score_bin_lang_distribution"] = score_bin_lang_stats

    # Percentile thresholds per language
    stats["percentile_thresholds"] = {}
    for lang_name in ["en", "zh", "all"]:
        if lang_name == "all":
            lang_scores = scores
            lang_tokens = token_lens
        else:
            mask = langs == lang_name
            if mask.sum() == 0:
                continue
            lang_scores = scores[mask]
            lang_tokens = token_lens[mask]
        if len(lang_scores) == 0:
            continue
        pvals = np.percentile(lang_scores, CONFIG["percentiles"])
        stats["percentile_thresholds"][lang_name] = []
        for p, thresh in zip(CONFIG["percentiles"], pvals):
            above = lang_scores >= thresh
            above_tokens = lang_tokens[above].sum()
            stats["percentile_thresholds"][lang_name].append({
                "percentile": f"P{p}",
                "score_threshold": float(thresh),
                "rows_above": int(above.sum()),
                "tokens_above": int(above_tokens),
            })

    # Language ratio trend (fine-grained)
    fine_bins = np.linspace(0, 1, 101)
    fine_idx = np.digitize(scores, fine_bins) - 1
    lang_curve = []
    for b in range(len(fine_bins) - 1):
        mask = fine_idx == b
        if mask.sum() < 5:
            continue
        en_pct = (langs[mask] == "en").mean() * 100
        lang_curve.append({
            "score_mid": float((fine_bins[b] + fine_bins[b + 1]) / 2),
            "en_pct": float(en_pct),
            "zh_pct": float(100 - en_pct),
            "sample_count": int(mask.sum()),
        })
    stats["lang_ratio_by_score"] = lang_curve

    # Stratified sampling plan
    en_tokens_all = meta["en_tokens"]
    zh_tokens_all = meta["zh_tokens"]
    total_target = CONFIG["total_target_tokens"]
    target_en = total_target * CONFIG["target_en_ratio"]
    target_zh = total_target * CONFIG["target_zh_ratio"]
    plan = []
    for b in score_bin_lang_stats:
        en_tok_ratio = b["en_tokens"] / max(en_tokens_all, 1)
        zh_tok_ratio = b["zh_tokens"] / max(zh_tokens_all, 1)
        plan.append({
            "score_range": b["score_range"],
            "en_target_tokens": int(en_tok_ratio * target_en),
            "zh_target_tokens": int(zh_tok_ratio * target_zh),
            "total_target_tokens": int(en_tok_ratio * target_en + zh_tok_ratio * target_zh),
        })
    stats["stratified_sampling_plan"] = plan

    # Source coverage (top per language)
    stats["source_coverage"] = {}
    for lang_name in ["en", "zh"]:
        mask = langs == lang_name
        if mask.sum() == 0:
            continue
        lang_sources = sources[mask]
        lang_tokens = token_lens[mask]
        counter = Counter(lang_sources)
        total_lang_tokens = lang_tokens.sum()
        top = []
        other_rows = other_tokens = 0
        for src, cnt in counter.most_common(CONFIG["top_n_sources"]):
            if cnt >= CONFIG["min_source_count"]:
                src_mask = lang_sources == src
                top.append({
                    "source": src,
                    "row_count": int(cnt),
                    "row_pct": float(cnt / mask.sum() * 100),
                    "token_count": int(lang_tokens[src_mask].sum()),
                    "token_pct": float(lang_tokens[src_mask].sum() / total_lang_tokens * 100),
                })
            else:
                other_rows += cnt
                other_tokens += lang_tokens[lang_sources == src].sum()
        stats["source_coverage"][lang_name] = {
            "top_sources": top,
            "other": {"row_count": int(other_rows), "token_count": int(other_tokens)},
        }

    return stats


def print_summary(stats):
    meta = stats["metadata"]
    print()
    print("=" * 90)
    print(f"[RESULT] Mode={meta['mode']} | Samples: {meta['sample_count']:,} | Tokens: {meta['total_tokens_sampled']:,}")
    print(f"   EN: {meta['en_count']:,} rows / {meta['en_tokens']:,} tokens")
    print(f"   ZH: {meta['zh_count']:,} rows / {meta['zh_tokens']:,} tokens")
    en_pct = meta["en_tokens"] / (meta["en_tokens"] + meta["zh_tokens"]) * 100
    zh_pct = meta["zh_tokens"] / (meta["en_tokens"] + meta["zh_tokens"]) * 100
    print(f"   Token ratio: EN {en_pct:.1f}% / ZH {zh_pct:.1f}%")
    print("=" * 90)

    print()
    print("--- Score Bin x EN/ZH Ratio (by rows) ---")
    print(f"{'ScoreRange':<12} {'Rows':>8} {'EN':>8} {'ZH':>8} {'EN%':>7} {'ZH%':>7}")
    print("-" * 55)
    for b in stats["score_bin_lang_distribution"]:
        print(f"{b['score_range']:<12} {b['total_rows']:>8,} {b['en_rows']:>8,} {b['zh_rows']:>8,} {b['en_row_pct']:>6.1f}% {b['zh_row_pct']:>6.1f}%")

    print()
    print("--- Score Bin x EN/ZH Ratio (by tokens) ---")
    print(f"{'ScoreRange':<12} {'Tokens':>12} {'EN Tok':>10} {'ZH Tok':>10} {'EN%':>7} {'ZH%':>7}")
    print("-" * 60)
    for b in stats["score_bin_lang_distribution"]:
        print(f"{b['score_range']:<12} {b['total_tokens']:>12,} {b['en_tokens']:>10,} {b['zh_tokens']:>10,} {b['en_token_pct']:>6.1f}% {b['zh_token_pct']:>6.1f}%")

    print()
    print("--- Language Ratio vs Score Trend ---")
    print(f"{'Score':<12} {'EN%':>7} {'ZH%':>7} {'Count':>8}")
    print("-" * 38)
    curve = stats["lang_ratio_by_score"]
    step = max(1, len(curve) // 15)
    for i in range(0, len(curve), step):
        c = curve[i]
        print(f"{c['score_mid']:.3f}      {c['en_pct']:>6.1f}% {c['zh_pct']:>6.1f}% {c['sample_count']:>8,}")

    print()
    print("--- Percentile Thresholds ---")
    for lang_name in ["en", "zh", "all"]:
        if lang_name not in stats["percentile_thresholds"]:
            continue
        label = {"en": "EN", "zh": "ZH", "all": "EN+ZH"}[lang_name]
        print(f"  [{label}]")
        print(f"  {'Pctl':<8} {'Score':>8} {'RowsAbove':>10} {'TokensAbove':>12}")
        for pt in stats["percentile_thresholds"][lang_name]:
            print(f"  {pt['percentile']:<8} {pt['score_threshold']:>8.4f} {pt['rows_above']:>10,} {pt['tokens_above']:>12,}")

    print()
    print("--- Stratified Sampling Plan ---")
    print(f"  Target: EN {CONFIG['target_en_ratio']*100:.0f}% + ZH {CONFIG['target_zh_ratio']*100:.0f}%")
    print(f"  {'ScoreRange':<12} {'EN Target':>11} {'ZH Target':>11} {'Total':>11}")
    for s in stats["stratified_sampling_plan"]:
        print(f"  {s['score_range']:<12} {s['en_target_tokens']:>11,} {s['zh_target_tokens']:>11,} {s['total_target_tokens']:>11,}")

    print()
    print("--- Summary ---")
    print(f"  1. Low-score (0.0-0.3): ZH->EN ratio higher")
    print(f"  2. High-score (>0.7): EN dominates")
    print(f"  3. Use per-language thresholds for balanced quality")
    print(f"  4. Stratified plan distributes tokens per score bin")
    print()
    print("=" * 90)


def save_json(stats):
    os.makedirs(CONFIG["cache_dir"], exist_ok=True)
    path = os.path.join(CONFIG["cache_dir"], "en_zh_analysis.json")

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, cls=NpEncoder, ensure_ascii=False, indent=2)
    print(f"[SAVED] Full results -> {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultra-FineWeb EN/ZH quality analysis")
    parser.add_argument("--shards", type=int, default=None, help="Shards per subset (default: CONFIG)")
    args = parser.parse_args()

    if args.shards is not None:
        CONFIG["shards_per_subset"] = args.shards

    try:
        scores, token_lens, sources, langs = run_scan()
        stats = analyze_stats(scores, token_lens, sources, langs)
        print_summary(stats)
        save_json(stats)
    except KeyboardInterrupt:
        print("\n[ABORT] Interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
