"""
Ultra-FineWeb / Ultra-FineWeb-L3 / UltraData-SFT-2605 数据抽取脚本
按 score 阈值过滤 + 按总大小 (GB) 限流输出到本地 parquet

数据说明:
  Ultra-FineWeb (L2)  = 过滤后的真实网页, 1T en + 120B zh, 有 score 字段
  Ultra-FineWeb-L3 (L3) = L2 基础上 LLM 合成改写, 4 个 subset:
    - en-QA, en-Multi-Style, zh-QA, zh-Multi-Style
    无 score 字段, 靠随机抽样
  UltraData-SFT-2605 = SFT 指令数据

用法:
  # 从 Ultra-FineWeb 抽 200GB（en+zh, score>=0.5）
  uv run python extract_data.py --dataset ufw --score 0.5 --target-gb 200
  
  # 从 Ultra-FineWeb-L3 抽 100GB（按比例抽所有 4 个 subset）
  uv run python extract_data.py --dataset ufw-l3 --target-gb 100
  
  # 从 UltraData-SFT-2605 抽 500MB
  uv run python extract_data.py --dataset sft --target-gb 0.5
  
  # 全量一键
  uv run python extract_data.py --all
"""
import time, os, sys, argparse
import numpy as np
import pyarrow.parquet as pq

sys.setrecursionlimit(20000)
OUTPUT_DIR = os.path.abspath("./extracted_data")


def _restore_fs():
    try:
        from huggingface_hub.hf_file_system import HfFileSystem
        from modelscope.msdatasets.utils import hf_datasets_util
        for attr in ["_hf_fs_init_original", "_hf_fs_open_original"]:
            if getattr(hf_datasets_util, attr, None) is not None:
                target = HfFileSystem.__init__ if "init" in attr else HfFileSystem._open
                target = getattr(hf_datasets_util, attr)
                setattr(hf_datasets_util, attr, None)
    except Exception:
        pass


def load_ms(name, split=None, subset=None):
    _restore_fs()
    from modelscope import MsDataset
    kwargs = {"use_streaming": True, "split": split or "train"}
    if subset:
        kwargs["subset_name"] = subset
    return MsDataset.load(name, **kwargs)


# ==================== 写入器 ====================
class ParquetWriter:
    """分批写 parquet，每 200MB 切一个新文件."""
    def __init__(self, output_dir, dataset_key, config_name):
        sub = config_name.replace("Ultra-FineWeb-", "").replace("-Synthetic", "") if config_name else "default"
        self.dir = os.path.join(output_dir, dataset_key, sub.replace("/", "_"))
        os.makedirs(self.dir, exist_ok=True)
        self.file_idx = 0
        self.rows = []
        self.total_rows = 0
        self.max_rows_per_file = 200_000

    def write(self, row_dict):
        self.rows.append(row_dict)
        self.total_rows += 1
        if len(self.rows) >= self.max_rows_per_file:
            self._flush()

    def _flush(self):
        if not self.rows:
            return
        import pandas as pd
        path = os.path.join(self.dir, f"part-{self.file_idx:04d}.parquet")
        pd.DataFrame(self.rows).to_parquet(path, index=False)
        self.file_idx += 1
        self.rows = []

    @property
    def total_gb(self):
        import glob
        total = 0
        for f in glob.glob(os.path.join(self.dir, "*.parquet")):
            total += os.path.getsize(f)
        return total / 1_000_000_000

    def close(self):
        self._flush()
        return self.total_gb


# ==================== Ultra-FineWeb (L2) ====================
def extract_ufw(score_threshold, target_gb, seed=42):
    """按 score 过滤 + 分片级抽样"""
    print(f"\n{'='*60}")
    print(f"[L2] Ultra-FineWeb | score>={score_threshold} | target={target_gb}GB")
    print(f"{'='*60}")

    rng = np.random.default_rng(seed)
    target_bytes = target_gb * 1_000_000_000
    written_bytes = 0

    configs = [
        ("en", "en", 2048, 0.50, 600_000),
        ("zh", "zh", 256, 0.40, 500_000),
    ]

    for lang, split, n_shards, gb_per, rows_per in configs:
        writer = ParquetWriter(OUTPUT_DIR, "l2", lang)
        shard_ids = list(range(n_shards))
        rng.shuffle(shard_ids)
        t0 = time.time()

        print(f"\n  [{lang}] {n_shards} shards, ~{gb_per}GB/shard")
        for sidx in shard_ids:
            if written_bytes >= target_bytes:
                break
            ds = load_ms("OpenBMB/Ultra-FineWeb", split=split)
            shard = ds.shard(num_shards=n_shards, index=sidx)
            kept = 0
            for row in shard:
                score = row.get("score")
                content = row.get("content", "")
                if score is None or not content or score < score_threshold:
                    continue
                writer.write({
                    "content": content,
                    "score": float(score),
                    "source": row.get("source", ""),
                    "lang": lang,
                })
                kept += 1
            if kept > 0:
                gb = kept / rows_per * gb_per
                written_bytes += gb * 1_000_000_000
                pct = min(written_bytes / target_bytes * 100, 100)
                print(f"    shard {sidx:4d}: kept {kept:>7,} | ~{gb:.2f}GB | "
                      f"total {written_bytes/1e9:.1f}/{target_gb}GB ({pct:.0f}%) "
                      f"[{time.time()-t0:.0f}s]", flush=True)

        actual_gb = writer.close()
        print(f"  [DONE {lang}] {writer.total_rows:,} rows, {actual_gb:.2f}GB -> {writer.dir}")

    print(f"  [L2 DONE] Total: ~{written_bytes/1e9:.1f}GB")


# ==================== Ultra-FineWeb-L3 ====================
def extract_ufw_l3(target_gb, seed=42):
    """4 个 subset: en/zh x QA/MultiStyle, 按分片数比例分配"""
    print(f"\n{'='*60}")
    print(f"[L3] Ultra-FineWeb-L3 | 4 subsets | target={target_gb}GB")
    print(f"{'='*60}")

    subsets = [
        ("Ultra-FineWeb-L3-en-QA-Synthetic",        616, 0.35, 500_000),
        ("Ultra-FineWeb-L3-en-Multi-Style-Synthetic",552, 0.35, 500_000),
        ("Ultra-FineWeb-L3-zh-QA-Synthetic",         310, 0.30, 400_000),
        ("Ultra-FineWeb-L3-zh-Multi-Style-Synthetic",286, 0.30, 400_000),
    ]
    total_shards = sum(s[1] for s in subsets)
    rng = np.random.default_rng(seed)
    target_bytes = target_gb * 1_000_000_000
    written_bytes = 0

    for name, n_shards, gb_per, rows_per in subsets:
        writer = ParquetWriter(OUTPUT_DIR, "l3", name)
        shard_ids = list(range(n_shards))
        rng.shuffle(shard_ids)
        t0 = time.time()
        subset_gb = target_gb * (n_shards / total_shards)  # 按比例分配
        subset_bytes = subset_gb * 1_000_000_000
        subset_written = 0

        print(f"\n  [{name}] {n_shards} shards, ~{gb_per}GB/shard, alloc={subset_gb:.1f}GB")
        for sidx in shard_ids:
            if subset_written >= subset_bytes:
                break
            ds = load_ms("OpenBMB/Ultra-FineWeb-L3", subset=name)
            shard = ds.shard(num_shards=n_shards, index=sidx)
            kept = 0
            for row in shard:
                content = row.get("content", "")
                if not content:
                    continue
                writer.write({
                    "uid": row.get("uid", ""),
                    "content": content,
                    "style": row.get("style", ""),
                })
                kept += 1
            if kept > 0:
                gb = kept / rows_per * gb_per
                subset_written += gb * 1_000_000_000
                written_bytes += gb * 1_000_000_000
                pct = min(subset_written / subset_bytes * 100, 100)
                print(f"    shard {sidx:4d}: kept {kept:>7,} | ~{gb:.2f}GB | "
                      f"subset {subset_written/1e9:.1f}/{subset_gb:.1f}GB ({pct:.0f}%) "
                      f"[{time.time()-t0:.0f}s]", flush=True)

        actual_gb = writer.close()
        print(f"  [DONE] {writer.total_rows:,} rows, {actual_gb:.2f}GB -> {writer.dir}")

    print(f"  [L3 DONE] Total: ~{written_bytes/1e9:.1f}GB")


# ==================== UltraData-SFT-2605 ====================
def extract_sft(target_gb, seed=42):
    """尝试多种方式加载 SFT 数据集"""
    print(f"\n{'='*60}")
    print(f"[SFT] UltraData-SFT-2605 | target={target_gb}GB")
    print(f"{'='*60}")

    rng = np.random.default_rng(seed)
    writer = ParquetWriter(OUTPUT_DIR, "sft", "default")
    target_bytes = target_gb * 1_000_000_000
    written_bytes = 0

    # 方法1: ModelScope
    token = os.environ.get("MODELSCOPE_TOKEN", "")
    for method_name, load_fn in [
        ("ModelScope (no token)",
         lambda: load_ms("OpenBMB/UltraData-SFT-2605")),
        ("ModelScope (with token)",
         lambda: load_ms("OpenBMB/UltraData-SFT-2605") if token else (_ for _ in ()).throw(Exception("no token"))),
        ("HF mirror",
         lambda: _load_hf("openbmb/UltraData-SFT-2605")),
    ]:
        try:
            print(f"  Trying {method_name}...")
            ds = load_fn()
            print(f"  OK! Features={ds.features}, n_shards={ds.n_shards}")
            t0 = time.time()
            for i, row in enumerate(ds):
                if written_bytes >= target_bytes:
                    break
                content = row.get("content") or row.get("text") or ""
                if not content:
                    continue
                row_size = sum(len(str(v)) for v in row.values()) * 2
                if rng.random() > min(target_bytes / written_bytes, 1.0) if written_bytes > 0 else True:
                    pass
                writer.write({k: str(v) for k, v in row.items()})
                written_bytes += row_size
                if i % 5000 == 0:
                    pct = min(written_bytes / target_bytes * 100, 100)
                    print(f"    {i:,} rows scanned, {written_bytes/1e6:.0f}/{target_gb*1000:.0f}MB "
                          f"({pct:.0f}%) [{time.time()-t0:.0f}s]", flush=True)
            actual_gb = writer.close()
            print(f"  [DONE SFT] {writer.total_rows:,} rows, {actual_gb:.2f}GB -> {writer.dir}")
            return
        except Exception as e:
            print(f"    Failed: {str(e)[:120]}")

    print("  [FAIL] UltraData-SFT-2605 not accessible from this network.")
    print("  Suggestions: set MODELSCOPE_TOKEN or download from HF manually.")


def _load_hf(name):
    import os as _os
    _os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from datasets import load_dataset
    return load_dataset(name, split="train", streaming=True)


# ==================== CLI ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultra-* 数据集抽取工具")
    parser.add_argument("--dataset", choices=["ufw", "ufw-l3", "sft"])
    parser.add_argument("--score", type=float, default=0.5, help="score阈值 (仅L2)")
    parser.add_argument("--target-gb", type=float, default=10, help="目标大小GB")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--all", action="store_true", help="全量: L2 200GB + L3 80GB + SFT 0.5GB")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    import extract_data as _m
    _m.OUTPUT_DIR = out_dir

    if args.all:
        extract_ufw(score_threshold=0.5, target_gb=200, seed=args.seed)
        extract_ufw_l3(target_gb=80, seed=args.seed)
        extract_sft(target_gb=0.5, seed=args.seed)
    elif args.dataset == "ufw":
        extract_ufw(args.score, args.target_gb, args.seed)
    elif args.dataset == "ufw-l3":
        extract_ufw_l3(args.target_gb, args.seed)
    elif args.dataset == "sft":
        extract_sft(args.target_gb, args.seed)
    else:
        parser.print_help()
