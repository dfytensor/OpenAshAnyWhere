# 复现包代码同步说明

- 同步时间：2026-11-26；来源提交：0620000（根目录已跟踪的代码文件，逐文件拷贝）
- 背景：此前复现包只同步 REPORT_v35.md，代码副本停在 2026-11-14（open_ash_v3.py 与主代码相差 529 行、
  train_ab.py 相差 235 行）——照旧复现包无法复现报告中任何 DMDA 时代（dm / dme / dms / dmh）的结果。
- 运行约定：训练与探针在 WSL 栈下运行，路径为 /mnt/f/OpenASH2605（勿用 Windows 的 F:\ 路径直接跑）。
- 数据完整性：所有加载点经 ckpt_guard.py 做 CRC 校验（F: 盘 2026-11-26 发生过两处静默损坏）。

## 复现旗舰结果的标准命令（WSL，cd /mnt/f/OpenASH2605）

    # 当前旗舰 dms@80k（@384 8.10，与 GDN@160k 持平）
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True wsl_venv/bin/python -u train_ab.py --arch v53dms         --steps 80000 --warmup 800 --eval_every 8000 --anchor_lambda 3e-6 --seq_len 384 --batch_size 32         --data minimind_data/pretrain_cached_1270238_384.pt --out full_v53dms_80k
    # GDN 同 token 对照（982M tokens）
    wsl_venv/bin/python -u train_gdn160k.py --steps 160000 --warmup 800 --batch_size 16 --out full_gdn_160k
    # 官方四探针（编辑 bench_longrange.py 的 MODEL_KEYS）
    wsl_venv/bin/python -u bench_longrange.py
    # O(1) 解码一致性（改动前 vs 改动后）
    bash run_test_dm_decode3.sh

## 关键 checkpoint 的异地备份
E:\OpenASH_ckpt_backup_20261126\（磁盘 0）：18 个关键 checkpoint + 数据缓存，共 10.3GB，已逐个 CRC 校验。
full_v53dme_40k 已损坏（第 3 层 o_read），隔离于 full_v53dme_40k_CORRUPT_o_read_L3，不可用于评测。

本次同步：187 个文件更新，8 个已是最新。
