# 复现包代码同步说明

- 同步时间：2026-11-28；来源提交：333ce02（根目录已跟踪的代码文件，逐文件拷贝）
- 首次同步背景（2026-11-26）：此前复现包只同步 REPORT_v35.md，代码副本停在 2026-11-14（open_ash_v3.py 与主代码相差 529 行、
  train_ab.py 相差 235 行）——照旧复现包无法复现报告中任何 DMDA 时代（dm / dme / dms / dmh）的结果。
- 运行约定：训练与探针在 WSL 栈下运行，路径为 /mnt/f/OpenASH2605（勿用 Windows 的 F:\ 路径直接跑）。
- 数据完整性：所有加载点经 ckpt_guard.py 做 CRC 校验（F: 盘 2026-11-26 发生过两处静默损坏）。
- 评测纪律（2026-11-28）：官方 bench 不得与训练并行（数据流常驻 GPU 约 3.9GB、大批次物化 logits，会把显存顶满）；
  与训练并行时用 light_eval.py（数据留内存、每批 ≤2048 token、进程显存硬上限 10%、结果逐模型落盘，
  设置 CKPT_ROOT=/mnt/e/OpenASH_loom 读取 E: 盘上的 checkpoint）；推理计时与显存测试必须显式 no_grad。
- E: 盘上的 checkpoint（/mnt/e/OpenASH_loom）：dml/dmq 的 40k 与 80k、协议对照 dms_40k_bs16x2（Loom 读端口 A/B，全部负结果）。

## 复现旗舰结果的标准命令（WSL，cd /mnt/f/OpenASH2605）

    # 当前旗舰 dms@80k（@384 8.10，与 GDN@160k 持平）
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True wsl_venv/bin/python -u train_ab.py --arch v53dms         --steps 80000 --warmup 800 --eval_every 8000 --anchor_lambda 3e-6 --seq_len 384 --batch_size 32         --data minimind_data/pretrain_cached_1270238_384.pt --out full_v53dms_80k
    # GDN 同 token 对照（982M tokens）
    wsl_venv/bin/python -u train_gdn160k.py --steps 160000 --warmup 800 --batch_size 16 --out full_gdn_160k
    # 官方四探针（编辑 bench_longrange.py 的 MODEL_KEYS）
    wsl_venv/bin/python -u bench_longrange.py
    # O(1) 解码一致性（改动前 vs 改动后）
    bash run_test_dm_decode3.sh
    bash run_test_loom_decode.sh
    bash run_decode_leak_probe2.sh

## 关键 checkpoint 的异地备份
E:\OpenASH_ckpt_backup_20261126\（磁盘 0）：18 个关键 checkpoint + 数据缓存，共 10.3GB，已逐个 CRC 校验。
full_v53dme_40k 已损坏（第 3 层 o_read），隔离于 full_v53dme_40k_CORRUPT_o_read_L3，不可用于评测。

本次同步：10 个文件更新，221 个已是最新。
