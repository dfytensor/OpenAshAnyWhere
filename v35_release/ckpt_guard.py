# -*- coding: utf-8 -*-
"""ckpt_guard.py — checkpoint / 数据缓存的完整性门（fail-closed）

背景（2026-11-26 事故）：F: 盘在两天内发生两处静默损坏——数据缓存 zip 中央目录单比特翻转、
full_v53dme_40k 的第 3 层 o_read 突发损坏。PyTorch 的 zip 读取器**不校验 CRC**，
payload 级损坏会被静默加载（坏权重参与评测、坏 token 参与训练）。
本模块在任何 torch.load 之前做全量 CRC 校验，发现坏成员立即抛错。
"""
import zipfile


def assert_zip_integrity(path, label=None):
    """全量 CRC 校验；任一成员损坏则抛 RuntimeError（带成员名）。返回成员数。"""
    z = zipfile.ZipFile(path)
    bad = z.testzip()
    if bad is not None:
        raise RuntimeError(
            '[ckpt_guard] INTEGRITY FAILURE in %s (%s): member %r failed CRC-32. '
            'Refusing to load (PyTorch would silently read the corrupted bytes). '
            'Check disk health and restore from backup.' % (path, label or 'file', bad))
    return len(z.infolist())
