# -*- coding: utf-8 -*-
"""v6 数据: minimind 字符流 (复用 v5 core5 的加载器), 输出随机 crop 的 (B,ctx) 批次"""
import sys, numpy as np, torch
sys.path.insert(0, r'F:\夸克\token_period_package\v5_minimind')
import core5 as C5

V = 1500


class CharStream:
    def __init__(self, ntr=16_000_000, ndev=2_000_000):
        docs, _ = C5.load_docs(target_chars=ntr + ndev + 300_000)
        chars, stoi = C5.build_vocab(docs, V)
        ids, bounds = C5.encode_docs(docs, stoi)
        self.train = torch.from_numpy(ids[:ntr].astype(np.int64))
        self.val = torch.from_numpy(ids[ntr:ntr + ndev].astype(np.int64))

    def crop(self, src, B, ctx, generator=None):
        n = len(src)
        pos = torch.randint(0, n - ctx - 1, (B,), generator=generator)
        out = torch.empty(B, ctx, dtype=torch.long)
        for i, p in enumerate(pos.tolist()):
            out[i] = src[p:p + ctx]
        return out


class SubStream:
    """子词流: 前 95% 训练 / 后 5% 验证"""
    def __init__(self, path=None, V_=2048):
        s = np.load(path or r'F:\OpenASH2605\v6_attbias\sub_stream.npy')
        n = len(s)
        cut = int(n * 0.95)
        self.train = torch.from_numpy(s[:cut].astype(np.int64))
        self.val = torch.from_numpy(s[cut:].astype(np.int64))

    def crop(self, src, B, ctx, generator=None):
        n = len(src)
        pos = torch.randint(0, n - ctx - 1, (B,), generator=generator)
        out = torch.empty(B, ctx, dtype=torch.long)
        for i, p in enumerate(pos.tolist()):
            out[i] = src[p:p + ctx]
        return out
