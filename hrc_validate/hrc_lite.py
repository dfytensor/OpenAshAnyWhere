#!/usr/bin/env python3
"""
HRC-Lite: Spark-X2.5-1.7B 三段分割原型

  em (共享, 冻结)
  ├─ 编码器 body[0:14]  + LoRA   → 段表征 → 打分 → top-B% → 记忆桥 (10 soft tokens)
  ├─ 记忆桥: mem_proj (新参数)    → 压缩段表征为解码器软前缀
  ├─ 解码器 body[14:28] + LoRA   → [mem_tokens + 局部窗口] → 预测续写
  └─ head (共享, 冻结, tied)

原参数全部冻结; 可训练 = LoRA(enc+dec) + mem_proj + seg_scorer
任务: doc[:1920] 编码 → 记忆 → 解码 doc[1920:2048]
"""
import sys, os, json, time, math
sys.path.insert(0, r"F:\Spark-X2.5-1.7B")
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

DEV = "cuda"
MDIR = r"F:\Spark-X2.5-1.7B"
SPLIT = 14
S_SEG = 32
TARGET = 128
WINDOW = 64


class HRCLite(nn.Module):
    def __init__(self, spark, split=SPLIT, top_frac=0.15, n_mem=10):
        super().__init__()
        self.spark = spark                       # 全冻结
        self.split = split
        self.top_frac = top_frac
        self.n_mem = n_mem
        self.d = spark.config.hidden_size
        # 冻结
        for p in self.spark.parameters():
            p.requires_grad_(False)
        # LoRA: 编码器 + 解码器 (逐层包装)
        enc_lcfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_k_v_proj", "out_proj"])
        dec_lcfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_k_v_proj", "out_proj"])
        for i in range(split):
            self.spark.model.layers[i] = get_peft_model(
                self.spark.model.layers[i], enc_lcfg)
        for i in range(split, 28):
            self.spark.model.layers[i] = get_peft_model(
                self.spark.model.layers[i], dec_lcfg)
        # 新参数
        self.seg_scorer = nn.Linear(self.d, 1).to(torch.bfloat16)
        self.mem_proj = nn.Sequential(
            nn.Linear(self.d, self.d).to(torch.bfloat16), nn.GELU(),
            nn.Linear(self.d, self.d).to(torch.bfloat16))

    def _rope_cache(self, seq_len, device):
        """按 layer_type 计算 RoPE cos/sin (复刻 model.forward 逻辑)."""
        cache = {}
        head_dim = self.spark.config.head_dim
        for lt in set(self.spark.config.layer_types):
            rp = self.spark.config.rope_parameters[lt]
            theta = rp.get("rope_theta", 10000)
            prf = rp.get("partial_rotary_factor", 1.0)
            dim = int(head_dim * prf)
            inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device,
                                                dtype=torch.float32) / dim))
            pos = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(pos, inv)
            emb = torch.cat([freqs, freqs], dim=-1)
            if prf < 1.0:
                pad = torch.ones(seq_len, head_dim - dim, device=device,
                                 dtype=torch.float32)
                emb = torch.cat([emb, pad], dim=-1)
            cos = emb.cos().to(torch.bfloat16)
            sin = emb.sin().to(torch.bfloat16)
            cache[lt] = (cos, sin)
        return cache

    def _run_layers(self, layers, x, seq_len, attention_mask=None):
        """跑一层列表, 按 layer_type 传 rope."""
        rope = self._rope_cache(seq_len, x.device)
        for layer in layers:
            lt = layer.layer_type if hasattr(layer, 'layer_type') else "full_attention"
            pe = rope.get(lt, rope.get("full_attention"))
            out = layer(x, position_embeddings=pe, attention_mask=attention_mask)
            x = out[0] if isinstance(out, tuple) else out
        return x

    def encode(self, doc_ids):
        x = self.spark.model.embedding(doc_ids)
        layers = [self.spark.model.layers[i] for i in range(self.split)]
        return self._run_layers(layers, x, doc_ids.shape[1])

    def decode(self, x):
        layers = [self.spark.model.layers[i] for i in range(self.split, 28)]
        x = self._run_layers(layers, x, x.shape[1])
        x = self.spark.model.norm(x)
        return self.spark.lm_head(x)

    def forward(self, doc_ids, target_ids):
        """doc_ids: [b, 1920]. target_ids: [b, 128]. 返回 target logits."""
        b = doc_ids.shape[0]
        # encode
        h_enc = self.encode(doc_ids)              # [b, 1920, d]
        n_seg = doc_ids.shape[1] // S_SEG
        seg_repr = h_enc.view(b, n_seg, S_SEG, self.d).mean(2)   # [b, n_seg, d]
        scores = self.seg_scorer(seg_repr).squeeze(-1)           # [b, n_seg]
        k = max(int(n_seg * self.top_frac), 1)
        top_idx = scores.topk(k, dim=1).indices                  # [b, k]
        # gather top segments
        mem = seg_repr.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, self.d))  # [b,k,d]
        mem_soft = self.mem_proj(mem)                            # [b, k, d] 软前缀
        # decode: [mem_soft | window embed | target embed[:-1]]
        win = doc_ids[:, -WINDOW:]                               # 局部窗口
        win_e = self.spark.model.embedding(win)
        tgt_e = self.spark.model.embedding(target_ids[:, :-1])
        dec_in = torch.cat([mem_soft, win_e, tgt_e], dim=1)      # [b, k+W+127, d]
        logits = self.decode(dec_in)                             # [b, k+W+127, V]
        # 对齐: 位置 k+W-1 预测 target[0]; k+W+t-1 预测 target[t]
        offset = self.n_mem + WINDOW
        pred = logits[:, offset - 1:offset - 1 + TARGET - 1]     # [b, 127, V]
        labels = target_ids[:, 1:]                               # [b, 127]
        return pred, labels, scores, mem_soft

    @torch.no_grad()
    def encode_memory(self, doc_ids):
        """推理: 编码 → 返回记忆软前缀."""
        h = self.encode(doc_ids)
        n_seg = doc_ids.shape[1] // S_SEG
        seg_repr = h.view(1, n_seg, S_SEG, self.d).mean(2)
        scores = self.seg_scorer(seg_repr).squeeze(-1)
        k = max(int(n_seg * self.top_frac), 1)
        top = scores.topk(k).indices
        mem = seg_repr[0, top]
        return self.mem_proj(mem).unsqueeze(0)                   # [1, k, d]

    @torch.no_grad()
    def generate(self, mem_soft, window_ids, n_tokens, temperature=0.8):
        """自回归生成."""
        win_e = self.spark.model.embedding(window_ids)
        cur = torch.cat([mem_soft, win_e], dim=1)               # [1, k+W, d]
        out_ids = []
        for _ in range(n_tokens):
            logits = self.decode(cur)
            logits = logits[0, -1].float() / temperature
            probs = F.softmax(logits, -1)
            nxt = torch.multinomial(probs, 1)
            out_ids.append(nxt.item())
            cur = torch.cat([cur, self.spark.model.embedding(nxt.view(1, 1))], dim=1)
        return out_ids
