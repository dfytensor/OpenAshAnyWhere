#!/usr/bin/env python3
"""
HRC-Spark: Spark-X2.5-1.7B 拆分为 encode/decode 两段 + LoRA 微调

  encode = layers[0:14] + LoRA-A   → 处理 doc[:1920]，压缩为 memory [k, 2048]
  decode = layers[14:28] + LoRA-B  → 只看 [memory + 最近窗口]，预测 doc[1920:]

加速逻辑: decode 每步只读 memory(k=10) + window(64) = 74 token 的 KV，
而不是 1920+ token 的 KV。上下文越长，加速比越大。
"""
import sys, os, json, time, math
sys.path.insert(0, r"F:\Spark-X2.5-1.7B")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

DEV = "cuda"
D = 2048
SPLIT = 14
N_MEM = 10
TARGET = 128
WINDOW = 64
MDIR = r"F:\Spark-X2.5-1.7B"


class HRCSplit(nn.Module):
    def __init__(self, spark, split=SPLIT, n_mem=N_MEM):
        super().__init__()
        self.spark = spark
        self.split = split
        self.n_mem = n_mem
        self.d = spark.config.hidden_size

        # 冻结原始参数
        for p in self.spark.parameters():
            p.requires_grad_(False)

        # LoRA 分别适配 encode / decode 层
        enc_lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                              target_modules=["q_k_v_proj", "out_proj", "gate_proj", "up_proj"])
        dec_lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                              target_modules=["q_k_v_proj", "out_proj", "gate_proj", "up_proj"])

        for i in range(split):
            self.spark.model.layers[i] = get_peft_model(self.spark.model.layers[i], enc_lcfg)
        for i in range(split, 28):
            self.spark.model.layers[i] = get_peft_model(self.spark.model.layers[i], dec_lcfg)

        # 记忆压缩头 (新参数): 段表征 → memory token
        self.seg_pool = nn.Linear(self.d, self.d, dtype=torch.bfloat16)
        self.mem_proj = nn.Linear(self.d, self.d, dtype=torch.bfloat16)
        self.mem_norm = nn.LayerNorm(self.d, dtype=torch.bfloat16)

    def _run_layers(self, layers, x, seq_len):
        """跑 Spark 层列表，处理 RoPE."""
        head_dim = self.spark.config.head_dim
        rope_cache = {}
        for lt in set(self.spark.config.layer_types):
            rp = self.spark.config.rope_parameters.get(lt, self.spark.config.rope_parameters.get("full_attention", {}))
            theta = rp.get("rope_theta", 10000)
            prf = rp.get("partial_rotary_factor", 1.0)
            dim = int(head_dim * prf)
            inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))
            pos = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            freqs = torch.outer(pos, inv)
            emb = torch.cat([freqs, freqs], dim=-1)
            if prf < 1.0:
                pad = torch.ones(seq_len, head_dim - dim, device=x.device, dtype=torch.float32)
                emb = torch.cat([emb, pad], dim=-1)
            rope_cache[lt] = (emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16))

        for layer in layers:
            lt = getattr(layer, 'layer_type', 'full_attention')
            # LoRA wrapper: 拿底层
            base_layer = layer.base_layer if hasattr(layer, 'base_layer') else layer
            actual_lt = getattr(base_layer, 'layer_type', 'full_attention')
            pe = rope_cache.get(actual_lt, rope_cache.get("full_attention"))
            out = layer(x, position_embeddings=pe)
            x = out[0] if isinstance(out, tuple) else out
        return x

    def encode(self, ctx_ids):
        """编码文档上下文 → memory tokens [b, n_mem, d]."""
        x = self.spark.model.embedding(ctx_ids)
        x = self._run_layers(
            [self.spark.model.layers[i] for i in range(self.split)],
            x, ctx_ids.shape[1])
        # 段池化: 每 32 token 一个段表征
        n_seg = x.shape[1] // 32
        seg_repr = x[:, :n_seg * 32, :].view(x.shape[0], n_seg, 32, self.d).mean(2)  # [b, n_seg, d]
        # 均匀采样 n_mem 个段 + 线性投影 → memory token
        idx = torch.linspace(0, n_seg - 1, self.n_mem, device=seg_repr.device).long()
        mem = self.mem_norm(self.mem_proj(self.seg_pool(seg_repr[:, idx])))  # [b, n_mem, d]
        return mem

    def decode(self, x):
        """解码: 已拼接的输入 → 后14层 → head."""
        x = self._run_layers(
            [self.spark.model.layers[i] for i in range(self.split, 28)],
            x, x.shape[1])
        x = self.spark.model.norm(x)
        return self.spark.lm_head(x)

    def forward(self, ctx_ids, target_ids):
        """训练: 编码 ctx → memory → 解码 target."""
        mem = self.encode(ctx_ids)                           # [b, n_mem, d]
        win = ctx_ids[:, -WINDOW:]                           # 局部窗口
        win_e = self.spark.model.embedding(win)               # [b, W, d]
        tgt_e = self.spark.model.embedding(target_ids[:, :-1])  # [b, T-1, d]
        # decode input = [mem | win | target[:-1]]
        dec_in = torch.cat([mem, win_e, tgt_e], dim=1)       # [b, n_mem + W + T-1, d]
        logits = self.decode(dec_in)                         # [b, n_mem + W + T-1, V]
        # 对齐: 位置 n_mem+W-1 预测 target[0]
        offset = self.n_mem + WINDOW
        pred = logits[:, offset - 1: offset - 1 + TARGET - 1]
        labels = target_ids[:, 1:]
        return pred, labels, mem


@torch.no_grad()
def full_ctx_nll(spark_raw, doc_ids, target_ids):
    """原始 Spark 全上下文 AR NLL (baseline)."""
    ids = torch.cat([doc_ids, target_ids[:, :-1]], dim=1)
    logits = spark_raw(ids).logits[0]
    lg = F.log_softmax(logits[doc_ids.shape[1] - 1: doc_ids.shape[1] - 1 + target_ids.shape[1] - 1].float(), -1)
    t = target_ids[0, 1:]
    nll = -lg.gather(1, t.unsqueeze(1)).squeeze(1)
    return nll.mean().item()


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)

    # 加载两个实例: 一个训练 HRC (带 LoRA), 一个做 baseline (原始)
    spark_train = AutoModelForCausalLM.from_pretrained(
        MDIR, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model = HRCSplit(spark_train, split=SPLIT, n_mem=N_MEM).to(DEV)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print("可训练: %.2fM / 全部: %.2fB (%.2f%%)" %
          (n_train / 1e6, sum(p.numel() for p in model.parameters()) / 1e9,
           n_train / n_all * 100 if (n_all := sum(p.numel() for p in model.parameters())) else 0),
          flush=True)

    # 数据: 500 训练 + 50 验证
    texts = []
    for p in [r"F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl",
              r"F:\OpenASH2605\minimind_data\sft_t2t_mini.jsonl"]:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    t = d.get("text", "")
                    if not t and "conversations" in d:
                        t = "".join(x.get("content", "") for x in d["conversations"])
                    if len(t) > 1500:
                        texts.append(t.replace("\n", ""))
                except Exception:
                    pass
                if len(texts) >= 3000:
                    break
    rng = np.random.RandomState(42)
    rng.shuffle(texts)
    docs = []
    buf, buf_len = [], 0
    for txt in texts:
        ids = tok.encode(txt, add_special_tokens=False)
        buf.append(ids)
        buf_len += len(ids)
        if buf_len >= 1920 + TARGET:
            merged = [t for sub in buf for t in sub][:1920 + TARGET]
            docs.append(torch.tensor([merged], device=DEV))
            buf, buf_len = [], 0
        if len(docs) >= 550:
            break
    print("docs:", len(docs), flush=True)
    train_docs, val_docs = docs[:500], docs[500:550]

    # 训练
    opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
    steps = 1000
    model.train()
    t0 = time.time()
    for st in range(steps):
        doc = train_docs[st % len(train_docs)]
        doc_ids = doc[:, :1920]
        target = doc[:, 1920:1920 + TARGET]
        pred, labels, _ = model(doc_ids, target)
        loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]).float(),
                               labels.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if st % 100 == 0 or st == steps - 1:
            print("  step %d/%d loss=%.4f (%.0fs)" %
                  (st, steps, loss.item(), time.time() - t0), flush=True)
        if (st + 1) % 500 == 0:
            torch.save({"model": model.state_dict(), "step": st + 1},
                       r"F:\夸克\hrc_validate\hrc_spark_ckpt.pt")

    # 评估: HRC-Lite vs 原始 Spark 全上下文
    model.eval()
    print("\n=== 评估 (val %d docs, %d tok 续写) ===" % (len(val_docs), TARGET), flush=True)

    # HRC-Lite
    tot_hrc = 0.0
    with torch.no_grad():
        for doc in val_docs:
            doc_ids = doc[:, :1920]
            target = doc[:, 1920:1920 + TARGET]
            pred, labels, _ = model(doc_ids, target)
            loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]).float(),
                                   labels.reshape(-1))
            tot_hrc += loss.item()
    print("  HRC-Lite (mem+window): %.4f nats/tok" % (tot_hrc / len(val_docs)), flush=True)

    # 原始 Spark 全上下文 (需要 base model 不带 LoRA)
    # 重新加载干净的 base
    del model
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(
        MDIR, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    base.eval()
    tot_ar = 0.0
    with torch.no_grad():
        for doc in val_docs:
            doc_ids = doc[:, :1920]
            target = doc[:, 1920:1920 + TARGET]
            nll = full_ctx_nll(base, doc_ids, target)
            tot_ar += nll
    print("  Spark 全上下文 AR:      %.4f nats/tok" % (tot_ar / len(val_docs)), flush=True)
    del base
    torch.cuda.empty_cache()

    print("\n完成 %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
