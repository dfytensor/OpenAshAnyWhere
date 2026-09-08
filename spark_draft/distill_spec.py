#!/usr/bin/env python3
"""
Spark 草稿模型蒸馏 + 推测解码 benchmark

草稿模型 = em(共享冻结) + 前4层(冻结+LoRA) + head(共享冻结)
蒸馏目标 = 完整 Spark 1.7B 的 next-token 分布 (KL 散度)
加速机制 = 草稿自回归出 γ 个候选 → 完整模型一次验证 → 接受/拒绝
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
MDIR = r"F:\Spark-X2.5-1.7B"
N_DRAFT_LAYERS = 4
GAMMA = 6  # 每轮草稿生成 token 数


class DraftModel(nn.Module):
    """草稿: em(共享) + 前 N 层(冻结+LoRA) + head(共享)."""
    def __init__(self, target, n_layers=N_DRAFT_LAYERS):
        super().__init__()
        self.target_ref = target
        self.embedding = target.model.embedding
        self.layers = torch.nn.ModuleList([target.model.layers[i] for i in range(n_layers)])
        self.head = target.lm_head
        self.norm = target.model.norm
        self.d = target.config.hidden_size
        self.head_dim = target.config.head_dim
        self.layer_types = target.config.layer_types[:n_layers]
        self.rope_parameters = target.config.rope_parameters

        # LoRA 加在前 N 层
        lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                          target_modules=["q_k_v_proj", "out_proj"])
        for i in range(n_layers):
            self.layers[i] = get_peft_model(self.layers[i], lcfg)

        self.trainable = [p for p in self.parameters() if p.requires_grad]

    def _rope(self, seq_len, device):
        """RoPE cos/sin for the dominant layer type (sliding_attention)."""
        lt = "sliding_attention"
        rp = self.rope_parameters.get(lt, self.rope_parameters.get("full_attention",
                                       self.rope_parameters))
        if isinstance(rp, dict) and "rope_theta" in rp:
            theta = rp["rope_theta"]
            prf = rp.get("partial_rotary_factor", 1.0)
        else:
            theta = 10000; prf = 1.0
        dim = int(self.head_dim * prf)
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv)
        emb = torch.cat([freqs, freqs], dim=-1)
        if prf < 1.0:
            pad = torch.ones(seq_len, self.head_dim - dim, device=device, dtype=torch.float32)
            emb = torch.cat([emb, pad], dim=-1)
        return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)

    def forward(self, ids):
        x = self.embedding(ids)
        cos, sin = self._rope(ids.shape[1], ids.device)
        pe = (cos, sin)
        for layer in self.layers:
            out = layer(x, position_embeddings=pe)
            x = out[0] if isinstance(out, tuple) else out
        x = self.norm(x)
        return self.head(x)


def main():
    t00 = time.time()
    tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)

    # ========== ① 加载目标模型 ==========
    print("① 加载完整 Spark 1.7B...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        MDIR, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    # ========== ② 训练前: 测初始接受率 ==========
    print("\n② 测初始接受率 (无训练草稿 = 前4层直接用)...", flush=True)
    # 加载草稿 (LoRA 随机初始化)
    draft = DraftModel(target).to(DEV)

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
                    if len(t) > 500:
                        texts.append(t.replace("\n", ""))
                except Exception:
                    pass
                if len(texts) >= 400:
                    break
    print("texts:", len(texts), flush=True)

    @torch.no_grad()
    def measure_acceptance(dmodel, texts, n_docs=20, seq_len=256):
        """测草稿 next-token argmax 与目标 argmax 的重合率."""
        total_match, total_pos = 0, 0
        for i in range(min(n_docs, len(texts))):
            ids = tok.encode(texts[i], add_special_tokens=False)[:seq_len]
            if len(ids) < 32:
                continue
            ids = [tok.bos_token_id or 0] + ids
            t_ids = torch.tensor([ids], device=DEV)
            # 目标分布
            t_logits = target(t_ids).logits[0]  # [L, V]
            # 草稿分布
            d_logits = dmodel(t_ids)[0]  # [L, V]
            # 比较 argmax
            t_am = t_logits.argmax(-1)
            d_am = d_logits.argmax(-1)
            match = (t_am == d_am).sum().item()
            total_match += match
            total_pos += len(ids) - 1
        return total_match / max(total_pos, 1)

    acc_before = measure_acceptance(draft, texts)
    print("  初始接受率: %.3f" % acc_before, flush=True)

    # ========== ③ KL 蒸馏训练 ==========
    print("\n③ KL 蒸馏训练 (500 步)...", flush=True)
    opt = torch.optim.AdamW(draft.trainable, lr=5e-5, weight_decay=0.01)
    SC = 10  # 每步用 SC=10 个短序列
    SL = 128  # 序列长度

    t0 = time.time()
    draft.train()
    for st in range(500):
        # 取一批文本 (固定长度截断)
        batch_ids = []
        for _ in range(SC):
            txt = texts[np.random.randint(len(texts))]
            ids = tok.encode(txt, add_special_tokens=False)[:SL]
            if len(ids) < SL:
                ids = ids + [0] * (SL - len(ids))
            batch_ids.append([tok.bos_token_id or 0] + ids)
        t_ids = torch.tensor(batch_ids, device=DEV)

        # 目标 logits
        with torch.no_grad():
            t_logits = target(t_ids).logits.float()  # [B, L, V]
            t_prob = F.softmax(t_logits / 1.0, -1)   # T=1 蒸馏

        # 草稿 logits
        d_logits = draft(t_ids).float()            # [B, L, V]
        d_logp = F.log_softmax(d_logits / 1.0, -1)

        # KL(draft || target)
        kl = F.kl_div(
            F.log_softmax(d_logits, -1),
            F.log_softmax(t_logits, -1),
            log_target=True, reduction="batchmean")

        opt.zero_grad(set_to_none=True)
        kl.backward()
        torch.nn.utils.clip_grad_norm_(draft.trainable, 1.0)
        opt.step()
        if st % 100 == 0 or st == 499:
            print("  step %d/%d KL=%.4f (%.0fs)" %
                  (st, 500, kl.item(), time.time() - t0), flush=True)

    draft.eval()
    print("蒸馏完成 %.0fs" % (time.time() - t0), flush=True)

    # ========== ④ 蒸馏后接受率 ==========
    print("\n④ 蒸馏后接受率...", flush=True)
    draft.eval()
    acc_after = measure_acceptance(draft, texts)
    print("  蒸馏后接受率: %.3f (之前 %.3f)" % (acc_after, acc_before), flush=True)

    # ========== ⑤ 推测解码 benchmark ==========
    print("\n⑤ 推测解码 benchmark...", flush=True)
    gamma = GAMMA

    @torch.no_grad()
    def spec_decode(model_t, model_d, prompt_ids, n_tokens=128, gamma=6):
        """标准贪心推测解码 (修正版)."""
        ids = list(prompt_ids)
        n_prompt = len(ids)
        accepted_total = 0
        rounds = 0
        while len(ids) - n_prompt < n_tokens:
            rounds += 1
            n_orig = len(ids)
            remaining = n_tokens - (n_orig - n_prompt)
            gamma_cur = min(gamma, remaining)

            # ① 草稿自回归出 gamma_cur 个候选
            cur = torch.tensor([ids], device=DEV)
            draft_toks = []
            for _ in range(gamma_cur):
                d_logits = model_d(cur)[0, -1]
                nxt = d_logits.argmax(-1).view(1, 1)
                draft_toks.append(nxt.item())
                cur = torch.cat([cur, nxt], dim=1)

            # ② 目标一次验证: 输入 = ids + draft_toks
            verify = torch.tensor([ids + draft_toks], device=DEV)
            t_logits = model_t(verify).logits[0]

            # ③ 接受: t_logits[n_orig-1+j] 预测 draft_toks[j]
            n_accept = 0
            for j in range(gamma_cur):
                if draft_toks[j] == t_logits[n_orig - 1 + j].argmax().item():
                    n_accept += 1
                else:
                    break

            # ④ 拼接
            for j in range(n_accept):
                ids.append(draft_toks[j])
            # bonus: 目标对下一个 token 的预测
            bonus_pos = n_orig - 1 + n_accept
            if bonus_pos < t_logits.shape[0]:
                ids.append(t_logits[bonus_pos].argmax().item())
            accepted_total += n_accept

        return ids[:n_prompt + n_tokens], accepted_total, rounds

    # 计时
    test_text = texts[0]
    prompt_ids = tok.encode(test_text, add_special_tokens=False)[:256]
    n_gen = 128

    # 标准 AR
    t_ar = time.time()
    prompt_t = torch.tensor([prompt_ids], device=DEV)
    with torch.no_grad():
        for _ in range(n_gen):
            logits = target(prompt_t).logits[0, -1]
            nxt = logits.argmax(-1).view(1, 1)
            prompt_t = torch.cat([prompt_t, nxt], dim=1)
    torch.cuda.synchronize()
    t_ar = time.time() - t0
    ar_tps = n_gen / t_ar

    # 推测解码
    t_sp = time.time()
    out_ids, acc, rounds = spec_decode(target, draft, prompt_ids, n_gen)
    torch.cuda.synchronize()
    t_sp = time.time() - t0
    sp_tps = n_gen / t_sp
    avg_accept = acc / rounds if rounds else 0

    print("\n===== 推理速度对比 =====")
    print("  标准AR:   %.2fs, %.1f tok/s" % (t_ar, ar_tps))
    print("  推测解码: %.2fs, %.1f tok/s" % (t_sp, sp_tps))
    print("  加速比: %.2fx" % (ar_tps / sp_tps if sp_tps else 0))
    print("  平均接受长度: %.1f / γ=%d" % (avg_accept, gamma))

    torch.save(draft.state_dict(), r"F:\夸克\spark_draft\draft_model.pt")
    print("\n草稿模型已存. 总用时 %.0fs" % (time.time() - t00))


if __name__ == "__main__":
    main()
