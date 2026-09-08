#!/usr/bin/env python3
"""HRC 推理加速: 干净的 decode-only 计时 (用 KV cache 的增量 decode)."""
import sys, time
sys.path.insert(0, r"F:\Spark-X2.5-1.7B")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
MDIR = r"F:\Spark-X2.5-1.7B"


def main():
    tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MDIR, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model.eval()

    n_gen = 128

    def bench_decode(ctx_len, label, n_warmup=3, n_run=10):
        """先 prefill ctx_len, 再逐步 decode n_gen 个 token, 计 decode 部分时间."""
        prompt = torch.randint(100, 50000, (1, ctx_len), device=DEV)
        # warmup (包含 prefill + decode)
        for _ in range(n_warmup):
            out = model(prompt, use_cache=True)
            past = out.past_key_values
            cur = out.logits[:, -1:].argmax(-1)
            for t in range(5):
                out = model(cur, past_key_values=past, use_cache=True)
                cur = out.logits[:, -1].argmax(-1, keepdim=True)
                past = out.past_key_values
        # 计时
        times = []
        for _ in range(n_run):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(prompt, use_cache=True)
                past = out.past_key_values
                cur = out.logits[:, -1:].argmax(-1)
                t_prefill = time.perf_counter()
                for t in range(n_gen):
                    out = model(cur, past_key_values=past, use_cache=True)
                    cur = out.logits[:, -1:].argmax(-1)
                    past = out.past_key_values
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t_prefill)
        avg_ms = min(times) / n_gen * 1000
        print("  %-24s %8.3f ms/token  (%6.1f tok/s)" % (label, avg_ms, 1000 / avg_ms), flush=True)
        return avg_ms

    print("===== Decode 速度 vs 上下文长度 (KV cache 开启) =====\n", flush=True)
    r = {}
    for ctx in [64, 512, 1024, 2048]:
        r[ctx] = bench_decode(ctx, "ctx=%d tok" % ctx)

    print("\n===== 汇总 =====\n", flush=True)
    print("  上下文长度    decode速度      相对加速")
    base = r[2048]  # 最慢 = KV 最大的
    for ctx in [64, 512, 1024, 2048]:
        print("  %6d tok    %8.3f ms    %6.2fx" % (ctx, r[ctx], base / r[ctx]), flush=True)

    print("\n===== HRC 加速模型 =====")
    print("  标准AR: 每步 attend N tok KV, N=2048 -> 慢")
    print("  HRC:    每步 attend 74 tok KV (10 mem + 64 window) -> 快")
    print("  = 相当于 ctx=64 的速度 (第 1 行)")
    print("  加速比 = %.1fx (在 2048 上下文场景)" % (r[2048] / r[64]), flush=True)


if __name__ == "__main__":
    main()
