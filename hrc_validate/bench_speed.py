#!/usr/bin/env python3
"""HRC 推理加速: 不同上下文长度下的生成速度 (核心 = KV cache 大小影响 decode 速度)."""
import sys, time
sys.path.insert(0, r"F:\Spark-X2.5-1.7B")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
MDIR = r"F:\Spark-X2.5-1.7B"


def bench_gen(model, ctx_len, n_gen=128, n_run=3):
    """生成 n_gen token, 计时."""
    prompt = torch.randint(100, 50000, (1, ctx_len), device=DEV)
    times = []
    for _ in range(n_run + 1):  # +1 warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(prompt, max_new_tokens=n_gen, do_sample=False,
                          use_cache=True, pad_token_id=2)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
    print("加载 Spark 1.7B (28层, 1M ctx)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MDIR, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model.eval()
    n_gen = 128

    print("\n===== 不同上下文长度下的生成速度 =====", flush=True)
    print("  %-12s %10s %10s %10s" % ("上下文长度", "prefill(s)", "生成(s)", "tok/s"), flush=True)

    results = {}
    for ctx_len in [64, 512, 1024, 2048]:
        prompt = torch.randint(100, 50000, (1, ctx_len), device=DEV)
        # prefill 时间
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            model(prompt, use_cache=True)
        torch.cuda.synchronize()
        t_prefill = time.perf_counter() - t0
        # 生成时间
        t_gen = bench_gen(model, ctx_len, n_gen)
        tps = n_gen / t_gen
        results[ctx_len] = (t_prefill, t_gen, tps)
        print("  %-12d %10.3f %10.3f %10.1f" % (ctx_len, t_prefill, t_gen, tps), flush=True)

    print("\n===== 关键对比: 2048 context vs 64 context =====", flush=True)
    t_2048 = results[2048][1]
    t_64 = results[64][1]
    print("  2048 tok KV: %.3fs for %d tok = %.1f tok/s" % (t_2048, n_gen, n_gen/t_2048))
    print("   64 tok KV: %.3fs for %d tok = %.1f tok/s" % (t_64, n_gen, n_gen/t_64))
    print("  加速比 (KV 缩减): %.1fx" % (t_2048 / t_64), flush=True)

    print("\n===== HRC 加速模型 =====", flush=True)
    print("  HRC 原理: encode 一次(1920 tok) + decode 只看 74 tok")
    print("  vs 标准 AR: 每步 attend 全部 N tok")
    print("  当 N=2048: 加速比 = t(2048KV) / t(74KV)")
    print("  当 N 更大: 加速比线性增长 (KV 随 N 线性增)")


if __name__ == "__main__":
    main()
