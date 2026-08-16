"""OpenASH Triton/CUDAGraph 加速基准测试.

跑法: python bench_triton.py
覆盖: 训练步 / 推理单步 / CUDA Graph 整步 / 真实自回归解码.
"""
import sys, time
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import torch
from open_ash import OpenASH
import ash_triton as AT

DEV = "cuda"


def bench(fn, n=100, warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / n * 1000


def main():
    torch.manual_seed(7)

    # ==== 1) 训练步: 最优配方梯度叠加 ====
    m = OpenASH(voc_size=23005, hidden_size=768, num_heads=8, num_layers=12).to(DEV)
    x = torch.randint(2, 23004, (8, 256), device=DEV)
    y = torch.randint(2, 23004, (8, 256), device=DEV)
    import types
    ce = torch.nn.functional.cross_entropy

    def t0_step():
        out, _ = m(x)
        loss = ce(out.reshape(-1, 23005), y.reshape(-1))
        m_opt.zero_grad(set_to_none=True); loss.backward(); m_opt.step()

    m_opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    t_base = bench(t0_step, n=30)

    # 最优配方: fused AdamW + bf16 + fast forward + compile(max-autotune)
    fast_c = torch.compile(AT.max_state_super_fast, dynamic=False)
    def sa_fast(self, xx, state=None):
        return fast_c(self, xx, state)
    for layer in m.decoder_layers:
        layer.self_attention_linear.forward = types.MethodType(
            sa_fast, layer.self_attention_linear)
    mc = torch.compile(m, dynamic=False, mode="max-autotune")
    m_opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
    def t1_step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = mc(x)
            loss = ce(out.reshape(-1, 23005), y.reshape(-1))
        m_opt.zero_grad(set_to_none=True); loss.backward(); m_opt.step()
    t1_step()  # compile warmup
    t_fast = bench(t1_step, n=30)
    print("=== 训练步 (12层, B=8, S=256) ===")
    print("基线 fp32:                        %.1f ms" % t_base)
    print("bf16+fused+compile(max-autotune): %.1f ms (%.2fx)" % (t_fast, t_base / t_fast))
    del m, mc, m_opt
    torch.cuda.empty_cache()

    # ==== 2) 推理: 单步 + Graph + 解码 ====
    m = OpenASH(voc_size=23005, hidden_size=768, num_heads=8, num_layers=12).to(DEV).eval()
    B, D, L = 4, 768, 12
    H, DH = 8, 96
    for layer in m.decoder_layers:
        layer.self_attention_linear.model_flag = "infer"
    static_tok = torch.randint(2, 23004, (B,), device=DEV)
    static_st = [torch.zeros(B, H, DH, device=DEV) for _ in range(L)]

    def decode_step():
        h = m.em(static_tok)
        for i, layer in enumerate(m.decoder_layers):
            h1, s2 = AT.ash_infer_step(layer.self_attention_linear, h, static_st[i])
            x_ln = layer.layer_norm(layer.alpha * layer.ffn(h1) + (1 - layer.alpha) * h)
            h = x_ln + h
            static_st[i].copy_(s2)
        return m.head_score(h).argmax(-1)

    with torch.no_grad():
        t_eager = bench(decode_step, n=100)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.no_grad():
            for _ in range(3): decode_step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            decode_step()
    with torch.no_grad():
        t_graph = bench(lambda: g.replay(), n=200)
    print("=== 自回归解码 (12层, B=4, emb+head+状态回写) ===")
    print("eager:     %.2f ms/token (%.0f tok/s)" % (t_eager, 1000 / t_eager))
    print("CUDAGraph: %.2f ms/token (%.0f tok/s)  %.1fx" % (
        t_graph, 1000 / t_graph, t_eager / t_graph))


if __name__ == "__main__":
    main()
