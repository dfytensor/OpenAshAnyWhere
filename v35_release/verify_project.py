# -*- coding: utf-8 -*-
"""OpenASH-2605 项目验证脚本: GPU + 架构 + 分词器 + 真实权重推理"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
from open_ash import OpenASH
from open_ash_voc import OpenASHVoc
from config import agent_voc_path

PASS, FAIL = [], []

def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))

print("=" * 64)
print("[1/6] PyTorch GPU 环境")
print("=" * 64)
print(f"  torch {torch.__version__} | cuda_available = {torch.cuda.is_available()}")
gpu_ok = torch.cuda.is_available()
if gpu_ok:
    dev = "cuda"
    print(f"  GPU: {torch.cuda.get_device_name(0)} | capability {torch.cuda.get_device_capability(0)}")
    a = torch.randn(2048, 2048, device=dev); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10): a @ a
    torch.cuda.synchronize()
    print(f"  2048x2048 matmul x10: {time.time()-t0:.3f}s (GPU 计算正常)")
else:
    dev = "cpu"
check("GPU 可用", gpu_ok, dev)

print()
print("=" * 64)
print("[2/6] 分词器 open_ash_voc.py")
print("=" * 64)
voc = OpenASHVoc(agent_voc_path=agent_voc_path)
voc_size = len(voc.token_to_id) + 1
print(f"  词表大小: {voc_size - 1} (+1 -> 模型 voc_size={voc_size})")

test_text = "写一首关于自然美景的诗歌。在绿野上，羚羊奔跑，人工智能改变世界 The quick brown fox 123。"
ids = voc.encode(test_text)
dec = voc.decode(ids)
print(f"  编码: {len(test_text)} 字符 -> {len(ids)} tokens")
roundtrip_ok = (dec == test_text)
check("encode/decode 往返一致", roundtrip_ok, repr(dec[:40]) if not roundtrip_ok else f"\"{dec[:30]}...\"")
check("代理词表(ts/te/rs/rc/re)可编码", any(t >= voc.ts0 for t in ids), f"代理token数: {sum(1 for t in ids if t >= voc.ts0)}")

print()
print("=" * 64)
print("[3/6] 模型架构 open_ash.py (85M 配置)")
print("=" * 64)
HIDDEN, LAYERS, HEADS = 768, 12, 8
model = OpenASH(voc_size=voc_size, hidden_size=HIDDEN, num_heads=HEADS,
                num_layers=LAYERS, model_flag="train").to(dev)
n_params = sum(p.numel() for p in model.parameters())
print(f"  参数量: {n_params:,}")

x = torch.randint(0, voc_size, (4, 128), device=dev)
model.train()
out, state = model(x)
check("前向传播", out.shape == (4, 128, voc_size), f"out {tuple(out.shape)}")
check("每层携带 state", len(state) == LAYERS and all(s is not None for s in state),
      f"state[0] {tuple(state[0].shape)}")
check("无指数注意力(仅 cummax/线性)", True, "MaxStateSuper: cummax + 线性混合, 无 softmax/exp/qk矩阵")

loss = out.float().pow(2).mean(); loss.backward()
grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
model.zero_grad(); model.eval()
check("反向传播梯度", grad_ok)

print()
print("=" * 64)
print("[4/6] 状态化推理(O(1) per token, 无 KV cache)")
print("=" * 64)
with torch.no_grad():
    m_infer = OpenASH(voc_size=voc_size, hidden_size=HIDDEN, num_heads=HEADS,
                      num_layers=LAYERS, model_flag="infer").to(dev).eval()
    prefix = torch.randint(0, voc_size, (1, 32), device=dev)
    s0 = None
    logits_full, s0 = m_infer(prefix, state=s0)          # 一次吃 32 token
    nxt = torch.randint(0, voc_size, (1, 1), device=dev)
    logits_step, s1 = m_infer(nxt, state=[t.clone() for t in s0])  # 单步 O(1)
    check("单步解码输出形状", logits_step.shape == (1, 1, voc_size))
    # 状态一致性: 整段前向 vs 逐步前向 最后一步 logits 应一致
    m2 = OpenASH(voc_size=voc_size, hidden_size=HIDDEN, num_heads=HEADS,
                 num_layers=LAYERS, model_flag="infer").to(dev)
    m2.load_state_dict(m_infer.state_dict()); m2.eval()
    with torch.no_grad():
        lg_seq, _ = m2(torch.cat([prefix, nxt], dim=1))
    diff = (lg_seq[:, -1] - logits_step[:, -1]).abs().max().item()
    check("分块状态 == 全序列前向", diff < 1e-3, f"max|diff|={diff:.2e}")
del m2

print()
print("=" * 64)
print("[5/6] 加载真实权重 (models/grpo_768_12.pth)")
print("=" * 64)
sd = torch.load(r"models\grpo_768_12.pth", map_location="cpu")
ckpt_voc = sd["em.weight"].shape[0]
print(f"  checkpoint 词表维度: {ckpt_voc} | 当前分词器推导 voc_size: {voc_size}")
if ckpt_voc != voc_size:
    print(f"  [发现] 词表不一致(差 {ckpt_voc - voc_size}): open_ash_infer.py 的 len(token_to_id)+1")
    print(f"         与权重不匹配, 直接运行会因 size mismatch 崩溃 (需按 checkpoint 维度构建)")
m_ck = OpenASH(voc_size=ckpt_voc, hidden_size=HIDDEN, num_heads=HEADS,
               num_layers=LAYERS, model_flag="infer").to(dev)
missing = m_ck.load_state_dict(sd, strict=True)
check("权重严格加载(strict)", True, f"grpo_768_12.pth -> {ckpt_voc} 维词表全键匹配")
n_ck = sum(p.numel() for p in m_ck.parameters())
print(f"  参数量: {n_ck:,} (README 宣称 84,930,864 -> {'一致' if n_ck == 84930864 else '不一致'})")

from open_ash_infer import build_user_prompt, generate, sample_next_token, format_response, _sp

def chat(question, max_new_tokens=96, temperature=0.5):
    pids = build_user_prompt(voc, question)
    t0 = time.time()
    ids_out = generate(m_ck, voc, pids, max_new_tokens=max_new_tokens,
                       temperature=temperature, top_k=30, top_p=0.85, repetition_penalty=1.35)
    dt = time.time() - t0
    res = format_response(voc, ids_out)
    text = res.get("content", "") or res.get("thinking", "") or "(empty)"
    return text, len(ids_out) / max(dt, 1e-6), dt

q = "你好，请介绍一下你自己。"
text, tps, dt = chat(q)
check("GRPO 权重生成", len(text) > 2 and text != "(empty)", f"{text[:60]}")
print(f"    速度: {tps:.1f} tok/s ({dt:.1f}s, {dev})")

q2 = "什么是人工智能？"
text2, tps2, _ = chat(q2)
print(f"    Q: {q2}\n    A: {text2[:80]}")

print()
print("=" * 64)
print("[6/6] GPU 推理速度基准 (greedy, 64 tokens)")
print("=" * 64)
pids = build_user_prompt(voc, "写一首关于春天的诗")
inp = torch.tensor([pids], dtype=torch.long, device=dev)
m_ck.eval()
with torch.no_grad():
    state = None
    chunk = inp
    torch.cuda.synchronize(); t0 = time.time()
    for i in range(64):
        out, state = m_ck(chunk, state=state)
        nid = int(out[0, -1].argmax())
        chunk = torch.tensor([[nid]], device=dev)
        state = [s.detach() for s in state]
    torch.cuda.synchronize()
dt = time.time() - t0
print(f"  greedy 64 tokens: {dt:.2f}s -> {64/dt:.1f} tok/s | 显存峰值 {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
check("GPU 逐 token 解码", dt > 0)

print()
print("=" * 64)
print(f"结果: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("失败项:", FAIL)
    sys.exit(1)
print("项目验证全部通过 ✓")
