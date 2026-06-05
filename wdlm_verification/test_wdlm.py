"""
WDLM 验证测试脚本
验证波动力学语言模型各组件的前向传播是否正常工作
"""

import sys
import os
import torch
import torch.nn.functional as F

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from wdlm import (
    QuantumStateEncoding,
    SchrodingerEvolution,
    WaveInterference,
    WaveResidualBlock,
    WaveMeasurement,
    WaveAttention,
    WaveDynamicsLanguageModel,
    EnhancedWaveDynamicsLM,
    WaveLanguageModelLoss,
    WaveOptimizer,
)


def test_quantum_state_encoding():
    print("=" * 60)
    print("测试 1: QuantumStateEncoding")
    print("=" * 60)
    vocab_size, hidden_dim = 1000, 256
    batch_size, seq_len = 2, 16

    model = QuantumStateEncoding(vocab_size, hidden_dim, n_qubits=8)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(input_ids)

    print(f"  输入形状: {input_ids.shape}")
    print(f"  输出形状: {output.shape}  (期望: [{batch_size}, {seq_len}, {hidden_dim}, 2])")
    assert output.shape == (batch_size, seq_len, hidden_dim, 2), f"形状错误: {output.shape}"
    assert not torch.isnan(output).any(), "输出包含NaN"
    assert not torch.isinf(output).any(), "输出包含Inf"

    # 检查波函数模长
    norm = torch.norm(output, dim=-1)
    print(f"  波函数模长范围: [{norm.min().item():.4f}, {norm.max().item():.4f}]")

    print("  PASSED\n")
    return True


def test_schrodinger_evolution():
    print("=" * 60)
    print("测试 2: SchrodingerEvolution")
    print("=" * 60)
    hidden_dim = 256
    batch_size, seq_len = 2, 16

    model = SchrodingerEvolution(hidden_dim, n_frequencies=8)
    psi = torch.randn(batch_size, seq_len, hidden_dim, 2)
    psi = psi / (torch.norm(psi, dim=-1, keepdim=True) + 1e-8)

    output = model(psi, steps=2)

    print(f"  输入形状: {psi.shape}")
    print(f"  输出形状: {output.shape}  (期望: [{batch_size}, {seq_len}, {hidden_dim}, 2])")
    assert output.shape == (batch_size, seq_len, hidden_dim, 2), f"形状错误: {output.shape}"
    assert not torch.isnan(output).any(), "输出包含NaN"
    assert not torch.isinf(output).any(), "输出包含Inf"

    # 检查归一化
    norm = torch.norm(output, dim=-1)
    print(f"  输出模长范围: [{norm.min().item():.4f}, {norm.max().item():.4f}]")
    print(f"  dt参数值: {model.dt.item():.4f}")

    print("  PASSED\n")
    return True


def test_wave_interference():
    print("=" * 60)
    print("测试 3: WaveInterference")
    print("=" * 60)
    hidden_dim = 256
    batch_size, seq_len = 2, 16
    n_waves = 4

    model = WaveInterference(hidden_dim, n_waves=n_waves)
    psi_list = []
    for _ in range(3):  # 3个演化后的波
        p = torch.randn(batch_size, seq_len, hidden_dim, 2)
        p = p / (torch.norm(p, dim=-1, keepdim=True) + 1e-8)
        psi_list.append(p)

    output = model(psi_list)

    print(f"  输入波数量: {len(psi_list)}")
    print(f"  每个波形状: {psi_list[0].shape}")
    print(f"  输出形状: {output.shape}  (期望: [{batch_size}, {seq_len}, {hidden_dim}, 2])")
    assert output.shape == (batch_size, seq_len, hidden_dim, 2), f"形状错误: {output.shape}"
    assert not torch.isnan(output).any(), "输出包含NaN"
    assert not torch.isinf(output).any(), "输出包含Inf"

    # 干涉后应该不同于任何输入
    similar_to_any = any(torch.allclose(output, p, atol=1e-4) for p in psi_list)
    print(f"  干涉混合: {'是' if not similar_to_any else '否(可能需要检查)'}")

    print("  PASSED\n")
    return True


def test_wave_residual_block():
    print("=" * 60)
    print("测试 4: WaveResidualBlock")
    print("=" * 60)
    hidden_dim = 256
    batch_size, seq_len = 2, 16

    model = WaveResidualBlock(hidden_dim, n_evolutions=3, n_interference_waves=4)
    psi = torch.randn(batch_size, seq_len, hidden_dim, 2)
    psi = psi / (torch.norm(psi, dim=-1, keepdim=True) + 1e-8)

    output = model(psi)

    print(f"  输入形状: {psi.shape}")
    print(f"  输出形状: {output.shape}  (期望: [{batch_size}, {seq_len}, {hidden_dim}, 2])")
    assert output.shape == (batch_size, seq_len, hidden_dim, 2), f"形状错误: {output.shape}"
    assert not torch.isnan(output).any(), "输出包含NaN"
    assert not torch.isinf(output).any(), "输出包含Inf"

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    print("  PASSED\n")
    return True


def test_wave_measurement():
    print("=" * 60)
    print("测试 5: WaveMeasurement")
    print("=" * 60)
    hidden_dim = 256
    vocab_size = 1000
    batch_size, seq_len = 2, 16

    model = WaveMeasurement(hidden_dim, vocab_size)
    psi = torch.randn(batch_size, seq_len, hidden_dim, 2)

    output = model(psi, measurement_type="amplitude")

    print(f"  输入形状: {psi.shape}")
    print(f"  输出形状: {output.shape}  (期望: [{batch_size}, {seq_len}, {vocab_size}])")
    assert output.shape == (batch_size, seq_len, vocab_size), f"形状错误: {output.shape}"
    assert not torch.isnan(output).any(), "输出包含NaN"
    assert not torch.isinf(output).any(), "输出包含Inf"

    # 检查可以计算softmax
    probs = F.softmax(output, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size, seq_len))
    print(f"  Logits范围: [{output.min().item():.4f}, {output.max().item():.4f}]")

    print("  PASSED\n")
    return True


def test_wave_attention():
    print("=" * 60)
    print("测试 6: WaveAttention")
    print("=" * 60)
    hidden_dim = 256
    n_heads = 8
    batch_size, seq_len = 2, 16

    model = WaveAttention(hidden_dim, n_heads=n_heads)
    psi = torch.randn(batch_size, seq_len, hidden_dim, 2)
    psi = psi / (torch.norm(psi, dim=-1, keepdim=True) + 1e-8)

    output = model(psi)

    print(f"  输入形状: {psi.shape}")
    print(f"  n_heads: {n_heads}, head_dim: {hidden_dim // n_heads}")
    print(f"  输出形状: {output.shape}  (期望: [{batch_size}, {seq_len}, {hidden_dim}, 2])")
    assert output.shape == (batch_size, seq_len, hidden_dim, 2), f"形状错误: {output.shape}"
    assert not torch.isnan(output).any(), "输出包含NaN"
    assert not torch.isinf(output).any(), "输出包含Inf"

    # 检查归一化
    norm = torch.norm(output, dim=-1)
    print(f"  输出模长范围: [{norm.min().item():.4f}, {norm.max().item():.4f}]")

    print("  PASSED\n")
    return True


def test_wdlm_basic():
    print("=" * 60)
    print("测试 7: WaveDynamicsLanguageModel (基础版)")
    print("=" * 60)
    vocab_size = 1000
    hidden_dim = 256
    num_layers = 2  # 用2层测试
    batch_size, seq_len = 2, 16

    model = WaveDynamicsLanguageModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        n_qubits=8,
        n_waves=4
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, psi = model(input_ids)

    print(f"  输入形状: {input_ids.shape}")
    print(f"  Logits形状: {logits.shape}  (期望: [{batch_size}, {seq_len}, {vocab_size}])")
    print(f"  Psi形状: {psi.shape}")
    assert logits.shape == (batch_size, seq_len, vocab_size), f"形状错误: {logits.shape}"
    assert psi.shape == (batch_size, seq_len, hidden_dim, 2), f"形状错误: {psi.shape}"
    assert not torch.isnan(logits).any(), "Logits包含NaN"
    assert not torch.isnan(psi).any(), "Psi包含NaN"

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,}")

    # 测试生成
    print(f"\n  测试生成功能...")
    input_ids = torch.randint(0, vocab_size, (1, 4))
    generated = model.generate(input_ids, max_length=10, temperature=1.0, top_k=50)
    print(f"  生成序列长度: {generated.shape[1]}")
    print(f"  生成序列: {generated[0].tolist()}")

    print("  PASSED\n")
    return True


def test_enhanced_wdlm():
    print("=" * 60)
    print("测试 8: EnhancedWaveDynamicsLM (增强版)")
    print("=" * 60)
    vocab_size = 1000
    hidden_dim = 256
    num_layers = 2
    n_heads = 8
    batch_size, seq_len = 2, 16

    model = EnhancedWaveDynamicsLM(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        n_heads=n_heads,
        n_qubits=8
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, psi = model(input_ids)

    print(f"  输入形状: {input_ids.shape}")
    print(f"  Logits形状: {logits.shape}  (期望: [{batch_size}, {seq_len}, {vocab_size}])")
    print(f"  Psi形状: {psi.shape}")
    assert logits.shape == (batch_size, seq_len, vocab_size), f"形状错误: {logits.shape}"
    assert psi.shape == (batch_size, seq_len, hidden_dim, 2), f"形状错误: {psi.shape}"
    assert not torch.isnan(logits).any(), "Logits包含NaN"
    assert not torch.isnan(psi).any(), "Psi包含NaN"

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    print("  PASSED\n")
    return True


def test_loss_and_optimizer():
    print("=" * 60)
    print("测试 9: 损失函数 & 优化器")
    print("=" * 60)
    vocab_size = 1000
    hidden_dim = 128
    num_layers = 1
    batch_size, seq_len = 2, 8

    model = WaveDynamicsLanguageModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    criterion = WaveLanguageModelLoss(alpha=0.1, beta=0.01)
    optimizer = WaveOptimizer(model, lr=1e-4, wave_lr=1e-3)

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 前向传播
    logits, psi = model(input_ids)
    total_loss, ce_loss = criterion(logits, labels, psi)

    print(f"  Logits形状: {logits.shape}")
    print(f"  总损失: {total_loss.item():.4f}")
    print(f"  CE损失: {ce_loss.item():.4f}")
    print(f"  波函数正则化: {(total_loss - ce_loss).item():.6f}")

    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()

    # 检查梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    print(f"  梯度传播: {'正常' if has_grad else '异常！'}")

    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 参数更新
    optimizer.step()
    optimizer.enforce_wave_constraints()

    print(f"  学习率: {optimizer.scheduler.get_last_lr()}")

    print("  PASSED\n")
    return True


def test_gradient_flow():
    """测试梯度是否能够正常流过整个模型"""
    print("=" * 60)
    print("测试 10: 梯度流测试 (端到端)")
    print("=" * 60)
    vocab_size = 500
    hidden_dim = 64
    batch_size, seq_len = 2, 8

    # 基础版
    model_basic = WaveDynamicsLanguageModel(
        vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=1
    )
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits_basic, psi_basic = model_basic(input_ids)
    loss_basic = F.cross_entropy(
        logits_basic.view(-1, vocab_size), labels.view(-1)
    )
    loss_basic.backward()

    grad_norm_basic = sum(
        p.grad.norm().item() for p in model_basic.parameters()
        if p.grad is not None
    )
    print(f"  基础版梯度总范数: {grad_norm_basic:.4f}")

    # 增强版
    model_enhanced = EnhancedWaveDynamicsLM(
        vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=1,
        n_heads=4
    )
    logits_enhanced, psi_enhanced = model_enhanced(input_ids)
    loss_enhanced = F.cross_entropy(
        logits_enhanced.view(-1, vocab_size), labels.view(-1)
    )
    loss_enhanced.backward()

    grad_norm_enhanced = sum(
        p.grad.norm().item() for p in model_enhanced.parameters()
        if p.grad is not None
    )
    print(f"  增强版梯度总范数: {grad_norm_enhanced:.4f}")

    print("  PASSED\n")
    return True


def main():
    print("\n" + "=" * 60)
    print("  波动力学语言模型 (WDLM) 验证测试")
    print("  基于博客: https://blog.csdn.net/weixin_32759777/article/details/161548709")
    print("=" * 60 + "\n")

    tests = [
        ("QuantumStateEncoding (波函数编码)", test_quantum_state_encoding),
        ("SchrodingerEvolution (薛定谔演化)", test_schrodinger_evolution),
        ("WaveInterference (波干涉)", test_wave_interference),
        ("WaveResidualBlock (波函数残差块)", test_wave_residual_block),
        ("WaveMeasurement (波函数测量)", test_wave_measurement),
        ("WaveAttention (波注意力)", test_wave_attention),
        ("WaveDynamicsLanguageModel (基础版WDLM)", test_wdlm_basic),
        ("EnhancedWaveDynamicsLM (增强版WDLM)", test_enhanced_wdlm),
        ("Loss & Optimizer (损失函数与优化器)", test_loss_and_optimizer),
        ("Gradient Flow (梯度流)", test_gradient_flow),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {name}")
            print(f"  错误: {str(e)}")
            import traceback
            traceback.print_exc()
            print()
            failed += 1

    print("=" * 60)
    print(f"  结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
