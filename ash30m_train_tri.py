"""ConvLinear-Triton 版 30M 预训练 (原版已有: loss 3.84 / 64s)."""
import sys
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
sys.path.insert(0, r"F:\OpenASH2605")
import os
ash30m = open(r"F:\OpenASH2605\copyfirst_redesign\ash30m_train.py", encoding="utf-8").read()
ash30m = ash30m.replace('torch.save(m0.state_dict(), os.path.join(OUT, "ash30m_orig.pth"))', 'pass')
ash30m = ash30m.replace('torch.save(m1.state_dict(), os.path.join(OUT, "ash30m_conv.pth"))',
                        'torch.save(m1.state_dict(), os.path.join(OUT, "ash30m_conv_tri.pth"))')
i = ash30m.find('    print("=== 原版预训练 ===")')
j = ash30m.find('print("=== ConvLinear版预训练 ===")')
body = ash30m[:i] + '    # 原版跳过 (loss 3.84 / 64s 已有)\n    ' + ash30m[j:]
body = body.replace('train_pretrain(m1, "Conv", steps=3000)', 'train_pretrain(m1, "ConvTri", steps=3000)')
exec(compile(body, "ash30m_train_modified", "exec"))
