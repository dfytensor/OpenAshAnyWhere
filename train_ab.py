"""统一 A/B 训练入口：同 seed / 同数据顺序 / 同预算（2500 步 × bs32 × L384）。

用法: python train_ab.py --arch v32 --out ab_v32
arch: v30 (pure cummax) | v31 (v3.1 decay) | v32 (v3.2 decay+现代化) | tf (Transformer 基线)
"""
import argparse
import json
import math
import os
import time

import torch

from open_ash_v3 import OpenASH_V3
from baseline_tf import TransformerLM

DEV = 'cuda'
DATA = r'F:\OpenASH2605\minimind_data\pretrain_cached_30000_384.pt'
VOC, HID, HEADS, LAYERS = 23004, 768, 8, 12


def build(arch):
    if arch in ('m2', 'gdn', 'dn'):
        import baseline_fla
        return baseline_fla.FlaLMWrapper(baseline_fla.BUILDERS[arch]())
    if arch == 'v36b1':
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=48, kv_backend='delta_fla', kv_beta_init=4.0)
    if arch == 'v43':
        # 历史对齐注：scale_v43_15k checkpoint 无 kv_out_gate（训练时该机制未生效/未创建），
        # kv_out_gdn 必须为 False——否则探针会走上训练从未见过的 RMS+门路径（loader DRIFT 已证）
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=48, kv_backend='delta_fla', kv_beta_init=4.0,
                          short_conv=True, kv_beta_tok=True, no_pos=True)
    if arch == 'v42':
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=48, kv_backend='delta_fla', kv_beta_init=4.0,
                          short_conv=True, moe_ffn=True, state_split=True)
    if arch == 'v46':
        # v4.6: GDN 核心——键宽突破 c 流上限（96+32=128）、MEM 组永久化、per-token β、RMS 直连
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=96, kv_backend='delta_fla', kv_beta_init=4.0,
                          short_conv=True, gdn_core=True, k_extra=16)
    if arch == 'v51':
        # v5.1: 公平规模裁决版——v50 架构 + tied head + 专家减半（98.7M ≤ GDN 106M）
        return OpenASH_V3(23004, 1024, 8, 6, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=128, kv_backend='delta_fla', kv_beta_init=4.0,
                          short_conv=True, moe_ffn=True, state_split=True, mem_heads=3,
                          mem_force=True, ffn_hidden=1024, tied_head=True)
    if arch == 'v50':
        # v5.0: 架构重构追 GDN——hidden 1024 / d_head 128（GDN 同构）/ kv128 / MEM 3/8 / 6 层肥层
        return OpenASH_V3(23004, 1024, 8, 6, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=128, kv_backend='delta_fla', kv_beta_init=4.0,
                          short_conv=True, moe_ffn=True, state_split=True, mem_heads=3,
                          mem_force=True, ffn_hidden=2048)
    if arch == 'v53l':
        # v5.3L: v53 冠军通路（恒开门 kv_out_gdn + per-token β）回灌 v50 容量骨架
        #       （hidden 1024 / d_head 128 / kv128 / 6 层肥层 FFN2048 / MEM 3/8）——容量与通路合体验证
        return OpenASH_V3(23004, 1024, 8, 6, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=128, kv_backend='delta_fla', kv_beta_init=4.0,
                          short_conv=True, moe_ffn=True, state_split=True, mem_heads=3,
                          mem_force=True, ffn_hidden=2048, kv_out_gdn=True, kv_beta_tok=True)
    if arch == 'v52':
        # v5.2: 恒开门单变量——KV 输出 RMS+per-channel 门（去除 kv_gate=0 阻断）
        #       （v49 冠军基座 6L×FFN2048 + 唯一变量 kv_out_gdn）
        return OpenASH_V3(VOC, HID, 8, 6, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=96, kv_backend='delta_fla', kv_beta_init=4.0,
                          short_conv=True, moe_ffn=True, state_split=True, mem_heads=3,
                          mem_force=True, ffn_hidden=2048, kv_out_gdn=True)
    if arch == 'v53':
        # v5.3: per-token β 直连单变量（sigmoid(w·k̂+b)，恒开下不被阻断）
        return OpenASH_V3(VOC, HID, 8, 6, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=96, kv_backend='delta_fla', kv_beta_init=4.0,
                          short_conv=True, moe_ffn=True, state_split=True, mem_heads=3,
                          mem_force=True, ffn_hidden=2048, kv_out_gdn=True, kv_beta_tok=True)
    if arch == 'v53m':
        # v5.3M: v53 冠军配方 + 专用 MEM 永久组增量隔离——c 流键 g=0/β=1、专用 q_mem、
        #        ones-init 门进 MEM 头槽位；标准通路不动（参数 ~118.7M，+1.3M 如实披露）。
        #        动机：v54 在 gap=128 召回 +0.93（家族最佳）但整体负——隔离长距收益不带 WRITE 宽键
        return OpenASH_V3(VOC, HID, 8, 6, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=96, kv_backend='delta_fla', kv_beta_init=4.0,
                          short_conv=True, moe_ffn=True, state_split=True, mem_heads=3,
                          mem_force=True, ffn_hidden=2048, kv_out_gdn=True, kv_beta_tok=True,
                          mem_extra=True)
    if arch == 'v54':
        # v5.4: 非对称双组 kv192——MEM 3×96 永久组 + WRITE 5×128 学习键组（两调用）
        return OpenASH_V3(VOC, HID, 8, 6, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=96, kv_backend='delta_fla', kv_beta_init=4.0,
                          short_conv=True, moe_ffn=True, state_split=True, mem_heads=3,
                          mem_force=True, ffn_hidden=2048, kv_out_gdn=True, asym_delta=True)
    if arch == 'v49':
        # v4.9: 肥层集中实验——6 层 × 专家 1024（总参数与 v44 持平），对标 GDN 5 层肥层集中
        return OpenASH_V3(VOC, HID, 8, 6, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=96, kv_backend='delta_fla', kv_beta_init=4.0,
                          short_conv=True, moe_ffn=True, state_split=True, mem_heads=3,
                          mem_force=True, ffn_hidden=2048)
    if arch == 'v48':
        # v4.8: MEM 头组（0:3）永久存储强制——g=0 + β=1 零衰减全覆盖（头组分化核心验证）
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=96, kv_backend='delta_fla', kv_beta_init=4.0,
                          short_conv=True, moe_ffn=True, state_split=True, mem_heads=3,
                          mem_force=True)
    if arch == 'v48b':
        # 归档复刻说明：scale_v53onv48_15k 的真实配置——v48 基座 + per-token β
        # （原 v53 计划为 v49 基座+恒开门+β，但当时 kv_out_gdn 穿透缺失且基座为 12L，见 REPORT §14）
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=96, kv_backend='delta_fla', kv_beta_init=4.0,
                          short_conv=True, moe_ffn=True, state_split=True, mem_heads=3,
                          mem_force=True, kv_beta_tok=True)
    if arch == 'v47':
        # v4.7: 键宽隔离实验——v44 冠军配方 + 128 维键（96 c 流 + 32 学习扩展）
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=96, kv_backend='delta_fla', kv_beta_init=4.0,
                          short_conv=True, moe_ffn=True, state_split=True, mem_heads=3,
                          gdn_core=True)
    if arch == 'v45':
        # v4.5: 容量档第二级（MEM 4 头 = 半数头永久记忆；kv_dh 受 c 流宽度约束 ≤96）
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=96, kv_backend='delta_fla', kv_beta_init=4.0,
                          short_conv=True, moe_ffn=True, state_split=True, mem_heads=4)
    if arch == 'v44':
        # v4.4: delta 容量扩充（kv_dh 48→96）+ MEM 组扩容（2→3 头），对标 GDN 纯 delta 容量
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=96, kv_backend='delta_fla', kv_beta_init=4.0,
                          short_conv=True, moe_ffn=True, state_split=True, mem_heads=3,
                          mem_force=True)
    if arch == 'v41':
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=48, kv_backend='delta_fla', kv_beta_init=4.0,
                          short_conv=True, id_weight=True)
    if arch == 'v40':
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=48, kv_backend='delta_fla', kv_beta_init=4.0,
                          short_conv=True, em_weights=True, ffn_hidden=2048)
    if arch == 'v39':
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=48, kv_backend='delta_chunk', kv_beta_init=4.0,
                          short_conv=True)
    if arch == 'v38':
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=48, kv_backend='delta_fla', kv_beta_init=4.0,
                          short_conv=True, kv_beta_tok=True)
    if arch == 'v37':
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=48, kv_backend='delta_fla', kv_beta_init=4.0,
                          short_conv=True)
    if arch == 'v36':
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04,
                          kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                          kv_dim=48, kv_backend='delta_fla')
    if arch == 'v30':
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=512, model_flag='train',
                          use_fused=True, use_decay=False)
    if arch == 'v31':
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=512, model_flag='train',
                          use_fused=True, use_decay=True)
    if arch == 'v32':
        return OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                          use_fused=True, use_decay=True,
                          pre_norm=True, ffn_act='silu',
                          decay_init=(0.0001, 0.35), decay_v_std=0.04)
    if arch == 'v33':
        m = OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                       use_fused=True, use_decay=True,
                       pre_norm=True, ffn_act='silu',
                       decay_init=(0.0001, 0.35), decay_v_std=0.04)
        # 分层多尺度：顶层 6 层改为慢通道（γ ∈ [0.995, 0.9995]，半衰期 138~1386 tok）
        # —— 保留 v3.2 的快通道学习，同时给远程逐字记忆专用的状态载体
        with torch.no_grad():
            for layer in m.decoder_layers[6:]:
                s = torch.linspace(0.0005, 0.005, HID)
                layer.self_attention_linear.decay_b.copy_(torch.log(torch.expm1(s)))
        return m
    if arch == 'v34':
        m = OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                       use_fused=True, use_decay=True,
                       pre_norm=True, ffn_act='silu',
                       decay_init=(0.0001, 0.35), decay_v_std=0.04,
                       kv_channel=True, kv_decay=0.98)
        # 分层多尺度（同 v33）：顶层 6 层慢通道
        with torch.no_grad():
            for layer in m.decoder_layers[6:]:
                s = torch.linspace(0.0005, 0.005, HID)
                layer.self_attention_linear.decay_b.copy_(torch.log(torch.expm1(s)))
        return m
    if arch == 'v35':
        m = OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                       use_fused=True, use_decay=True,
                       pre_norm=True, ffn_act='silu',
                       decay_init=(0.0001, 0.35), decay_v_std=0.04,
                       kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True)
        with torch.no_grad():
            for layer in m.decoder_layers[6:]:
                s = torch.linspace(0.0005, 0.005, HID)
                layer.self_attention_linear.decay_b.copy_(torch.log(torch.expm1(s)))
        return m
    if arch == 'v35s':
        m = OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                       use_fused=True, use_decay=True,
                       pre_norm=True, ffn_act='silu',
                       decay_init=(0.0001, 0.35), decay_v_std=0.04,
                       kv_channel=True, kv_decay=0.98, kv_norm_keys=True, kv_beta=True,
                       kv_dim=48)
        with torch.no_grad():
            for layer in m.decoder_layers[6:]:
                s = torch.linspace(0.0005, 0.005, HID)
                layer.self_attention_linear.decay_b.copy_(torch.log(torch.expm1(s)))
        return m
    if arch == 'v35d':
        m = OpenASH_V3(VOC, HID, HEADS, LAYERS, max_seq_len=2048, model_flag='train',
                       use_fused=True, use_decay=True,
                       pre_norm=True, ffn_act='silu',
                       decay_init=(0.0001, 0.35), decay_v_std=0.04,
                       kv_channel=True, kv_decay=0.98, kv_norm_keys=True,
                       kv_beta=False, kv_delta=True)
        with torch.no_grad():
            for layer in m.decoder_layers[6:]:
                s = torch.linspace(0.0005, 0.005, HID)
                layer.self_attention_linear.decay_b.copy_(torch.log(torch.expm1(s)))
        return m
    if arch == 'tf':
        return TransformerLM(VOC, HID, HEADS, 11, max_seq_len=2048)
    raise ValueError(arch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', required=True)
    ap.add_argument('--steps', type=int, default=2500)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=150)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data', default=r'F:\OpenASH2605\minimind_data\pretrain_cached_30000_384.pt')
    ap.add_argument('--eval_every', type=int, default=2000)
    ap.add_argument('--anchor_lambda', type=float, default=0.0,
                    help='召回锚定：kv_beta_w/o_mix 向初值的相对 L2 惩罚系数（0=关闭）')
    args = ap.parse_args()
    arch = args.arch
    os.makedirs(args.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    L = 384

    torch.manual_seed(2024)
    raw = torch.load(args.data, map_location='cpu', weights_only=True)
    stream = torch.cat([d.long() for d in raw]).cuda()
    n_val = 200_000
    val_stream = stream[-n_val:]
    train_stream = stream[:-n_val]
    val_win = val_stream.unfold(0, L + 1, 1)
    train_win = train_stream.unfold(0, L + 1, 1)
    gen = torch.Generator(device='cpu').manual_seed(23)

    def crop(bs):
        while True:
            i = torch.randint(0, train_win.shape[0], (bs,), generator=gen)
            yield train_win[i].cuda()

    torch.manual_seed(2024)
    model = build(arch).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[{args.arch}] params: {n_params:,}')

    n_win_eval = 256
    @torch.no_grad()
    def evaluate(bs=32):
        model.eval()
        n_win = min(256, val_win.shape[0])
        stride = max(1, val_win.shape[0] // n_win)
        starts = (torch.arange(0, n_win, device=DEV) * stride)
        tot, cnt = 0.0, 0
        for s in range(0, n_win, bs):
            idx = starts[s:s + bs]
            win = val_win[idx]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, _ = model.forward_loss(win[:, :-1], win[:, 1:], fused_ce=True)
            tot += loss.item() * win[:, :-1].numel()
            cnt += win[:, :-1].numel()
        model.train()
        return tot / cnt

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    # v5.3R: 召回锚定——向初始值锚定的绝对 L2 惩罚（仅 kv_beta_w + o_mix）。
    #   依据：移植实验证明 15k→40k 的召回侵蚀由 β 锐化（kv_beta_w 范数翻倍，LM 代价仅 +0.03）
    #   与 o_mix 漂移（MEM 头读出下降，单因子恢复 +0.39）因果驱动；全局 WD 方向对 o_mix 错误
    #   （拉向 0 而非 init 4.0），故用 toward-init 锚定。注意：kv_beta_w 为零初始化，
    #   禁用相对 L2（分母→0 会硬锁死 β），必须用绝对 L2；λ=4e-4 对应无约束漂移总量
    #   （β_w ~6 + o_mix ~120 的 ‖Δ‖² 合计）时惩罚 ≈ 0.05 loss。λ=0 关闭（默认）。
    anchored = []
    if args.anchor_lambda > 0:
        for n_, p_ in model.named_parameters():
            if n_.endswith('kv_beta_w') or n_.endswith('o_mix'):
                anchored.append((n_, p_, p_.detach().clone()))
        print(f'[anchor] recall anchor on {len(anchored)} tensors, lambda={args.anchor_lambda}')

    def anchor_loss():
        if not anchored:
            return torch.zeros((), device=DEV)
        s = torch.zeros((), device=DEV)
        for _, p_, p0 in anchored:
            s = s + (p_.float() - p0.float()).pow(2).sum()
        return args.anchor_lambda * s

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * p)) * 0.9 + args.lr * 0.1

    it = crop(args.batch_size)
    log, t0 = [], time.time()
    model.train()
    for step in range(1, args.steps + 1):
        lr = lr_at(step)
        for gp in opt.param_groups:
            gp['lr'] = lr
        batch = next(it)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = model.forward_loss(batch[:, :-1], batch[:, 1:], fused_ce=True)
        if anchored:
            loss = loss + anchor_loss()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            log.append({'step': step, 'loss': loss.item(), 'lr': lr,
                        'anchor': float(anchor_loss()),
                        'ms_per_step': (time.time() - t0) / step * 1000})
            print(f'[{args.arch}] step {step:5d} | loss {loss.item():.3f} | lr {lr:.2e} | '
                  f'{(time.time() - t0) / step * 1000:.0f} ms/step')
        if step % args.eval_every == 0 or step == args.steps:
            vl = evaluate()
            log[-1]['val_loss'] = vl
            print(f'[{args.arch}]   >> val {vl:.4f} | ppl {math.exp(min(vl, 20)):.2f}')

    torch.save(model.state_dict(), os.path.join(args.out, 'model.pt'))
    json.dump({'arch': args.arch, 'params': n_params, 'log': log},
              open(os.path.join(args.out, 'train_log.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'[{args.arch}] done -> {args.out}/')


if __name__ == '__main__':
    main()
