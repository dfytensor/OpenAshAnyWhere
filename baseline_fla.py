"""FLA 库基线适配器：Mamba-2 / Gated DeltaNet / Transformer（官方实现，HF CausalLM 接口）。

接口与 train_ab.build() 对齐：forward_loss(ids, targets, fused_ce=...) -> (loss, None)。
参数量通过层数调整对齐 ~96-105M。
"""
import torch
import torch.nn.functional as F

from fla.models import (Mamba2Config, Mamba2ForCausalLM,
                        GatedDeltaNetConfig, GatedDeltaNetForCausalLM,
                        DeltaNetConfig, DeltaNetForCausalLM,
                        TransformerConfig, TransformerForCausalLM)

VOC = 23004


def build_mamba2(layers=18):
    cfg = Mamba2Config(vocab_size=VOC, hidden_size=768, num_hidden_layers=layers,
                       attn_mode='chunk')
    return Mamba2ForCausalLM(cfg)


def build_gdn(layers=5):
    cfg = GatedDeltaNetConfig(vocab_size=VOC, hidden_size=768, num_hidden_layers=layers)
    return GatedDeltaNetForCausalLM(cfg)


def build_deltanet(layers=9):
    cfg = DeltaNetConfig(vocab_size=VOC, hidden_size=768, num_hidden_layers=layers)
    return DeltaNetForCausalLM(cfg)


def build_tf_fla(layers=12):
    cfg = TransformerConfig(vocab_size=VOC, hidden_size=768, num_hidden_layers=layers,
                            num_heads=12)
    return TransformerForCausalLM(cfg)


BUILDERS = {'m2': build_mamba2, 'gdn': build_gdn, 'dn': build_deltanet, 'tf-fla': build_tf_fla}


class FlaLMWrapper(torch.nn.Module):
    """统一接口：forward_loss(ids, targets, fused_ce=...) -> (loss, None)。

    HF CausalLM 惯例：labels 与 input_ids 相同，模型内部自动 shift。
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward_loss(self, ids, targets, fused_ce=True, **_):
        out = self.model(input_ids=ids, labels=targets)
        return out.loss, None

    @torch.no_grad()
    def forward(self, x, state=None, pos_offset=0):
        return self.model(input_ids=x).logits, None


def build_fla(arch):
    key = arch.split('_')[0] if '_' in arch else arch
    num = arch.split('_')[-1] if '_' in arch else ''
    model = BUILDERS[key](int(num) if num.isdigit() else 18)
    return FlaLMWrapper(model)
