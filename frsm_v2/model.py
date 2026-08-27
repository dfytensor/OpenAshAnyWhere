"""
FRSM v2: 多层堆叠版本
核心改动: 将单层 4 尺度 → N 层堆叠，每层独立 4 尺度 + 残差连接
目标: 验证深度能否降低 LM loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleRecurrentBlock(nn.Module):
    """单尺度门控递归块 (不变)"""
    def __init__(self, d_model, expansion_factor=2.0):
        super().__init__()
        self.W_forget = nn.Linear(d_model * 2, d_model)
        self.W_input = nn.Linear(d_model * 2, d_model)
        self.W_cand = nn.Linear(d_model * 2, d_model)
        nn.init.constant_(self.W_forget.bias, 1.0)
        nn.init.constant_(self.W_input.bias, -2.0)
        self.norm_h = nn.LayerNorm(d_model)
        self.norm_x = nn.LayerNorm(d_model)

    def forward(self, h_prev, x, compute_critical_loss=False):
        combined = torch.cat([self.norm_h(h_prev), self.norm_x(x)], dim=-1)
        f = torch.sigmoid(self.W_forget(combined))
        i = torch.sigmoid(self.W_input(combined))
        cand = torch.tanh(self.W_cand(combined))
        h_new = f * h_prev + i * cand

        crit_loss = torch.tensor(0.0, device=h_prev.device)
        if compute_critical_loss:
            crit_loss = F.mse_loss(torch.norm(h_new, dim=-1, keepdim=True),
                                   torch.ones_like(h_new[:, :1]))
        return h_new, crit_loss


class FRSM_Layer(nn.Module):
    """单层 FRSM: 4 尺度 + 融合"""
    def __init__(self, d_model, num_scales=4, expansion_factor=2.0):
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        self.input_proj = nn.Linear(d_model, d_model)
        self.scales = nn.ModuleList([
            ScaleRecurrentBlock(d_model, expansion_factor)
            for _ in range(num_scales)
        ])
        self.fusion = nn.Linear(d_model * num_scales, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)

    def forward(self, x_step, h_prev=None, compute_critical_loss=False):
        B, D = x_step.shape
        if h_prev is None:
            h = [torch.zeros(B, D, device=x_step.device) for _ in range(self.num_scales)]
        else:
            h = h_prev

        inp = self.input_proj(x_step)
        next_h = []
        crit_total = torch.tensor(0.0, device=x_step.device)

        for s in range(self.num_scales):
            # 层间步数用全局计数器: 传入 t 参数
            # 简化: 所有尺度始终用同一个 step 序号
            h_s_new, crit_s = self.scales[s](h[s], inp, compute_critical_loss)
            next_h.append(h_s_new)
            crit_total = crit_total + crit_s

        fused = self.fusion_norm(self.fusion(torch.cat(next_h, dim=-1)))
        return fused, next_h, crit_total


class MultiLayerFRSM(nn.Module):
    """多层 FRSM"""
    def __init__(self, vocab_size, d_model=256, num_scales=4, num_layers=3,
                 expansion_factor=2.0, spectral_radius_target=0.99):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_scales = num_scales
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            FRSM_Layer(d_model, num_scales, expansion_factor)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        self.spectral_radius_target = spectral_radius_target
        self.critical_reg_coeff = 0.01

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x, h_prev=None, return_state=False, compute_critical_loss=False):
        batch, seq_len = x.shape

        # 初始化每层的多尺度状态
        if h_prev is None:
            all_states = [
                [torch.zeros(batch, self.d_model, device=x.device)
                 for _ in range(self.num_scales)]
                for _ in range(self.num_layers)
            ]
        else:
            all_states = [[hs.clone() for hs in layer_state] for layer_state in h_prev]

        x_emb = self.embed(x)
        outputs = []
        crit_total = torch.tensor(0.0, device=x.device)

        for t in range(seq_len):
            inp = x_emb[:, t, :]

            for layer_idx in range(self.num_layers):
                # 第 layer_idx 层的尺度用 t 作为 step 序号
                # 多尺度更新: scale s 在 t % (2^s) == 0 时更新
                next_h = []
                for s in range(self.num_scales):
                    if t % (2 ** s) == 0:
                        h_new, crit = self.layers[layer_idx].scales[s](
                            all_states[layer_idx][s], inp, compute_critical_loss
                        )
                        next_h.append(h_new)
                        crit_total = crit_total + crit
                    else:
                        next_h.append(all_states[layer_idx][s])
                all_states[layer_idx] = next_h

                # 融合 → 作为下一层的输入
                fused = self.layers[layer_idx].fusion(
                    torch.cat(all_states[layer_idx], dim=-1)
                )
                inp = self.layers[layer_idx].fusion_norm(fused) + inp  # 残差

            out = self.output_norm(inp)
            logits = self.output_proj(out)
            outputs.append(logits.unsqueeze(1))

        logits_seq = torch.cat(outputs, dim=1)

        if return_state:
            return logits_seq, all_states, crit_total
        return logits_seq

    def generate_step(self, token, h_prev, step_counter):
        """step_counter 用于决定哪些尺度更新"""
        with torch.no_grad():
            x_emb = self.embed(token)
            inp = x_emb.squeeze(1)

            for layer_idx in range(self.num_layers):
                next_h = []
                for s in range(self.num_scales):
                    if step_counter % (2 ** s) == 0:
                        h_new, _ = self.layers[layer_idx].scales[s](
                            h_prev[layer_idx][s], inp, compute_critical_loss=False
                        )
                        next_h.append(h_new)
                    else:
                        next_h.append(h_prev[layer_idx][s])
                h_prev[layer_idx] = next_h
                fused = self.layers[layer_idx].fusion(
                    torch.cat(h_prev[layer_idx], dim=-1)
                )
                inp = self.layers[layer_idx].fusion_norm(fused) + inp

            out = self.output_norm(inp)
            logits = self.output_proj(out)
            return logits, h_prev
