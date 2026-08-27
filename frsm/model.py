import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleRecurrentBlock(nn.Module):
    def __init__(self, d_model, expansion_factor=2.0):
        super().__init__()
        hidden_dim = int(d_model * expansion_factor)

        self.W_z = nn.Linear(d_model + d_model, hidden_dim)
        self.W_h = nn.Linear(d_model + d_model, hidden_dim)
        self.W_out = nn.Linear(hidden_dim, d_model)

        self.input_norm = nn.LayerNorm(d_model)
        self.state_norm = nn.LayerNorm(d_model)

    def forward(self, h_prev, x, compute_critical_loss=False):
        h_normed = self.state_norm(h_prev)
        x_normed = self.input_norm(x)
        combined = torch.cat([h_normed, x_normed], dim=-1)

        gate = torch.sigmoid(self.W_z(combined))
        candidate = torch.tanh(self.W_h(combined))

        h_mixed = gate * candidate

        h_new = self.W_out(h_mixed)

        critical_loss = torch.tensor(0.0, device=h_prev.device)
        if compute_critical_loss:
            h_new_norm = torch.norm(h_new, dim=-1, keepdim=True)
            target_norm = torch.ones_like(h_new_norm)
            critical_loss = F.mse_loss(h_new_norm, target_norm)

        return h_new, critical_loss


class FractalRecursiveStateMachine(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_scales: int = 4,
        expansion_factor: float = 2.0,
        spectral_radius_target: float = 0.99,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_scales = num_scales

        self.embed = nn.Embedding(vocab_size, d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        self.input_proj = nn.Linear(d_model, d_model)

        self.scales = nn.ModuleList([
            ScaleRecurrentBlock(d_model, expansion_factor)
            for _ in range(num_scales)
        ])

        self.scale_fusion = nn.Linear(d_model * num_scales, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)

        self.spectral_radius_target = spectral_radius_target
        self.critical_reg_coeff = 0.01

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x, h_prev=None, return_state=False, compute_critical_loss=False):
        batch, seq_len = x.shape

        if h_prev is None:
            h = [torch.zeros(batch, self.d_model, device=x.device)
                 for _ in range(self.num_scales)]
        else:
            h = [h_prev[s].clone() for s in range(self.num_scales)]

        x_emb = self.embed(x)

        outputs = []
        critical_loss_total = torch.tensor(0.0, device=x.device)

        for t in range(seq_len):
            inp = self.input_proj(x_emb[:, t, :])

            next_h = []
            for s in range(self.num_scales):
                update_period = 2 ** s
                if t % update_period == 0:
                    h_s_new, scale_critical_loss = self.scales[s](
                        h[s], inp, compute_critical_loss=compute_critical_loss
                    )
                    next_h.append(h_s_new)
                    critical_loss_total = critical_loss_total + scale_critical_loss
                else:
                    next_h.append(h[s])

            h = next_h

            h_combined = torch.cat(h, dim=-1)
            h_out = self.scale_fusion(h_combined)
            h_out = self.fusion_norm(h_out)

            logits = self.output_proj(h_out)
            outputs.append(logits.unsqueeze(1))

        logits_seq = torch.cat(outputs, dim=1)

        if return_state:
            return logits_seq, h, critical_loss_total
        else:
            return logits_seq

    def generate_step(self, token, h_prev):
        with torch.no_grad():
            x_emb = self.embed(token)
            inp = self.input_proj(x_emb.squeeze(1))

            next_h = []
            for s in range(self.num_scales):
                h_s_new, _ = self.scales[s](h_prev[s], inp, compute_critical_loss=False)
                next_h.append(h_s_new)

            h_combined = torch.cat(next_h, dim=-1)
            h_out = self.scale_fusion(h_combined)
            h_out = self.fusion_norm(h_out)
            logits = self.output_proj(h_out)

            return logits, next_h
