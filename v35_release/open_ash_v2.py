import time

import torch
from torch import nn, optim


class MaxStateSuper(torch.nn.Module):
    def __init__(self, dim_size, heads, model_flag="train"):
        super(MaxStateSuper, self).__init__()
        self.heads = heads
        self.d_head = int(dim_size // heads)
        self.model_flag = model_flag
        assert dim_size % heads == 0, "Dimension size must be divisible by head size."

        self.combined = nn.Linear(dim_size, 4 * dim_size, bias=False)
        self.alpha1 = torch.nn.Parameter(torch.tensor(0.5))
        self.alpha2 = torch.nn.Parameter(torch.tensor(0.5))
        self.alpha3 = torch.nn.Parameter(torch.tensor(0.5))

        self.head_linear = torch.nn.Linear(heads * 5, heads, bias=False)

    def forward(self, x, state=None):
        b, s, d = x.shape
        combined = self.combined(x).view(b, s, 4, self.heads, -1)
        out, out1, out2, out3 = combined.unbind(2)

        out = out.permute(0, 3, 1, 2)
        out1 = out1.permute(0, 3, 1, 2)
        out2 = out2.permute(0, 3, 1, 2)
        out3 = out3.permute(0, 3, 1, 2)

        if state is None:
            out4, _ = torch.cummax(out2, dim=2)
            state = out4[:, :, -1:]
        else:
            out4, _ = torch.cummax(torch.cat([state, out2], dim=2), dim=2)
            if self.model_flag == "train":
                out4 = out4[:, :, 1:]
            else:
                out4 = out4[:, :, -1:]
            state = out4[:, :, -1:]

        out_l = self.gen_model(out, out1, out2, out3, out4)
        out = out_l.transpose(1, 2).contiguous().view(b, s, d)

        return out, state

    def gen_model(self, a, b, c, d, e):
        combined = torch.cat([a, b, c, d, e], dim=-1)
        combined = self.head_linear(combined) * e
        term1 = a * b
        term2 = self.alpha1 * b + self.alpha2 * d
        term3 = a * (self.alpha3 * e + d)
        term4 = b * (c + e)
        return term1 + term2 + term3 + term4 + c * e + combined


class PhaseGateFFN(torch.nn.Module):
    """sin+cos gating replaces ReLU (inspired by WDLM)"""
    def __init__(self, hidden_size):
        super(PhaseGateFFN, self).__init__()
        h = hidden_size
        self.value_proj = nn.Linear(h, h, bias=False)
        self.gate_proj = nn.Linear(h, h, bias=False)
        self.out_proj = nn.Linear(h, h, bias=False)

    def forward(self, x):
        v = self.value_proj(x)
        g = self.gate_proj(x)
        return self.out_proj(v * (torch.sin(g) + torch.cos(g)) * 0.5)


class DecoderLayer(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, model_flag):
        super(DecoderLayer, self).__init__()
        self.self_attention_linear = MaxStateSuper(hidden_size, num_heads, model_flag)
        self.ffn = PhaseGateFFN(hidden_size)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        self.alpha = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x, state=None):
        x1, state = self.self_attention_linear(x, state)
        x = self.layer_norm(self.alpha * self.ffn(x1) + (1 - self.alpha) * x)
        return x, state


class OpenASH_V2(torch.nn.Module):
    def __init__(self, voc_size, hidden_size, num_heads, num_layers, model_flag="train"):
        super(OpenASH_V2, self).__init__()
        self.em = torch.nn.Embedding(voc_size, hidden_size, padding_idx=0)
        self.decoder_layers = torch.nn.ModuleList(
            [DecoderLayer(hidden_size, num_heads, model_flag) for _ in range(num_layers)])
        self.head_score = nn.Linear(hidden_size, voc_size, bias=False)

    def forward(self, x, state=None):
        x = self.em(x)
        if state is None:
            state = [None] * len(self.decoder_layers)
        i = 0
        for ii, decoder_layer in enumerate(self.decoder_layers):
            x1, state[i] = decoder_layer(x, state[i])
            x = x1 + x
            i += 1
        x_out = self.head_score(x)
        return x_out, state


if __name__ == '__main__':
    voc_size = 12506
    num_layers = 8
    hidden_size = 2 ** 6 * num_layers
    num_heads = num_layers
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 1000

    model = OpenASH_V2(voc_size=voc_size, hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers)
    params = 0
    for i in model.parameters():
        if i.shape != torch.Size([]):
            params += i.numel()
    print(f"Params: {params}")

    criterion = nn.CrossEntropyLoss(ignore_index=3)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()
    for epoch in range(num_epochs):
        data = torch.randint(low=0, high=voc_size, size=(batch_size, 50))
        input_tensor = data[:, :-1]
        target_tensor = data[:, 1:]

        output, _ = model(input_tensor)
        output = output.reshape(-1, voc_size)
        target_tensor = target_tensor.reshape(-1)

        loss = criterion(output, target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}--')

    print("Training complete.{}".format(time.time() - start_time))
