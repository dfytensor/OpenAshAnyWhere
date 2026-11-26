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

        # 合并线性层，减少参数数量[1](@ref4
        self.combined = nn.Linear(dim_size, 4 * dim_size, bias=False)
        self.alpha1 = torch.nn.Parameter(torch.tensor(0.5))
        self.alpha2 = torch.nn.Parameter(torch.tensor(0.5))
        self.alpha3 = torch.nn.Parameter(torch.tensor(0.5))

        self.head_linear = torch.nn.Linear(heads * 5, heads, bias=False)
        # self.dim_to_linear = torch.nn.Linear(5, 1, bias=False)

        # 使用1x1卷积替代复杂的alpha参数交互[4](@ref)

    def forward(self, x, state=None):
        b, s, d = x.shape
        # 合并线性变换并分割为4个部分[1](@ref)
        combined = self.combined(x).view(b, s, 4, self.heads, -1)
        out, out1, out2, out3 = combined.unbind(2)

        # 调整维度: [b, s, heads, d_head] -> [b,d_head,s, heads]
        out = out.permute(0, 3, 1, 2)
        out1 = out1.permute(0, 3, 1, 2)
        out2 = out2.permute(0, 3, 1, 2)
        out3 = out3.permute(0, 3, 1, 2)

        # 使用累积最大值操作替代softmax[1](@ref)
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
        combined = self.head_linear(combined)*e
        # combined = self.dim_to_linear(
        #     combined.view([combined.shape[0], combined.shape[1], combined.shape[2], -1, 5])).squeeze(-1)
        term1 = a * b
        term2 = self.alpha1 * b + self.alpha2 * d
        term3 = a * (self.alpha3 * e + d)
        term4 = b * (c + e)
        return term1 + term2 + term3 + term4 + c * e+combined


class FeedForward(torch.nn.Module):
    def __init__(self, hidden_size):
        super(FeedForward, self).__init__()
        self.ffn1 = torch.nn.Linear(hidden_size, hidden_size)
        self.ffn2 = torch.nn.Linear(hidden_size, hidden_size)
        self.gate = torch.nn.Linear(hidden_size, hidden_size)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x1 = self.ffn1(x)
        x2 = self.relu(self.gate(x))
        xx = x1 * x2
        x = self.ffn2(xx)
        return x


class DecoderLayer(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, model_flag):
        super(DecoderLayer, self).__init__()
        # self.self_attention = MaxStateSuper(hidden_size, num_heads, model_flag)
        self.self_attention_linear = MaxStateSuper(hidden_size, num_heads, model_flag)
        # self.self_attention = MaxState(hidden_size, num_heads)
        self.ffn = FeedForward(hidden_size)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)

        self.alpha = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x, state=None, ):
        # x1, state = self.self_attention(x, state)
        # x = self.layer_norm(self.alpha * self.ffn(x1) + (1 - self.alpha) * x)
        x1, state = self.self_attention_linear(x, state)
        x = self.layer_norm(self.alpha * self.ffn(x1) + (1 - self.alpha) * x)

        return x, state


class OpenASH(torch.nn.Module):
    def __init__(self, voc_size, hidden_size, num_heads, num_layers, model_flag="train"):
        super(OpenASH, self).__init__()
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

    # 定义超参数
    voc_size = 12506
    num_layers = 8
    hidden_size = 2 ** 6 * num_layers
    num_heads = num_layers
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 1000

    # 初始化模型
    model = OpenASH(voc_size=voc_size, hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers)
    params = 0
    # [i.shape[0]  and len(i.shape) == 1 elif i.shape[1] * i.shape[0]
    for i in model.parameters():
        if i.shape != torch.Size([]):
            params += i.numel()
    print(params)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=3)  # 忽略填充标记的损失计算
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 模拟一些训练数据（实际应用中应该使用真实的数据集）

    # 训练循环
    start_time = time.time()
    for epoch in range(num_epochs):
        data = torch.randint(low=0, high=voc_size, size=(batch_size, 50))  # 输入序列长度为50
        input_tensor = data[:, :-1]
        target_tensor = data[:, 1:]

        # 前向传播
        output, _ = model(input_tensor)

        # 将输出reshape以适应 CrossEntropyLoss 的输入要求
        output = output.reshape(-1, voc_size)
        target_tensor = target_tensor.reshape(-1)

        # 计算损失
        loss = criterion(output, target_tensor)
        # output_mean = (torch.nn.functional.softmax(output, -1)-1).mean()**2
        # c = loss.item() / 50
        # loss = loss - output_mean
        # loss = los
        optimizer.zero_grad()  # 清除梯度

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}--')

    print("Training complete.{}".format(time.time() - start_time))
