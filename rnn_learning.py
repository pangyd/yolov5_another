import torch
import torch.nn as nn
import torch.optim as optim

# 设置随机数种子
torch.manual_seed(0)

# 生成数据
data = torch.tensor([i for i in range(1)], dtype=torch.float32)
target = torch.tensor([i+1 for i in range(64)], dtype=torch.float32)

# 将数据转换为(seq_len, batch_size, input_size)形式
data = data.unsqueeze(0).unsqueeze(0)
target = target.unsqueeze(0).unsqueeze(0)

# input: seq_length, batch_size, input_size
# h0: num_layers, batch_size, hidden_size
# output: seq_length, batch_size, hidden_size
# hn: num_layers, batch_size, hidden_size

# 定义模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[-1, :, :])
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=torch.device('cpu'))

model = RNN(1, 64, 1)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(100):
    hidden = model.init_hidden(1)
    model.zero_grad()
    output, hidden = model(data, hidden)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# 测试模型
with torch.no_grad():
    test_data = torch.tensor([50.], dtype=torch.float32)
    test_data = test_data.unsqueeze(0).unsqueeze(0)
    hidden = model.init_hidden(1)
    # print(test_data.shape)
    # print(hidden.shape)
    prediction, _ = model(test_data, hidden)
    print(prediction[0][0].item())
