import torch


def _():
    batch_size = 1
    seq_size = 3
    input_size = 4
    hidden_size = 2

    cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

    dataset = torch.randn(seq_size, batch_size, input_size)
    hidden = torch.zeros(batch_size, hidden_size)
    print(hidden.shape)

    for idx, input in enumerate(dataset):
        print("="*20, idx, "="*20)
        print("input_size=", input.shape)
        hidden = cell(input, hidden)
        print("hidden_size=", hidden.shape)

    for i, input in enumerate(dataset):
        print(input.shape)
        print(cell(input, hidden))


    batch_size = 1
    seq_len = 3
    input_size = 4
    hidden_size = 2
    num_layers = 1

    cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    input = torch.randn(seq_len, batch_size, input_size)   # input: seq_size, batch_size, input_size
    hidden = torch.zeros(num_layers, batch_size, hidden_size)   # hidden: num_layers, batch_size, hidden_size

    output, hidden = cell(input, hidden)

    print("output_size=", output.shape)
    print("hidden_size", hidden.shape)


def __():
    str1 = "hello"
    str2 = ""
    list = []
    for i in range(len(str1)):
        list.append(str1[i])

    import pandas as pd
    import numpy as np
    import torch
    import torch.functional as F

    df = pd.DataFrame(list)

    x = pd.get_dummies(df)
    one_hot = np.asarray(x)
    print(one_hot.shape)
    # from sklearn.preprocessing import OneHotEncoder
    # one_hot = OneHotEncoder()
    # one_hot = one_hot.fit_transform(df).toarray()
    # print(one_hot)
    input = torch.tensor(one_hot)
    hidden = torch.zeros(5, 1, 1)

    cell = torch.nn.RNN(input_size=4, hidden_size=1, num_layers=1)
    h, hidden = cell(input, hidden)
    print(h.shape, hidden.size)
    softmax = F.softmax(hidden)


import torch
from torch import nn

list = ["e", "h", "l", "o"]
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]

eye = [[1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 1, 0],
       [0, 0, 0, 1]]

one_hot_data = [eye[x] for x in x_data]

inputs = torch.tensor(one_hot_data, dtype=torch.float32).view(-1, 1, 4)
labels = torch.LongTensor(y_data).view(-1, 1)

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.rnncell = nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)

    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)

model = Model(4, 4, 1)
loss_func = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(15):
    loss = 0.0
    optim.zero_grad()
    hidden = model.init_hidden()
    for input, label in zip(inputs, labels):
        print(label.shape, hidden.shape)
        hidden = model(input, hidden)
        loss += loss_func(hidden, label)
    loss.backward()
    optim.step()
    print("第%d轮迭代：loss=%.4f" % (epoch+1, loss))


import torch
from torch import nn

list = ["e", "h", "l", "o"]
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]

eye = [[1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 1, 0],
       [0, 0, 0, 1]]

one_hot_data = [eye[x] for x in x_data]

inputs = torch.tensor(one_hot_data, dtype=torch.float32).view(-1, 1, 4)
labels = torch.LongTensor(y_data).view(-1, 1)
print(labels.shape)

class Rnn(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers):
        super(Rnn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, input):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        h, hidden = self.rnn(input, hidden)
        return h.view(-1, self.hidden_size, 1)

rnn = Rnn(4, 4, 1, 1)
loss_func = nn.CrossEntropyLoss()
optim = torch.optim.Adam(rnn.parameters(), lr=0.05)

# inputs.shape = (5, 1, 4), hidden.shape(1, 1, 4)
for epoch in range(15):
    optim.zero_grad()
    h = rnn(inputs)
    loss_value = loss_func(h, labels)
    print("第%d轮迭代：loss=%.4f" % (epoch + 1, loss_value))
    loss_value.backward()
    optim.step()
print(h.shape, labels.shape)



import torch
from torch import nn

num_class = 4
input_size = 4
hidden_size = 8
embedding_size = 10
num_layers = 2
batch_size = 1
seqLen = 5

list = ["e", "h", "l", "o"]
x_data = [[1, 0, 2, 2, 3]]
y_data = [3, 1, 2, 3, 2]

inputs = torch.LongTensor(x_data)   # (batch, seq_len)
labels = torch.LongTensor(y_data)   # batch * seq_len

class embedding(nn.Module):
    def __init__(self):
        super(embedding, self).__init__()
        self.emb = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_size)
        self.rnn = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)
        x = self.emb(x)
        x, _ = self.rnn(x, hidden)
        x = self.linear(x)
        return x.view(-1, num_class)   # 转换成矩阵

embed = embedding()

loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(embed.parameters(), lr=0.01)

for epoch in range(15):
    optim.zero_grad()
    hidden = embed(inputs)   # (batch, seqLen, EmbeddingSize)
    loss_value = loss(hidden, labels)
    loss_value.backward()
    optim.step()
    print("第%d轮迭代：Loss=%.4f" % (epoch+1, loss_value))

