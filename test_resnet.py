from torchvision import transforms
import torch
import numpy as np
from torchvision import datasets, utils
from torch.utils.data import DataLoader
from resnet50 import *
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                transforms.Resize((224, 224))])

training_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
testing_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

# 导入残差网络
res50 = ResNet50(Bottleneck)

# 返回字典
batch_size = 64
train_data = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_data = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True, drop_last=True)

images, labels = next(iter(train_data))
print(images.shape)
img = utils.make_grid(images)   # 多图拼成一张图

img = img.numpy().transpose((1, 2, 0))
mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
img = img * std + mean
print([labels[i] for i in range(64)])
plt.imshow(img)
plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = res50.to(device)
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

epochs = 10
for epoch in range(epochs):
    running_loss = 0
    running_correct = 0
    model.train()
    print("Epoch {}/{}".format(epoch+1, epochs))
    print("-" * 10)
    # 训练
    for x_train, y_train in train_data:
        x_train, y_train = x_train.to(device), y_train.to(device)
        optimizer.zero_grad()   # 清空当前梯度
        outputs = model(x_train)
        _, pred = torch.max(outputs.data, 1)
        loss = cost(outputs, y_train)

        loss.backward()   # \
        optimizer.step()  # /
        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train.data)

    # 测试
    testing_loss = 0
    testing_correct = 0
    model.eval()
    for x_test, y_test in test_data:
        x_test, y_test = x_test.to(device), y_test.to(device)
        outputs = model(x_test)
        _, pred = torch.max(outputs.data, 1)
        loss = cost(outputs, y_test)

        testing_loss += loss.item()
        testing_correct += torch.sum(pred == y_test.data)

    print("Training Loss:{:.4f}, Training Accuracy:{:.4f}".format(running_loss/len(training_data), 100*running_correct/len(training_data)))
    print("Testing Loss:{:.4f}, Testing Accuracy:{:.4f}".format(testing_loss/len(testing_data), 100*testing_correct/len(testing_data)))



