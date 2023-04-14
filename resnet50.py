import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from torchvision import datasets, utils
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


"""residual net"""


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channels * 4)
        )

    def forward(self, x):
        out = self.down(x)
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1, 1, 1], padding=[0, 1, 0], downsample=False):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=padding[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=stride[2], padding=padding[2], bias=False),
            nn.BatchNorm2d(out_channels * 4)
        )
        self.stride = stride

        if downsample:
            # self.downsample = DownSample(in_channels, out_channels, stride=1)
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride[1], bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        out = self.bottleneck(x)

        # if self.downsample is not None:
        #     residual = self.downsample(x)
        # else:
        #     residual = x

        residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, Bottleneck, num_classes=10):
        self.in_channels = 64
        super(ResNet50, self).__init__()
        # 残差块前处理
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.__make_layer(Bottleneck, 64, [[1, 1, 1]] * 3, [[0, 1, 0]] * 3)
        self.layer2 = self.__make_layer(Bottleneck, 128, [[1, 2, 1]] + [[1, 1, 1]] * 3, [[0, 1, 0]] * 4)
        self.layer3 = self.__make_layer(Bottleneck, 256, [[1, 2, 1]] + [[1, 1, 1]] * 5, [[0, 1, 0]] * 6)
        self.layer4 = self.__make_layer(Bottleneck, 512, [[1, 2, 1]] + [[1, 1, 1]] * 2, [[0, 1, 0]] * 3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def __make_layer(self, block, out_channels, strides, paddings):
        layer = []
        for i in range(0, len(strides)):
            # 第一层降采样
            if i == 0:
                layer.append(block(self.in_channels, out_channels, strides[0], paddings[0], downsample=True))
            else:
                layer.append(block(self.in_channels, out_channels, strides[i], paddings[i]))
            self.in_channels = out_channels * 4
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out


class My_dataset(Dataset):
    """读取自己的数据集"""
    def __init__(self, label_path):
        super(My_dataset, self).__init__()
        with open(label_path, "r") as f:
            self.imgs = list(map(lambda x: x.strip().split(" "), f))

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        # 读取img
        img = Image.open(img_path)
        img = img.convert("RGB")
        trans = transforms.Compose([transforms.Resize((320, 320)), transforms.ToTensor()])
        img = trans(img)
        label = int(label)
        return img, label

    def __len__(self):
        return len(self.imgs)


def data_prepare(transform):
    training_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    testing_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=training_data, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=testing_data, batch_size=64, shuffle=True, drop_last=True)
    return train_loader, test_loader


def train(model, train_loader, num_training, loss_func, optimizer, i):
    running_loss = 0
    total = 0
    correct = 0
    # 模型训练
    res50.train()
    for img, label in train_loader:
        output = model(img)
        loss = loss_func(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_training += 1
        running_loss += loss.item()

        print("第{}个batch的损失为：{}".format(num_training, round(running_loss, 4)))
        running_loss = 0
        # if num_training % 5 == 0:
        #     print("{}-{}次训练的平均损失为：{}".format(num_training-4, num_training, round(running_loss / 5, 4)))
        #     running_loss = 0

        # 计算准确率
        _, predict = torch.max(output.data, dim=1)
        total += label.size(0)   # 样本数
        correct += (predict == label).sum().item()
    print("训练准确率:{}%".format(round(100 * correct / total, 2)))

    torch.save(model, "res50_{}.pth".format(i + 1))


def test(model, test_loader, loss_func):
    total = 0
    correct = 0
    # 模型验证
    res50.eval()
    with torch.no_grad():
        for img, label in test_loader:
            output = model(img)
            loss = loss_func(output, label)

            _, predict = torch.max(output, 1)  # 每行最大值以及最大值的位置
            total += label.size(0)  # 验证机的数据量
            correct += (predict == label).sum()  # label不是稀疏矩阵
    print("测试集准确率={}%".format(round(correct / total * 100, 2)))


if __name__ == "__main__":

    pd.set_option("display.max_rows", None)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    # 准备数据
    # train_loader, test_loader = data_prepare(transform)

    # 自己的数据集
    mydata = My_dataset("./label.txt")   # mydata: 可迭代数据集
    data_loader = DataLoader(mydata, batch_size=4, shuffle=True, drop_last=True)

    # 导入残差网络
    res50 = ResNet50(Bottleneck)
    print(res50)

    # images, labels = next(iter(train_loader))
    # img = utils.make_grid(images)
    #
    # img = img.numpy().transpose((1, 2, 0))

    epoch = 10
    num_training = 0
    num_testing = 0

    # 设置损失函数和优化器
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=res50.parameters(), lr=1e-5)

    for i in range(epoch):
        print("第{}轮训练开始".format(i+1))

        train(res50, data_loader, num_training, loss_func, optimizer, i)

        # test(res50, test_loader, loss_func)





