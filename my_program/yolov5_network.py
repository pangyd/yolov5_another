import torch
from torch import nn
import numpy as np


def autopad(k, p=None):
    if p is None:
        p = k // 2
    return p


class Focus(nn.Module):
    def __init__(self, c1=3, c2=64, k=1, s=1):
        super(Focus, self).__init__()
        self.conv1 = Conv(c1, 12, k, s, autopad(k))
        self.conv2 = Conv(12, c2, k, s, autopad(k))
    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1)
        x = self.conv2(x)
        return x


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x + torch.sigmoid(x)


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p))
        self.bn = nn.BatchNorm2d(c2, eps=0.01, momentum=0.1)
        self.silu = SiLU()
    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))


class BottleNeck(nn.Module):
    """csp bottleneck"""
    def __init__(self, c1, c2, e=0.5, shortcut=True):
        super(BottleNeck, self).__init__()
        c_ = int(c1 * e)
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c_, c2, 3, 1)
        self.act = shortcut and c1 == c2

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.act else self.conv2(self.conv1(x))


class C3(nn.Module):
    """3 * conv + csp bottleneck"""
    def __init__(self, c1, c2, shortcut=True, e=0.5, n=1):
        super(C3, self).__init__()
        c_ = int(c2 * e)
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c1, c_, 1, 1)
        self.conv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[BottleNeck(c_, c_) for _ in range(n)])

    def forward(self, x):
        x = self.conv3(torch.cat(self.m(self.conv1(x)), self.conv2(x)))
        return x


class SPPBottleneck(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPPBottleneck, self).__init__()
        c_ = c1 // 2
        self.conv1 = Conv(c1, c_)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.conv2 = Conv(c_ * (len(k) + 1), c2)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + self.m(x), dim=1)
        x = self.conv2(x)
        return x


class CSPDarknet(nn.Module):
    def __init__(self, base_channel, in_channel, phi, pretrained):
        """
        :param base_channel: 原始通道数，默认3
        :param in_channel: 转换通道数，默认64
        :param phi: 模型类型，默认s
        :param pretrained: 是否需要预训练模型
        """
        super(CSPDarknet, self).__init__()
        # 640, 640, 3 --> 320, 320, 64
        self.focus = Focus(base_channel, in_channel, k=3)
        # 320, 320, 64 --> 160, 160, 128
        self.res1 = nn.Sequential(Conv(in_channel, in_channel*2, k=3, s=2),
                                  C3(in_channel*2, in_channel*2))
        # 160, 160, 128 --> 80, 80, 256
        self.res2 = nn.Sequential(Conv(in_channel*2, in_channel*4, k=3, s=2),
                                  C3(in_channel*4, in_channel*4))
        # 80, 80, 256 --> 40, 40, 512
        self.res3 = nn.Sequential(Conv(in_channel*8, in_channel*8, k=3, s=2),
                                  C3(in_channel*8, in_channel*8))
        # 40, 40, 512 --> 20, 20, 1024
        self.res4 = nn.Sequential(Conv(in_channel*16, in_channel*16, k=3, s=2),
                                  SPPBottleneck(in_channel*16, in_channel*16),
                                  C3(in_channel*16, in_channel*16))
        if pretrained:
            url = {
                's': 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_s_backbone.pth',
                'm': 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_m_backbone.pth',
                'l': 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_l_backbone.pth',
                'x': 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_x_backbone.pth',
            }[phi]
            # checkpoint = torch.hub.load_state_dict_from_url(url=url, model_dir="./model_dir")
            # self.load_state_dict(checkpoint, strict=False)

    def forward(self, x):
        x = self.focus(x)
        x = self.res1(x)
        fea1 = self.res2(x)
        fea2 = self.res3(fea1)
        fea3 = self.res4(fea2)
        return fea1, fea2, fea3


class Yolo(nn.Module):
    def __init__(self, anchor_mask, classes_num, backbone, phi, pretrained, input_size=[640, 640]):
        super(Yolo, self).__init__()
        depth_dict = {'s': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33}
        width_dict = {'s': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25}
        in_channel = int(depth_dict[phi] * 64)   # l: 64
        base_width = max(round(width_dict[phi]*3), 1)   # l: 3
        if backbone == "cspdarknet":
            self.backbone = CSPDarknet(3, in_channel, phi, pretrained=True)

        # 20, 20, 1024 --> 20, 20, 512
        self.conv1 = Conv(in_channel*16, in_channel*8, 3, 1, 1)
        # 20, 20, 512 --> 40, 40, 512
        self.up1 = nn.Upsample(scale_factor=2)
        # 40, 40, 1024 --> 40, 40, 512
        self.csp1 = C3(in_channel*16, in_channel*8, n=base_width)

        # 40, 40, 512 --> 40, 40, 256
        self.conv2 = Conv(in_channel*8, in_channel*4, 3, 1, 1)
        # 40, 40, 256 --> 80, 80, 256
        self.up2 = nn.Upsample(scale_factor=2)
        # 80, 80, 512 --> 80, 80, 256 --> yolo head 1
        self.csp2 = C3(in_channel*8, in_channel*4, n=base_width)

        # 80, 80, 256 --> 40, 40, 256
        self.down1 = Conv(in_channel*4, in_channel*4, 3, 2, 1)
        # 40, 40, 512 --> 40, 40, 512 --> yolo head 2
        self.csp3 = C3(in_channel*8, in_channel*8)

        # 40, 40, 512 --> 20, 20, 512
        self.down2 = Conv(in_channel*8, in_channel*8, 3, 2, 1)
        # 20, 20, 1024 --> 20, 20, 1024 --> yolo head 3
        self.csp4 = C3(in_channel*16, in_channel*16)

        # 80, 80, 256 --> 80, 80, 4+1+num_classes
        self.yolo_head3 = nn.Conv2d(in_channel*4, 3 * (len(anchor_mask[2]) + 1 + classes_num), 1, 1)
        # 40, 40, 512 --> 40, 40, 4+1+num_classes
        self.yolo_head2 = nn.Conv2d(in_channel*8, 3 * (len(anchor_mask[2]) + 1 + classes_num), 1, 1)
        # 20, 20, 1024 --> 20, 20, 4+1+num_classes
        self.yolo_head1 = nn.Conv2d(in_channel*16, 3 * (len(anchor_mask[2]) + 1 + classes_num), 1, 1)

    def forward(self, x):
        # feature extraction
        fea1, fea2, fea3 = self.backbone(x)

        x1 = self.conv1(fea3)
        x1_up = self.up1(x1)

        x2 = self.conv2(self.csp1(torch.cat([fea2, x1_up], dim=1)))
        x2_up = self.up2(x2)

        x3 = self.csp2(torch.cat([fea1, x2_up], dim=1))
        x3_down = self.down1(x3)

        x2 = self.csp3(torch.cat([x3_down, x2], dim=1))
        x2_down = self.down2(x2)

        x1 = self.csp4(torch.cat([x2_down, x1], dim=1))

        yolo1 = self.yolo_head1(x1)
        yolo2 = self.yolo_head2(x2)
        yolo3 = self.yolo_head3(x3)

        return yolo1, yolo2, yolo3


anchor_mask = np.ones(shape=(3, 4))
model = Yolo(anchor_mask, 80, backbone="cspdarknet", phi="l", pretrained=True)
print(model)





