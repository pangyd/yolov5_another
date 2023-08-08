import os

import cv2
import torch
import cv2 as cv


# net = torch.load("logs/best_epoch_weights_5000.pth", map_location=torch.device("cpu"))
#
# print("网络类型：", type(net))
# print("网络长度", len(net))
#
# for key in net.keys():
#     print(key)
#     # print("{}:{}".format(key, net[key]))

# import os
# os.system("git")

from PIL import Image
import numpy as np


def horizontal_flip(img, box):
    h, w, c = img.shape
    img = img[:, ::-1, :]
    box[:, [0, 2]] = w - box[:, [2, 0]]
    return img, box


def vertical_flip(img, box):
    h, w, c = img.shape
    img = img[::-1, :, :]
    box[:, [1, 3]] = h - box[:, [3, 1]]
    return img, box


def rot90_clockwise(img, box):
    h, w, c = img.shape
    img = cv.transpose(img)
    img = cv.flip(img, 1)
    box[:, [0, 1, 2, 3]] = box[:, [3, 0, 1, 2]]
    box[:, [0, 2]] = h - box[:, [0, 2]]
    return img, box


def rotate_angle(img, box):
    h, w, c = img.shape
    # 获取旋转类的对象
    rotate_matrix = cv.getRotationMatrix2D((w/2, h/2), 180, 1)
    img_new = cv.warpAffine(img, rotate_matrix, dsize=(w, h))
    cv.imshow("img_new", img_new)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return img_new, box


def affine_change(img, box):
    h, w, c = img.shape

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[100, 100], [200, 50], [100, 250]])
    affine_matrix = cv.getAffineTransform(pts1, pts2)
    img_new = cv.warpAffine(img, affine_matrix, (w, h))
    return img_new, box


# 透视变换：改变拍摄角度
def perspective_transform(img, box):
    h, w, c = img.shape
    pts1 = np.float32([[30, 50], [200, 50], [80, 200], [50, 20]])
    pts2 = np.float32([[70, 100], [200, 50], [100, 250], [15, 50]])
    perspect_change = cv.getPerspectiveTransform(pts1, pts2)
    img = cv.warpPerspective(img, M=perspect_change, dsize=(w, h))
    return img, box


def pyrimid_change(img):
    img_de = cv.pyrDown(img)
    img_in = cv.pyrDown(img)
    return img, img_de, img_in


def check():
    import matplotlib.pyplot as plt
    # 原始img和box
    img = Image.open("../144.jpg")
    img_gray = Image.new("RGB", size=(128, 180), color=(128, 128, 128))
    img.paste(img_gray, box=(100, 200))
    img = np.array(img, dtype=np.uint8)
    img_gray = np.array(img_gray, dtype=np.uint8)
    box = np.array([np.array([100, 200, 228, 380])])
    # cv.imshow("img", img)
    plt.subplot(121)
    plt.imshow(img)
    plt.title("old")
    plt.axis(False)

    img = cv.imread("../144.jpg")
    img_new, box = affine_change(img, box)
    img_new[100:228, 429-380:429-200, :] = cv.transpose(img_gray)
    plt.subplot(122)
    plt.imshow(img_new)
    plt.title("new")
    plt.axis(False)
    plt.show()


#     cv.imshow("img_new", img_new)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
# rot_check()


def detect_box():
    img = Image.open("../144.jpg")
    img_gray = Image.new("RGB", size=(128, 128), color=(128, 128, 128))
    img.paste(img_gray, box=(100, 200))
    box = np.array([np.array([100, 200, 228, 328])])
    # img.show()

    img = np.array(img, dtype=np.uint8)
    img_gray = np.array(img_gray, dtype=np.uint8)
    img = img[:, :, :-1]
    print(img.shape)
    print(img_gray.shape)
    cv.imshow("img", img)

    img = cv.imread("../144.jpg")
    img, box = vertical_flip(img, box)
    img[429-328:429-200, 100:228, :] = img_gray

    cv.imshow("img_new", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 复制标签对应的原图到训练集中
def copy_train():
    import shutil
    smoke_json = os.listdir("../smoke/smoke_json")
    images = os.listdir("../smoke/images")
    if not os.path.exists("../smoke/smoke_image"):
        os.mkdir("../smoke/smoke_image")

    for j in smoke_json:
        j_file = j.split(".")[0]
        for i in images:
            if i in os.listdir("../smoke/smoke_image"):
                continue
            else:
                i_file = i.split(".")[0]
                if j_file == i_file:
                    shutil.copy("smoke/images/{}".format(i), "../smoke/smoke_image/")
        if j == smoke_json[-1]:
            break



# import re
#
# if re.match(r'[a-z]+_trainval\.txt', 'trainval.txt'):
#     print("匹配成功")
# else:
#     print("匹配失败")

# import json
# json_list = []
# with open('a.json', 'a') as file:
#     for i in range(10):
#         json_list.append({'a': i, 'b': i+1})
#     json.dump(json_list, file)


def get_data(dir):
    sub1_dir = os.listdir(dir)
    sub1_dir_list = []

    # 第一子目录
    for sub1 in sub1_dir:   # sub1: ...银行
        sub1_dir_list.append(os.path.join(dir, sub1))   # multi_test_data/测试图片集01_42150/...银行

    sub2_dir_list = []
    all_len = 0
    for bank_dir in sub1_dir_list:
        sub2_list = os.listdir(bank_dir)   # 1, 2, 3
        for sub2 in sub2_list:
            sub2_path = os.path.join(bank_dir, sub2)
            sub2_dir_list.append(sub2_path)

            # 生成目录
            # output_dir = sub2_path.replace('multi_test_data', out_dir)
            # if not os.path.isdir(output_dir):
            #     os.makedirs(output_dir)
    for sub2_dir in sub2_dir_list:
        all_len += len(os.listdir(sub2_dir))
    print(all_len)
    return sub2_dir_list


def remove_shadow():
    """去阴影，适用于彩色图"""
    img = cv.imread("huidan_data/remove_shadow.jpg")
    h, w = img.shape[0], img.shape[1]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 绘制图像直方图，找到阴影部分
    hist = cv.calcHist([gray], [0], None, [256], [0, 255])
    #计算阈值
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # 定义结构元素
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    # 对图像进行形态学操作
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    # cv.imshow("ori", img)
    # cv.imshow("_", opening)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

def noise_inhibition():
    """噪声抑制"""
    img = cv.imread("huidan_data/0415_007_1_huidan.jpg")
    h, w = img.shape[0], img.shape[1]
    # img_denoised = cv.fastNlMeansDenoising(img, None, 10, 7, 21)   # 灰度图
    # h: 过滤器强度，t, s: 奇数
    img_denoised = cv.fastNlMeansDenoising(img, None, h=30, templateWindowSize=7, searchWindowSize=21)
    cv.namedWindow("ori", 0)
    cv.resizeWindow("ori", int(w/4), int(h/4))
    cv.imshow("ori", img)
    cv.namedWindow("denoised", 0)
    cv.resizeWindow("denoised", int(w/4), int(h/4))
    cv.imshow("denoised", img_denoised)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # cv.imwrite("huidan_data/remove_shadow_done2.jpg", img_denoised)
# noise_inhibition()

def brightness():
    img = cv.imread("huidan_data/shuiying.jpg")
    h, w = img.shape[0], img.shape[1]
    # 增强对比度和亮度
    alpha = 1.5  # 对比度增强因子
    beta = 50  # 亮度增强因子

    # 对比度增强
    img_contrast = cv.convertScaleAbs(img, alpha=alpha, beta=0)

    # 亮度增强
    img_brightness = cv.addWeighted(img_contrast, 1, np.zeros(img.shape, img.dtype), 0, beta)

    cv.namedWindow("ori", 0)
    cv.resizeWindow("ori", int(w / 4), int(h / 4))
    cv.imshow("ori", img)
    cv.namedWindow("bright", 0)
    cv.resizeWindow("bright", int(w/4), int(h/4))
    cv.imshow("bright", img_brightness)
    cv.waitKey(0)
    cv.destroyAllWindows()


def filter():
    img = cv.imread("huidan_data/0415_007_1_huidan.jpg")
    h, w = img.shape[0], img.shape[1]
    # img_filter = cv.bilateralFilter(img, d=-1, sigmaColor=50, sigmaSpace=5)   # 双边滤波
    # img_filter = cv.medianBlur(img, ksize=9)   # 中值滤波
    img_filter = cv.GaussianBlur(img, ksize=(3, 3), sigmaX=1, sigmaY=1)   # 高斯滤波
    # img_filter = cv.boxFilter(img, ddepth=-1, ksize=(5, 5), normalize=True)   # 方框滤波
    # img_filter = cv.blur(img, ksize=(3, 3))   # 均值滤波
    cv.namedWindow("ori", 0)
    cv.resizeWindow("ori", int(w / 4), int(h / 4))
    cv.imshow("ori", img)
    cv.namedWindow("filter", 0)
    cv.resizeWindow("filter", int(w/4), int(h/4))
    cv.imshow("filter", img_filter)
    cv.waitKey(0)
    cv.destroyAllWindows()
# filter()


def shaping_noise():
    img = cv.imread("huidan_data/remove_shadow.jpg")
    h, w = img.shape[0], img.shape[1]
    cv.namedWindow("ori", 0)
    cv.resizeWindow("ori", int(w / 4), int(h / 4))
    cv.imshow("ori", img)
    # 2. cv2.MORPH_OPEN 先进行腐蚀操作，再进行膨胀操作
    kernel = np.ones((10, 10), np.uint8)
    opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    cv.namedWindow("opening", 0)
    cv.resizeWindow("opening", int(w / 4), int(h / 4))
    cv.imshow('opening', opening)
    cv.imwrite("huidan_data/opening.jpg", opening)

    # 3. cv2.MORPH_CLOSE 先进行膨胀，再进行腐蚀操作
    kernel = np.ones((10, 10), np.uint8)
    closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    cv.namedWindow("closing", 0)
    cv.resizeWindow("closing", int(w / 4), int(h / 4))
    cv.imshow('closing', closing)

    cv.waitKey(0)
    cv.destroyAllWindows()


def binary_process():
    img = cv.imread("huidan_data/remove_white.jpg")
    h, w = img.shape[0], img.shape[1]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.namedWindow("gray", 0)
    cv.resizeWindow("gray", int(w/4), int(h/4))
    cv.imshow("gray", gray)
    thre = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]   # 二值化
    cv.namedWindow("result", 0)
    cv.resizeWindow("result", int(w/4), int(h/4))
    cv.imshow("result", thre)
    kernal = cv.getStructuringElement(cv.MORPH_RECT, (50, 50))   # 形态学操作
    morph = cv.morphologyEx(thre, cv.MORPH_CLOSE, kernal)
    morph_inv = cv.bitwise_not(morph)
    # 全白的图像
    blank = np.zeros_like(img)
    blank.fill(255)
    result = cv.bitwise_and(img, blank, mask=morph_inv)
    # cv.namedWindow("result", 0)
    # cv.resizeWindow("result", int(w/4), int(h/4))
    # cv.imshow("result", result)
    cv.waitKey(0)
    cv.destroyAllWindows()


from torch import nn
import torch
from torchvision import transforms, datasets, models

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
model.add_module("my_layer", nn.Linear(in_features=10, out_features=3))
# model.layer4.add_module("my_conv", nn.Conv2d(512, 256, kernel_size=1, stride=1))

fast_rcnn = models.detection.FasterRCNN
mask_rcnn = models.detection.MaskRCNN
model = models.mobilenet_v2().features
model.out_channels = 1280
# model.avgpool = nn.Sequential()
# model.fc = nn.Sequential()
model = fast_rcnn(backbone=model, num_classes=10)
# print(model)

# from torch.autograd import Variable
# x = Variable(torch.ones(3, 3), requires_grad=True)
# y = x + 2
# print(y)
#
# z = y * y * 3
# out = z.mean()
# out.backward()
# print(x.grad)

# import numpy as np
# x = np.random.randn(3, 5, 5)
# print(x)
# for i in range(3):
#     ex = np.exp(x[i, :, :])
#     print(ex)
#     s = np.sum(ex)
#     print(s)
#     x[i, :, :] = ex / s
# print(np.argmax(x, axis=0))


optim = torch.optim.SGD(params=[0, 1], lr=0.1)
schedule = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)

