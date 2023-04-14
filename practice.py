import os
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
    img = Image.open("144.jpg")
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

    img = cv.imread("144.jpg")
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
    img = Image.open("144.jpg")
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

    img = cv.imread("144.jpg")
    img, box = vertical_flip(img, box)
    img[429-328:429-200, 100:228, :] = img_gray

    cv.imshow("img_new", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 复制标签对应的原图到训练集中
def copy_train():
    import shutil
    smoke_json = os.listdir("smoke/smoke_json")
    images = os.listdir("smoke/images")
    if not os.path.exists("smoke/smoke_image"):
        os.mkdir("smoke/smoke_image")

    for j in smoke_json:
        j_file = j.split(".")[0]
        for i in images:
            if i in os.listdir("smoke/smoke_image"):
                continue
            else:
                i_file = i.split(".")[0]
                if j_file == i_file:
                    shutil.copy("smoke/images/{}".format(i), "smoke/smoke_image/")
        if j == smoke_json[-1]:
            break
copy_train()



