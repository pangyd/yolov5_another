import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt


"""
低通滤波：去噪，模糊图像
高通滤波（频率高的部分通过）：有利于找到图像边界
"""


# 方框滤波
def box_filter(img):
    img_filter = cv.boxFilter(img, ddepth=-1, ksize=(50, 50), normalize=True)
    return img_filter


# 均值滤波
def mean_filter(img):
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_filter = cv.blur(img, ksize=(3, 3))
    return img_filter


# 高斯滤波  --  去除噪声，保留更多的图像细节，但不会考虑像素是否位于边界，所以会把边界模糊掉
def gaussian_filter(img):
    """中心和周围像素服从正态分布，给定标准差可以求出每个像素得权值，归一化再加权平均"""
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_filter = cv.GaussianBlur(img, ksize=(3, 3), sigmaX=1, sigmaY=1)
    kernal_x = cv.getGaussianKernel(3, 0)
    plt.figure(figsize=(12, 6))
    plt.title("gaussian blur")
    plt.imshow(img_filter)
    plt.show()


# 中值滤波  --  用像素点临近灰度的中值代替搞搞掂灰度值
def median_filter(img):
    img_filter = cv.medianBlur(img, ksize=9)
    return img_filter


# 双边滤波  --  有效保证边缘信息
def bi_filter(img):
    """空间高斯权重，灰度值相似度高斯权重"""
    img_filter = cv.bilateralFilter(img, d=-1, sigmaColor=50, sigmaSpace=30)
    return img_filter


# 直方图均匀化
def rectangle_mean(img):
    img_new = cv.equalizeHist(img)
    return img_new

img = cv.imread("144.jpg")
cv.imshow("img", img)
img = rectangle_mean(img)
cv.imshow("img_new", img)
cv.waitKey(0)
cv.destroyAllWindows()