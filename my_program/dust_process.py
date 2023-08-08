import cv2 as cv
import numpy as np
import os
from PIL import Image
import xml.etree.ElementTree as ET
import shutil
import warnings

warnings.filterwarnings("ignore")


def rename_file(path):
    imgs = os.listdir(path)
    for img in imgs:
        new_img = int(img.split(".")[0])
        os.rename(path+img, path+"{:>03d}.jpg".format(new_img))


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def random_zoom(img_path, img_file, input_size, boxes, jitter=0.3):
    """随即缩放 -- 保存成新图片"""
    img = Image.open(img_path + img_file)
    img = img.convert("RGB")

    iw, ih = img.size
    w, h = input_size[0], input_size[1]

    rs = iw / ih * rand(1-jitter, 1+jitter) / rand(1-jitter, 1+jitter)
    scaler = rand(0.25, 2)
    if rs < 1:
        nh = int(h * scaler)
        nw = int(nh * rs)
    else:
        nw = int(w * scaler)
        nh = int(nw / rs)
    img = img.resize((nw, nh), Image.BICUBIC)

    # 缩放后粘贴在固定大小[640, 640]的图上
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    img_new = Image.new("RGB", (w, h), (128, 128, 128))
    img_new.paste(img, (dx, dy))

    # 随机翻转
    if rand() > 0.5:
        img_new = img_new.transpose(Image.FLIP_LEFT_RIGHT)

    # 转换成ndarray
    img_new = np.array(img_new, np.uint8)
    img_new = img_new[:, :, [1, 0, 2]]
    # cv.imshow("img", img_new)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return img_new, boxes, w, h, nw, nh, iw, ih, dx, dy


def hsv_changing(img, hu=0.5, s=0.5, v=0.5):
    dtype = img.dtype
    r = np.random.uniform(-1, 1, 3) * [hu, s, v] + 1
    hu, s, v = cv.split(cv.cvtColor(img, cv.COLOR_RGB2HSV))

    x = np.arange(0, 256, dtype=r.dtype)
    lut_h = ((r[0] * x) % 180).astype(dtype)
    lut_s = np.clip(r[1] * x, 0, 255).astype(dtype)
    lut_v = np.clip(r[2] * x, 0, 255).astype(dtype)

    img_new = cv.merge((cv.LUT(hu, lut_h), cv.LUT(s, lut_s), cv.LUT(v, lut_v)))
    img_new = cv.cvtColor(img_new, cv.COLOR_HSV2RGB)
    return img_new


def judge_box(img_path, img_file, img, xml_path, xml_file, boxes):
    """
    判断缩放后标签列表是否为None
    None则不生成新图片，not None则生成新图片
    """
    if len(boxes) > 0:
        img_name = img_file.split(".")[0]
        img_copy = "{}_new.jpg".format(img_name)
        cv.imwrite(img_path+img_copy, img)
        # shutil.copy(img_path + img_file, img_path + img_copy)

        # 生成新标签文件
        xml_name = xml_file.split(".")[0]
        xml_copy = "{}_new.xml".format(xml_name)
        shutil.copy(xml_path + xml_file, xml_path + xml_copy)

        et = ET.parse(xml_path + xml_copy)
        annotation = et.getroot()
        objs = annotation.findall("object")
        # 删除缩放后超过边界的图像标签
        b = len(objs)
        for obj in objs:
            if len(boxes) == len(objs):
                break
            annotation.remove(obj)
            b -= 1

        # 修改标签坐标
        # for i, (obj, box) in enumerate(zip(objs, boxes)):
        for i, obj in enumerate(objs):
            bnd = obj.find("bndbox")
            bnd.find("xmin").text = str(boxes[i][:, 0][0])
            bnd.find("ymin").text = str(boxes[i][:, 1][0])
            bnd.find("xmax").text = str(boxes[i][:, 2][0])
            bnd.find("ymax").text = str(boxes[i][:, 3][0])
        et.write(os.path.join(xml_path, xml_copy))


def get_boss(xml_path):
    et = ET.parse(xml_path)
    annotation = et.getroot()
    objs = annotation.findall("object")
    boxes = []
    for obj in objs:
        box = []
        bnd = obj.find("bndbox")
        xmin = int(bnd.find("xmin").text)
        ymin = int(bnd.find("ymin").text)
        xmax = int(bnd.find("xmax").text)
        ymax = int(bnd.find("ymax").text)

        box.append(xmin)
        box.append(ymin)
        box.append(xmax)
        box.append(ymax)

        boxes.append(box)
    return boxes


def correct_box(xml_path, xml_file, boxes, w, h, nw, nh, iw, ih, dx, dy):
    # print(boxes)
    boxes = np.array([[np.array(boxes[i])] for i in range(len(boxes))])
    # print("old:", boxes)
    # for i in range(len(boxes)):

    et = ET.parse(xml_path + xml_file)
    root = et.getroot()
    objs = root.findall("object")

    i = 0
    while i < len(boxes):
        boxes[i] = np.array(boxes[i])
        boxes[i][:, [0, 2]] = boxes[i][:, [0, 2]] * nw / iw + dx
        boxes[i][:, [1, 3]] = boxes[i][:, [1, 3]] * nh / ih + dy
        boxes[i][:, 0: 2][boxes[i][:, 0: 2] < 0] = 0
        boxes[i][:, 2][boxes[i][:, 2] > w] = w
        boxes[i][:, 3][boxes[i][:, 3] > h] = h
        box_x = boxes[i][:, 2] - boxes[i][:, 0]
        box_y = boxes[i][:, 3] - boxes[i][:, 1]
        if not np.logical_and(box_x > 1, box_y > 1):
            boxes = np.delete(arr=boxes, obj=i, axis=0)
            root.remove(objs[i])   # 删除对应的object
        i += 1
    et.write(xml_path + xml_file)
    # print("new:", boxes)
    return boxes


def delete_redundancy(path, file):
    if "new_new" in file:
        os.remove(path + file)


def main():
    input_size = [640, 640]
    img_root = "./images/"
    xml_root = "./images_xml/"
    img_list = os.listdir(img_root)
    xml_list = os.listdir(xml_root)

    for img_file, xml_file in zip(img_list, xml_list):
        if os.path.exists(img_root + img_file) & os.path.exists(xml_root + xml_file):
            # 获取标签位置
            boxes = get_boss(os.path.join(xml_root, xml_file))

            # 随即缩放
            img, boxes, w, h, nw, nh, iw, ih, dx, dy = random_zoom(img_root, img_file, input_size, boxes)

            # 修改标签位置
            boxes = correct_box(xml_root, xml_file, boxes, w, h, nw, nh, iw, ih, dx, dy)

            # 生成新图像及对应的标签
            judge_box(img_root, img_file, img, xml_root, xml_file, boxes)

    img_list = os.listdir(img_root)
    xml_list = os.listdir(xml_root)
    for img_file, xml_file in zip(img_list, xml_list):
        delete_redundancy(img_root, img_file)
        delete_redundancy(xml_root, xml_file)


def practice_enhence():
    img = cv.imread("../images/001.jpg")
    cv.imshow("img", img)
    img_new = hsv_changing(img)
    cv.imshow("img_new", img_new)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    # img_path = "./images/"
    # rename_file(img_path)

    # main()

    practice_enhence()

