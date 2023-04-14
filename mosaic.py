import cv2 as cv
from PIL import Image
import os
import numpy as np
import random
import xml.etree.ElementTree as ET
from torchvision import transforms

# mosaic后的图像大小
h, w = 640, 640

# 设置一张640*640的灰度图
img_gray = Image.new("RGB", (h, w), (128, 128, 128))
# img_ = np.array(img_gray)
# cv.imshow("img_", img_)

"""
将灰度图划分成四个区域，确定中心点
district 1:(center_w, center_h)
district 2:(w - center_w, center_h)
district 3:(center_w, h - center_h)
district 4:(w - center_w, h - center_h)
"""
center = random.uniform(0.4, 0.6)
center_h = int(h * center)
center_w = int(w * center)
district_size = [(center_w, center_h), (w - center_w, center_h), (center_w, h - center_h), (w - center_w, h - center_h)]
xy_list = [(0, 0), (center_w, 0), (0, center_h), (center_w, center_h)]

# 获取原数据
img_path = "img"
img_list = os.listdir(img_path)[:4]
# 读取原xml
xml_path = "images_xml"
xml_list = os.listdir(xml_path)[:4]

xml_new_filename = "new.xml"
tree = ET.parse(open(xml_new_filename))
root = tree.getroot()
new_object = ET.Element("object")   # 添加新object

# 随机取四张
for i, district, xy in zip(range(4), district_size, xy_list):
    random_num = random.randint(0, len(img_list))

    # img = cv.imread(os.path.join(path, img_list[random_num]))
    img = cv.imread(os.path.join(img_path, img_list[i]))

    # 获取原始xml标签文件的object
    xml_file = open(os.path.join(xml_path, xml_list[i]))
    ori_tree = ET.parse(xml_file)
    ori_root = ori_tree.getroot()
    ori_obj_list = ori_root.findall("object")

    # 随机缩放
    img_h, img_w = img.shape[0], img.shape[1]
    random_size = random.uniform(1.3, 1.7)
    img = cv.resize(img, (int(img_w / random_size), int(img_h / random_size)))

    # 随机裁剪
    img_h, img_w = img.shape[0], img.shape[1]
    top = random.uniform(0.1, 0.3)
    bottom = random.uniform(0.7, 0.9)
    top_w, top_h = int(img_w * top), int(img_h * top)
    bottom_w, bottom_h = int(img_w * bottom), int(img_h * bottom)
    img = img[top_h: bottom_h, top_w: bottom_w, :]

    # 修改标签位置
    for ori_obj in ori_obj_list:
        ori_bnd = ori_obj.find("bndbox")
        xmin = int(ori_bnd.find("xmin").text)
        xmax = int(ori_bnd.find("xman").text)
        ymin = int(ori_bnd.find("ymin").text)
        ymax = int(ori_bnd.find("ymax").text)

        # 随即缩放后
        xmin, xmax = int(xmin / random_size), int(xmax / random_size)
        ymin, ymax = int(ymin / random_size), int(ymax / random_size)
        # 随机裁剪后
        # if (xmin >= top_w) & (ymin >= top_h) & (xmax <= bottom_w) & (ymax <= bottom_h):
        #     xmin, ymin = xmin - top_w, ymin - top_h
        #     xmax, ymax = bottom_w - xmax, bottom_h - ymax
        # elif (bottom_w >= xmin >= top_w) & (bottom_h >= ymin >= top_h) & (xmax >= bottom_w) & (ymax >= bottom_h):
        #     xmin, ymin = xmin - top_w, ymin - top_h
        #     xmax, ymax = bottom_w, bottom_h
        # elif (xmin <= top_w) & (ymin <= top_h) & (top_w <= xmax <= bottom_w) & (top_h <= ymax <= bottom_h):
        #     xmin, ymin = top_w, top_h
        #     xmax, ymax = bottom_w - xmax, bottom_h - ymax
        # elif ((xmin >= bottom_w) & (ymin >= bottom_h)) | ((xmax <= top_w) & (ymax <= top_h)):
        #     xmin, xmax, ymin, ymax = 0, 0, 0, 0
        # elif (xmin <= top_w) & (ymin <= top_h) & (xmax >= bottom_w) & (ymax >= bottom_h):
        #     xmin, ymin = top_w, top_h
        #     xmax, ymax = bottom_w, bottom_h

        # 各点坐标相对独立，因此分开讨论即可
        xmin = xmin - top_w
        xmax = xmax - bottom_w
        ymin = ymin - top_h
        ymax = ymax - bottom_h
        if xmin < top_w:
            xmin = 0
        if ymin < top_h:
            ymin = 0

        if (xmin > bottom_w) | (xmax < top_w) | (ymin > bottom_h) | (ymax < top_h):
            # 裁剪区域没有标签，删除该object
            # ori_root.remove(ori_obj)
            pass
        else:
            # 判断有无删除原始object，没删除则添加新object
            name = ET.SubElement(new_object, "name")
            name.text = "dust"
            pose = ET.SubElement(new_object, "pose")
            pose.text = "Unspecified"
            truncated = ET.SubElement(new_object, "truncated")
            truncated.text = "0"
            difficult = ET.SubElement(new_object, "difficult")
            difficult.text = "0"

            # mosaic不同区域的坐标随宽高变化
            if i == 0:
                pass
            elif i == 1:
                xmin, xmax = xmin + xy[0], xmax + xy[0]
            elif i == 2:
                ymin, ymax = ymin + xy[1], ymax + xy[1]
            else:
                xmin, xmax, ymin, ymax = xmin + xy[0], xmax + xy[0], ymin + xy[1], ymax + xy[1]

            bndbox = ET.SubElement(new_object, "bndbox")
            x_min = ET.SubElement(bndbox, "xmin")
            x_min.text = str(xmin)
            y_min = ET.SubElement(bndbox, "ymin")
            y_min.text = str(ymin)
            x_max = ET.SubElement(bndbox, "xmax")
            x_max.text = str(xmax)
            y_max = ET.SubElement(bndbox, "ymax")
            y_max.text = str(ymax)

            root.append(new_object)
            tree.write("*.xml")


    # paste_img_list.append(img_list[random_num])
    h, w = img.shape[0], img.shape[1]

    if (h > district[0]) & (w > district[1]):
        img = img[: district[0], : district[1], :]
    elif (h > district[0]) & (w <= district[1]):
        img = img[: district[0], :, :]
        img = cv.resize(img, (district[0], district[1]))
    elif (h <= district[0]) & (w > district[1]):
        img = img[:, : district[1], :]
        img = cv.resize(img, (district[0], district[1]))
    else:
        img = cv.resize(img, (district[0], district[1]))

    # ndarray --> PIL
    img_new = Image.fromarray(img.astype("uint8")).convert("RGB")
    # paste
    # print(xy)
    img_gray.paste(img_new, xy)

# PIL --> ndarray
img_cv = np.array(img_gray)
cv.imshow("mosaic", img_cv)
cv.waitKey(0)
cv.destroyAllWindows()


