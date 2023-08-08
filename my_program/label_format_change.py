import os

import cv2
import numpy as np
import xml.etree.ElementTree as ET
import pickle
from PIL import Image

classes = ["penlin"]


def convert_regulation(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1]) / 2
    y = (box[2] + box[3]) / 2
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def xml2txt(img_name):
    read_path = open("{}.xml".format(img_name))
    save_path = open("{}.txt".format(img_name), "w")

    tree = ET.parse(read_path)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    for obj in root.iter("object"):
        difficult = obj.find("difficult").text
        cls = obj.find("name").text
        if (cls not in classes) & (difficult == 1):
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find("bndbox")
        b = (float(xmlbox.find("xmin").text), float(xmlbox.find("xmax").text),
             float(xmlbox.find("ymin").text), float(xmlbox.find("ymax").text))
        bb = convert_regulation((w, h), b)
        save_path.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + "\n")


def txt2xml(file_name, img_path, txt_path, save_dir):
    # xml's body
    xml_head = """<annotation>
    <folder>JPEGImages</folder>
    <filename>{}</filename>
    <path>/cpfs/00140ca457f70b1b-000001/yfzx/pycharm_project/demo/pyd_project/my_ppocr/PaddleDetection/dataset/voc/VOCdevkit\{}</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>{}</width>
        <height>{}</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    """
    xml_obj = """<object>
        <name>bill</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{}</xmin>
            <ymin>{}</ymin>
            <xmax>{}</xmax>
            <ymax>{}</ymax>
        </bndbox>
    </object>
    """
    xml_end = """
</annotation>"""

    save_xml = open("{}/{}.xml".format(save_dir, file_name.split('.')[0]), "w")
    save_xml.write(xml_head + xml_obj * len(open(txt_path).readlines()) + xml_end)
    save_xml.close()

    # img = cv2.imread(img_path)
    img = Image.open(img_path)
    h, w = img.size[1], img.size[0]

    # 打开xml
    tree = ET.parse("{}/{}.xml".format(save_dir, file_name.split('.')[0]))
    root = tree.getroot()
    # 修改属性
    root.find('filename').text = '{}.jpg'.format(file_name.split('.')[0])
    root.find('path').text = '/data/yfzx/pycharm_project/demo/pyd_project/my_ppocr/PaddleDetection/dataset/voc/VOCdevkit/VOC2007/JPEGImages/{}.jpg'.format(file_name.split('.')[0])

    size = root.find('size')
    size.find('height').text = str(h)
    size.find('width').text = str(w)

    # 将txt中的数据转换成xml需要的数据
    objs = root.findall('object')
    with open(txt_path, 'r') as txt:
        for obj, line in zip(objs, txt.readlines()):
            line = line.strip().split(" ")
            center_x = round(float(str(line[1]).strip()) * w)
            center_y = round(float(str(line[2]).strip()) * h)
            box_w = round(float(str(line[3]).strip()) * w)
            box_h = round(float(str(line[4]).strip()) * h)

            xmin = str(int(center_x - box_w / 2))
            xmax = str(int(center_x + box_w / 2))
            ymin = str(int(center_y - box_h / 2))
            ymax = str(int(center_y + box_h / 2))

            bndbox = obj.find('bndbox')
            bndbox.find('xmin').text = xmin
            bndbox.find('xmax').text = xmax
            bndbox.find('ymin').text = ymin
            bndbox.find('ymax').text = ymax

    tree.write("{}/{}.xml".format(save_dir, file_name.split('.')[0]), encoding='UTF-8')

    # # while True:
    # for line in open("alter.txt").readlines():
    #     # line = open("alter.txt", "r").readline()
    #
    #     # if line == "":
    #     #     break
    #     if "filename" in line:
    #         line = line.replace("{}", "{}.jpg".format(file_name.split('.')[0]))
    #         save_txt.write(line)
    #     elif "path" in line:
    #         line = line.replace("{}", "{}.jpg".format(file_name.split('.')[0]))
    #         save_txt.write(line)
    #     elif "width" in line:
    #         line = line.replace("{}", "{}".format(w))
    #         save_txt.write(line)
    #     elif "height" in line:
    #         line = line.replace("{}", "{}".format(h))
    #         save_txt.write(line)
    #     elif "depth" in line:
    #         line = line.replace("{}", "3")
    #         save_txt.write(line)
    #     # elif "<name>" in line:
    #     #     line = line.replace("{}", label_name)
    #     #     save_txt.write(line)
    #     elif "xmin" in line:
    #         line = line.replace("{}", "{}".format(xmin))
    #         save_txt.write(line)
    #     elif "xmax" in line:
    #         line = line.replace("{}", "{}".format(xmax))
    #         save_txt.write(line)
    #     elif "ymin" in line:
    #         line = line.replace("{}", "{}".format(ymin))
    #         save_txt.write(line)
    #     elif "ymax" in line:
    #         line = line.replace("{}", "{}".format(ymax))
    #         save_txt.write(line)
    #     elif "" in line:
    #         pass
    #     else:
    #         save_txt.write(line)
    # save_txt.close()


def find_diff(img_list, xml_list):
    xml_list = [x.split('.')[0] for x in xml_list]
    for img in img_list:
        if img.split('.')[0] not in xml_list:
            print(img)


txt_dir = '/data/yfzx/pycharm_project/demo/pyd_project/my_ppocr/PaddleDetection/dataset/voc/VOCdevkit/VOC2007/Annotations_txt'
img_dir = '/data/yfzx/pycharm_project/demo/pyd_project/my_ppocr/PaddleDetection/dataset/voc/VOCdevkit/VOC2007/JPEGImages'
txt_list = os.listdir(txt_dir)
img_list = os.listdir(img_dir)
txt_list = sorted(txt_list)
img_list = sorted(img_list)
# print(os.path.join(img_dir, img_list[0]))
# print(os.path.join(txt_dir, txt_list[0]))

save_dir = '/data/yfzx/pycharm_project/demo/pyd_project/my_ppocr/PaddleDetection/dataset/voc/VOCdevkit/VOC2007/Annotations'
xml_list = os.listdir(save_dir)

# txt2xml('two_963.txt', os.path.join(img_dir, 'two_963.jpg'), os.path.join(txt_dir, 'two_963.txt'), save_dir)

# for txt_file, img in zip(txt_list, img_list):
#     try:
#         txt2xml(txt_file, os.path.join(img_dir, img), os.path.join(txt_dir, txt_file), save_dir)
#     except Exception as result:
#         print(img)

print(len(img_list))
print(len(xml_list))

# find_diff(img_list, xml_list)

