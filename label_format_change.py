import os

import cv2
import numpy as np
import xml.etree.ElementTree as ET
import pickle

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


def txt2xml():
    # xml's body
    xml_head = """<annotation>
    <folder>JPEGImages</folder>
    <filename>{}</filename>
    <path>D:\daima\yolov4-pytorch-master\VOCdevkit\VOC2007\JPEGImages\{}</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>{}</width>
        <height>{}</height>
        <depth>{}</depth>
    </size>
    <segmented>0</segmented>
    """
    xml_obj = """<object>
        <name>{}</name>
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

    # xml's labels
    xml_new = "144_new"
    label_name = "penlin"

    img = cv2.imread("VOCdevkit/VOC2007/JPEGImages/144.jpg")
    h, w = img.shape[0], img.shape[1]

    # 讲txt中的数据转换成xml需要的数据
    with open("144.txt", "r") as txt:
        for line in txt.readlines():
            line = line.strip().split(" ")
            center_x = round(float(str(line[1]).strip()) * w)
            center_y = round(float(str(line[2]).strip()) * h)
            box_w = round(float(str(line[3]).strip()) * w)
            box_h = round(float(str(line[4]).strip()) * h)

            xmin = str(int(center_x - box_w / 2))
            xmax = str(int(center_x + box_w / 2))
            ymin = str(int(center_y - box_h / 2))
            ymax = str(int(center_y + box_w / 2))

    save_xml = open("alter.txt", "w")
    save_xml.write(xml_head + xml_obj * len(open("144.txt").readlines()) + xml_end)
    save_xml.close()

    save_txt = open("{}.xml".format(xml_new), "a", encoding="utf-8")

    # while True:
    for line in open("alter.txt").readlines():
        # line = open("alter.txt", "r").readline()

        # if line == "":
        #     break
        if "filename" in line:
            line = line.replace("{}", "{}.jpg".format(xml_new))
            save_txt.write(line)
        elif "path" in line:
            line = line.replace("{}", "{}.jpg".format(xml_new))
            save_txt.write(line)
        elif "width" in line:
            line = line.replace("{}", "{}".format(w))
            save_txt.write(line)
        elif "height" in line:
            line = line.replace("{}", "{}".format(h))
            save_txt.write(line)
        elif "depth" in line:
            line = line.replace("{}", "3")
            save_txt.write(line)
        elif "<name>" in line:
            line = line.replace("{}", label_name)
            save_txt.write(line)
        elif "xmin" in line:
            line = line.replace("{}", "{}".format(xmin))
            save_txt.write(line)
        elif "xmax" in line:
            line = line.replace("{}", "{}".format(xmax))
            save_txt.write(line)
        elif "ymin" in line:
            line = line.replace("{}", "{}".format(ymin))
            save_txt.write(line)
        elif "ymax" in line:
            line = line.replace("{}", "{}".format(ymax))
            save_txt.write(line)
        elif "" in line:
            pass
        else:
            save_txt.write(line)
    save_txt.close()





