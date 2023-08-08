import glob
import xml.etree.ElementTree as ET
import xml.dom.minidom
import os

xml_path = "../smoke/annotations"


# 修改标签名
def alter_label(path):
    i = 0
    for xml_file in glob.glob(path + '/*.xml'):
        # print(xml_file)
        tree = ET.parse(xml_file)
        obj_list = tree.getroot().findall('object')
        for per_obj in obj_list:
            if per_obj[0].text == '喷淋':    # 错误的标签“33”
                per_obj[0].text = 'penlin'    # 修改成“44”
                i = i+1

        tree.write(xml_file)    # 将改好的文件重新写入，会覆盖原文件
    print('共完成了{}处替换'.format(i))


def alter_path(xml_path):
    origin_path = r'{}'.format(xml_path)
    save_path = r'{}'.format(xml_path)
    xml_list = os.listdir(origin_path)
    for xmlfile in xml_list:
        xml_topic = xmlfile.replace('xml', 'jpg')
        dom = xml.dom.minidom.parse(os.path.join(origin_path, xmlfile))   # dom解析
        root = dom.documentElement   # 得到文档元素对象
        item = root.getElementsByTagName('path')   # 获取相应对象的内容
        # 修改内容
        for i in item:
            i.firstChild.data = '/suncere/pyd/yolov5-pytorch-main/VOCdevkit/VOC2007/JPEGImages/{}'.format(xml_topic)
        with open(os.path.join(save_path, xmlfile), 'w') as f:
            dom.writexml(f)
alter_path(xml_path)


def alter_folder():
    origin_path = r'./VOCdevkit/VOC2007/Annotations'
    save_path = r'./VOCdevkit/VOC2007/Annotations'
    xml_list = os.listdir(origin_path)
    for xmlfile in xml_list:
        # xml_topic = xmlfile.replace('xml', 'jpg')
        dom = xml.dom.minidom.parse(os.path.join(origin_path, xmlfile))   # dom解析
        root = dom.documentElement   # 得到文档元素对象
        item = root.getElementsByTagName('folder')   # 获取相应对象的内容
        # 修改内容
        for i in item:
            i.firstChild.data = 'JPGEImages'
        with open(os.path.join(save_path, xmlfile), 'w') as f:
            dom.writexml(f)


def alter_labelname():
    origin_path = r'./VOCdevkit/VOC2007/xmls'
    save_path = r'./VOCdevkit/VOC2007/Annotations'
    xml_list = os.listdir(origin_path)
    for xmlfile in xml_list:
        dom = xml.dom.minidom.parse(os.path.join(save_path, xmlfile))   # dom解析
        root = dom.documentElement   # 得到文档元素对象
        item = root.getElementsByTagName('name')   # 获取相应对象的内容
        # 修改内容
        for i in item:
            i.firstChild.data = 'penlin'
        with open(os.path.join(save_path, xmlfile), 'w') as f:
            dom.writexml(f)


def save_txt(path):
    imgs = os.listdir(path)
    with open("../label.txt", "w") as f:
        for img in imgs:
            f.write(path + "/" + img + " " + "1" + "\n")


