import xml.etree.ElementTree as ET

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

tree = ET.fromstring(xml_head + xml_end)
root = ET.ElementTree(tree)
root.write("a.xml")