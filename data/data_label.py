# 根据GERALD数据集标注文件创建yolo格式的标注文件，并全部置于train当中


import os
import xml.etree.ElementTree as ET

def convert_annotation(xml_path, output_dir, class_dict):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 获取图像尺寸
    size = root.find('size')
    width = float(size.find('width').text)  # 转换为float类型
    height = float(size.find('height').text)  # 转换为float类型
    
    yolo_lines = []
    
    # 遍历所有object元素
    for obj in root.findall('object'):
        # 获取类别名称
        class_name = obj.find('name').text.strip()
        
        # 动态分配类别ID
        if class_name not in class_dict:
            class_dict[class_name] = len(class_dict)
        class_id = class_dict[class_name]
        
        # 提取边界框坐标（使用float处理）
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        # 转换为YOLO格式
        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        
        # 保留六位小数
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
        yolo_lines.append(yolo_line)
    
    # 写入输出文件
    if yolo_lines:
        output_path = os.path.join(output_dir, os.path.basename(xml_path).replace('.xml', '.txt'))
        with open(output_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

def main():
    # 配置路径（根据实际情况修改）
    folder_a = "../GERALD/dataset/Annotations"
    folder_b = "../GERALD/yolo_ger/train/labels"
    
    # 确保输出目录存在
    os.makedirs(folder_b, exist_ok=True)
    
    # 初始化类别字典
    class_dict = {}
    
    # 遍历所有XML文件
    for filename in os.listdir(folder_a):
        if filename.endswith('.xml'):
            xml_path = os.path.join(folder_a, filename)
            convert_annotation(xml_path, folder_b, class_dict)
    
    # 生成并保存类别映射文件（可选）
    with open(os.path.join(folder_b, 'classes.txt'), 'w') as f:
        for class_name, class_id in sorted(class_dict.items(), key=lambda x: x[1]):
            f.write(f"{class_name}\n")
    
    print(f"转换完成！共处理 {len(class_dict)} 个类别")
    print("类别映射关系：", class_dict)

if __name__ == '__main__':
    main()