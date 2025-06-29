# 标注可视化
# OSDaR23数据集的遮挡现象通过此检出

import os
import cv2
import numpy as np

def visualize_yolo_annotations(image_dir, label_dir, output_dir, classes_file, thickness=2):
    """
    将YOLO格式的标注可视化到图像上
    
    参数:
    image_dir: 原始图像目录路径
    label_dir: 标签文件目录路径 
    output_dir: 输出图像保存路径
    classes_file: 包含类别名称的文本文件路径
    thickness: 边界框线条粗细
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取类别列表
    with open(classes_file) as f:
        classes = [line.strip() for line in f.readlines()]
    
    # 为每个类别生成随机颜色
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)
    processed_count = 0


    # 遍历所有图像文件
    for img_file in os.listdir(image_dir):
        if processed_count > 30:
            break

        if img_file.lower().split('.')[-1] not in ['jpg', 'jpeg', 'png']:
            continue
            
        # 构建对应标签文件路径
        base_name = os.path.splitext(img_file)[0]
        label_path = os.path.join(label_dir, base_name + '.txt')
        
        if not os.path.exists(label_path):
            continue
            
        # 读取图像
        img_path = os.path.join(image_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"无法读取图像: {img_path}")
            continue
            
        img_h, img_w = image.shape[:2]
        
        # 读取标注信息
        with open(label_path) as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
                
            # 解析YOLO格式
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_w
            y_center = float(parts[2]) * img_h
            width = float(parts[3]) * img_w
            height = float(parts[4]) * img_h
            
            # 转换为左上角坐标
            x1 = int(x_center - width/2)
            y1 = int(y_center - height/2)
            x2 = int(x_center + width/2)
            y2 = int(y_center + height/2)
            
            # 获取类别颜色和名称
            color = tuple(map(int, colors[class_id]))
            label = classes[class_id] if class_id < len(classes) else str(class_id)
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # 添加类别标签
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (x1, y1 - 20), (x1 + tw, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        # 保存结果
        processed_count += 1
        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, image)
        print(f"已处理: {img_file}")

if __name__ == "__main__":
    # 使用示例 - 请根据实际情况修改路径
    visualize_yolo_annotations(
        image_dir=".",
        label_dir=".",
        output_dir=".",
        classes_file=".",
        thickness=2
    )