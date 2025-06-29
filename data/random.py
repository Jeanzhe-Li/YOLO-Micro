# 随机划分训练集与验证集，比例为8:2，保证验证集有1000张图充分验证性能
# 使用前请先改好yolo文件夹路径

import os
import random
import shutil
# 配置路径
dataset_root = '.'  # 改为你的yolo文件夹路径
train_image_dir = os.path.join(dataset_root, 'train', 'images')
train_label_dir = os.path.join(dataset_root, 'train', 'labels')
valid_image_dir = os.path.join(dataset_root, 'valid', 'images')
valid_label_dir = os.path.join(dataset_root, 'valid', 'labels')

# 创建valid文件夹（如果不存在）
os.makedirs(valid_image_dir, exist_ok=True)
os.makedirs(valid_label_dir, exist_ok=True)

# 获取所有图片文件名（不带扩展名）
image_files = [os.path.splitext(f)[0] for f in os.listdir(train_image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# 随机选择20%
random.seed(42)  # 设置随机种子保证可重复性
num_valid = int(len(image_files) * 0.2)
valid_samples = random.sample(image_files, num_valid)

# 移动文件
for sample in valid_samples:
    # 移动图片文件
    for ext in ['.jpg', '.png', '.jpeg']:  # 检查所有可能的图片格式
        src_img = os.path.join(train_image_dir, sample + ext)
        if os.path.exists(src_img):
            shutil.move(src_img, os.path.join(valid_image_dir, sample + ext))
            break
    
    # 移动标注文件
    src_label = os.path.join(train_label_dir, sample + '.txt')
    if os.path.exists(src_label):
        shutil.move(src_label, os.path.join(valid_label_dir, sample + '.txt'))

print(f'Moved {len(valid_samples)} samples to validation set')
print('Done!')