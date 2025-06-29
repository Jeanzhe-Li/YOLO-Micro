# 去掉出现总次数少于15的类别，并删除标注，更新类别号，更新data.yaml文件
# 同一目录下的data.yaml文件是最终版本


import os
import yaml
import numpy as np

# === 路径配置 ===
label_dirs = ['..yolo_ger/train/labels', '..yolo_ger/valid/labels']
image_dirs = ['..yolo_ger/train/images', '..yolo_ger/valid/images']
yaml_path = '..yolo_ger/data.yaml'  # 你的 data.yaml 路径
min_freq = 15                    # 小于这个阈值的类别会被删除

# === 读取 data.yaml ===
with open(yaml_path, 'r') as f:
    data_yaml = yaml.safe_load(f)

original_names = data_yaml['names']
num_classes = len(original_names)

# === 统计类别频次 ===
cls_freq = np.zeros(num_classes, dtype=int)

for label_dir in label_dirs:
    for file in os.listdir(label_dir):
        if file.endswith('.txt'):
            with open(os.path.join(label_dir, file), 'r') as f:
                for line in f:
                    if line.strip():
                        cls = int(line.split()[0])
                        cls_freq[cls] += 1

# === 找出需要删除的类别 ===
rare_classes = set(np.where(cls_freq < min_freq)[0])
print(f"稀有类别（将被删除）: {sorted(rare_classes)}")

# === 构造新类别映射表 ===
old_to_new = {}
new_names = []
new_cls_id = 0
for old_id in range(num_classes):
    if old_id not in rare_classes:
        old_to_new[old_id] = new_cls_id
        new_names.append(original_names[old_id])
        new_cls_id += 1

print(f"类别编号映射表: {old_to_new}")

# === 更新标注文件并删除无效图像 ===
for label_dir, image_dir in zip(label_dirs, image_dirs):
    for file in os.listdir(label_dir):
        if not file.endswith('.txt'):
            continue

        label_path = os.path.join(label_dir, file)
        with open(label_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            if cls_id in old_to_new:
                parts[0] = str(old_to_new[cls_id])
                new_lines.append(' '.join(parts) + '\n')

        if not new_lines:
            # 删除标签文件和图像文件
            os.remove(label_path)
            image_name = file.replace('.txt', '.jpg')  # or .png
            image_path = os.path.join(image_dir, image_name)
            if os.path.exists(image_path):
                os.remove(image_path)
            print(f"已删除空标签与图像: {file}")
        else:
            with open(label_path, 'w') as f:
                f.writelines(new_lines)

# === 更新 data.yaml ===
data_yaml['names'] = new_names
data_yaml['nc'] = len(new_names)

with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml)

print("data.yaml 已更新")
print(f"现在类别总数: {len(new_names)}，类别名称: {new_names}")