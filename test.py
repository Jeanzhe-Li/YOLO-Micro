import torch
from ultralytics import YOLO
import os
import glob

# 使用.pt文件加载完整模型
model = YOLO('./user_data/model_pt/baseline.pt')

# 获取文件夹中所有图片文件
image_folder = "/Users/lijingzhe/Desktop/明日青秀-齐柏林飞车-结题材料/明日青秀-齐柏林飞车-成果材料/valid/images"
image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) + \
             glob.glob(os.path.join(image_folder, "*.jpeg")) + \
             glob.glob(os.path.join(image_folder, "*.png"))

# 限制最多处理300张图片
max_images = 300
image_paths = image_paths[:max_images]

# 输出将处理的图片数量
print(f"将处理 {len(image_paths)} 张图片（最多300张）")

# 预测并保存结果
for img_path in image_paths:
    results = model.predict(
        source=img_path,
        save=True,
        project='output1',
        name='example'
    )
    # 如果需要，可以在这里对每个图片的结果进行额外处理