import torch
import time
import psutil
import os
import platform
import numpy as np
from ultralytics import YOLO

# ---- 载入 YOLO 模型 ----
model_path = './user_data/model_pt/best.pt'  # 替换为你的模型路径
model = YOLO(model_path)
model.model.to('cpu').eval()

# ---- 生成随机输入数据 (BCHW 格式，float32, 0~1) ----
dummy_input = torch.randn(1, 3, 640, 640)

# ---- 推理性能测试 ----
print("开始 YOLO CPU 推理测试...")
with torch.no_grad():
    start = time.time()
    for _ in range(10):  # 多次推理以取平均
        _ = model.model(dummy_input)
    end = time.time()

avg_infer_time = (end - start) / 10
print(f"平均推理时间（CPU）：{avg_infer_time:.4f} 秒")

# ---- 系统资源占用信息 ----
process = psutil.Process(os.getpid())
mem_info = process.memory_info()
print(f"当前内存使用: {mem_info.rss / 1024 ** 2:.2f} MB")
print(f"CPU 核心数: {psutil.cpu_count(logical=True)}")
print(f"操作系统: {platform.platform()}")