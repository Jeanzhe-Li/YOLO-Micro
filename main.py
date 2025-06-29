import torch
from ultralytics import YOLO
# import wandb

# # 初始化 wandb（设置项目、超参数和监控硬件）
# wandb.init(
#     project="your_project",  # wandb 项目名称
#     config={
#         "model": "your_model",
#         "epochs": 300,
#         "batch_size": 16,
#         "imgsz": 640,
#         "optimizer": "SGD",
#         "dataset": "./yolo_ger/data.yaml"
#     }
# )

# # 检查 GPU 信息并记录到 wandb
# device_info = {
#     "GPU_available": torch.cuda.is_available(),
#     "GPU_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
# }
# wandb.config.update(device_info)  # 将设备信息添加到配置

# 从YAML创建新模型
model = YOLO('./user_data/model_data/yolo11s.yaml')

# 训练模型
train_results = model.train(
    data="./yolo_ger/data.yaml",
    epochs=300,
    imgsz=640,
    device=0,
    batch=16,
    workers=8,
    amp=False,
    patience=60,
    optimizer="SGD",
    name="your_name",
    verbose=False,  # 关闭冗余控制台输出
    lr0=0.008,
    iou=0.5,
    box=6.0,
    cls=1.5,
    copy_paste=0.3,
    mixup=0.2,
)

# 验证模型并记录指标
metrics = model.val()
# wandb.log({
#     "val/mAP50": metrics.box.map50,
#     "val/mAP50-95": metrics.box.map,
#     "val/precision": metrics.box.p,
#     "val/recall": metrics.box.r,
# })

# # 可选：上传最佳模型权重
# best_model_path = f"{train_results.save_dir}/weights/best.pt"
# wandb.save(best_model_path)

# # 结束 wandb 运行
# wandb.finish()