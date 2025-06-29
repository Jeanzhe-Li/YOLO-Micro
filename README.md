<<<<<<< HEAD
# YOLO-Micro
=======
# YOLO-Micro: Tiny Object Detection with YOLO11 Optimization

## 项目概述
本项目旨在设计一个实时小目标检测模型，该模型利用轻量化的骨干网络，对小目标检测增强优化的检测头，达到实时性的同时，提升了检测精度。该任务基于YOLO11s，优化后的YOLO-Micro，在参数减半的情况下仍能提升检测性能。消融实验进一步验证优势。同时，我们发现了全局注意力在检测中的短板：背景类似时全局依赖无法发挥作用（如GERALD数据集，每张图都是车头摄像头视角），反而增加计算量，性价比低。

## 安装

本实现基于 Ultralytics YOLO 与 Python3。要运行代码，您需要安装以下依赖：

- Ultralytics==8.3.34
- PyTorch==2.5.1
- Python==3.12(ubuntu22.04)
- torch==2.2.2 
- torchvision==0.17.2
- timm==1.0.14
- albumentations==2.0.4
- onnx==1.14.0
- onnxruntime==1.15.1
- pycocotools==2.0.7
- PyYAML==6.0.1
- scipy==1.13.0
- onnxslim==0.1.31
- onnxruntime-gpu==1.18.0
- gradio==4.44.1
- opencv-python==4.9.0.80
- psutil==5.9.8
- py-cpuinfo==9.0.0
- huggingface-hub==0.23.2
- safetensors==0.4.3
- numpy==1.26.4
- wandb==0.19.11

您可以简单地运行以下命令进行依赖安装：
```python
pip install -r requirements.txt
```


## 仓库结构

以下是本仓库的结构概述：
```python
project/
├── code/  # 存储自定义模块与需替换的模块
│   ├── RCCA.py  # ResCCA,一种结合交叉注意力与残差连接的轻量全局注意力模块，相比原版本CCA更适用于检测任务，但全局注意力不适用于本任务
│   ├── tasks.py  # 需使用此文件代替./python3.12/site-packages/ultralytics/nn/modules/tasks.py方可运行修改后的模型
├── data/  # 包含数据清洗与增强代码
│   ├── Augementation.py  # 随机剪裁与亮度调整
│   ├── DeAugementation.py  # 取消增强
│   ├── cv.py          # 可视化YOLO格式标注
│   ├── data_clean.py  # 清洗出现频率过少的类别
│   ├── data_label.py  # 将数据集标注改为标准YOLO格式
│   ├── random.py      # 随机划分训练集与测试集
│   └── data.yaml      # 示例，应包含数据集的类别与相对路径
├── prediction_result/  # 运行结果日志
│   ├── baseline.txt          # yolo11s的运行结果(nc = 60，未删除低频类别，imgsz=960)
│   ├── best.txt              # yolo-Micro的运行结果(nc = 60，未删除低频类别，imgsz=960)
│   ├── ultra_baseline.txt    # yolo11s的运行结果(nc = 51，imgsz=640)
│   ├── ultra_best.txt        # yolo-Micro的运行结果(nc = 51，imgsz=640)
│   ├── without_upsample.txt  # 消融实验：未加强上采样
│   ├── without_cbam.txt      # 消融实验：未增加注意力
│   └── model_best.pth        # 最佳结果的模型参数(yolo-Micro)
├── user_data/  # 自定义模型yaml文件
│   └── model_pt/ #保存训练完的模型
│       ├── baseline.pt   # yolo11s
│       └── best.pt       # yolo-Micro
│   └── model_data/ #模型yaml文件
│       ├── yolo11s.yaml     # baseline
│       ├── yolo11_cca.yaml  # 引入Criss-cross Attention
│       ├── Myolo11s.yaml    # yolo-Micro
│       ├── Upyolo11s.yaml   # 未加强上采样
│       └── Attyolo11s.yaml  # 未增加注意力
├── main.py  # 主训练脚本
├── test.py  # 测试脚本
├── transfer.py  # 模型权重格式转换脚本
├── cpu_test.py  # cpu上运行测试脚本
├── Aachen_Duesseldorf.mp4t=739.266667.jpg # 测试样例
├── requirements.txt  # 依赖包列表文件
└── README.md  # 本说明文件

```

## 运行流程

### 1. 数据准备
- 首先，下载数据集`GERALD`(https://publications.rwth-aachen.de/record/980030/files/GERALD.zip)
- 将数据集放置于project目录下即可。

### 2. 数据清洗
- 首先运行data_label.py将数据集转化为标准YOLO格式
- 运行random.py划分训练集与验证集（8:2）
- 运行data_clean清洗掉总出现次数少于15的类别
- 最后数据集格式为:
project/
├── yolo_ger/  # 数据集
│   └── train/ 
│       ├── images/
│       └── labels/      
│   └── valid/ 
│       ├── images/
│       └── labels/   
│   └── data.yaml  #此文件请参考示例
- 您可自由选择是否运行Augementation.py，也可以通过DeAugementation.py轻松取消增强

### 3. 模型修改
- 选择user_data/model_data/中的yaml文件中或自行编辑修改模型
- 如需使用CBAM模块，需要您使用./code/tasks.py替换ultralytics库中的./python3.12/site-packages/ultralytics/nn/modules/tasks.py

### 4. 模型训练与评估
- 您可以运行主脚本 `main.py` 来开始训练过程。您也可以自由修改超参数。
- 训练完成后，脚本将自动在验证集上评估模型，模型.pt文件与模型结果会自动保存在生成的.runs文件夹中，它还会绘制训练和验证的损失、准确率和 F1 分数曲线。

### 5. 模型测试
- 您可使用test.py来测试任意德国铁路交通标识图片，这需要你修改图片地址。

### 6. 消融实验
- 通过使用不同的模型yaml文件执行消融实验，以评估小目标优化是否必要。

### 备注：YOLO的权重保存.pt有其特有的封装格式，若需将其转为state_dict,请使用代码transfer.py

## 模型亮点

- **上采样增强**：三次上采样与拼接，在保证模型运行速度是增强对微小目标的检测能力。
- **轻量级注意力**：在验证了全局注意力不适用此任务后，使用轻量注意力模块优化对小目标的敏感度。
- **数据预处理**：包括对图像（调整大小，马赛克、小目标复制粘贴等数据增强）的全面数据预处理。
- **训练技术**：结合warm_up,SGD,early_stop等一系列适用于CNN的训练技术。

## 实验结果 

        模型        params        GFLOPs        P        R        mAP50        mAP50-95
    YOLO11s(960px)   9.4M         21.4       0.734     0.585      0.639        0.353
    YOLO11s(640px)   9.4M         21.4       0.748     0.46       0.518        0.286
  YOLO-Micro(960px)  4.6M         17.7       0.817     0.623      0.691        0.381 
  YOLO-Micro(640px)  4.6M         17.5       0.762     0.568      0.617        0.331
  NoUpsample(640px)  5.9M         13.1       0.697     0.429      0.498        0.262
  NoAttention(640px) 4.1M         17.1       0.762     0.52       0.588        0.31
 
## 模型部署

YOLO11 是一款轻量级的实时目标检测模型，在传统 CNN 架构中实现了 SOTA 性能，且支持在 GPU、CPU 以及多种边缘设备上的
灵活部署。尽管更新一代的 YOLO12 引入了更强大的AreaAttention注意力机制，具备更高的检测性能，但由于其依赖 flash-
attention 加速，对硬件有较高要求（仅支持如 T4、Quadro RTX 系列、RTX20/30/40 系列、RTX A5000/6000、A30/40、
A100、H100 等特定设备）。类似 RT-DETR 这样的 transformer 架构模型，虽然在精度上有所提升，但往往以牺牲运行速度和
模型体积为代价。

综合考虑赛事方对模型部署方面的考察与当前技术发展现状，我们选择 YOLO11 作为 baseline 模型。在此基础上，我们对网络结构
进行了优化，调整了通道配置，设计出更加轻量化的 YOLO-Micro。该模型在大幅降低了参数规模、减少浮点计算量的同时，显著提升
准确率和漏检率，针对轨道交通标志识别任务具备更高的精度与部署友好性。

以下为 YOLO-Micro 在 CPU 上的实际运行性能测试：

    ```
    开始 YOLO CPU 推理测试...
    平均推理时间（CPU）：0.2252 秒
    当前内存使用: 678.22 MB
    CPU 核心数: 8
    操作系统: macOS-14.2.1-arm64-arm-64bit
    ```
>>>>>>> 4127949 (First Submission)
