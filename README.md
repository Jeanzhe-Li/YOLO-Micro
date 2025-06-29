# YOLO-Micro: Lightweight Real-time Small Object Detection

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.3.34-green.svg)](https://ultralytics.com/)

</div>

## 📋 项目简介

YOLO-Micro 是一个专门针对小目标检测优化的轻量级实时检测模型。基于 YOLO11s 架构，通过创新的网络结构优化，在**参数量减半**的情况下实现了**检测性能的显著提升**。

### 🎯 核心特性

- **轻量化设计**：参数量仅为 4.6M，相比 YOLO11s 减少 51%
- **性能提升**：mAP50 提升 8.1%，mAP50-95 提升 7.9%
- **实时检测**：CPU 推理时间仅 0.225 秒
- **部署友好**：支持 GPU、CPU 及多种边缘设备部署

### 🚀 主要创新

1. **三级上采样增强**：通过多层次特征融合，显著提升小目标检测能力
2. **轻量级注意力机制**：针对性优化小目标特征提取
3. **自适应数据增强**：包括小目标复制粘贴、马赛克增强等策略
4. **优化训练策略**：结合 warm-up、SGD、early-stop 等技术

## 🛠️ 安装指南

### 环境要求

- Python 3.12+
- PyTorch 2.2.2+
- CUDA 11.8+ (GPU 训练，可选)

### 快速安装

```bash
# 克隆仓库
git clone https://github.com/Jeanzhe-Li/YOLO-Micro.git
cd YOLO-Micro

# 安装依赖
pip install -r requirements.txt
```

### 依赖说明

主要依赖包括：
- `ultralytics==8.3.34` - YOLO 框架
- `torch==2.2.2` - 深度学习框架
- `opencv-python==4.9.0.80` - 图像处理
- `albumentations==2.0.4` - 数据增强

完整依赖列表请查看 [requirements.txt](requirements.txt)

## 📁 项目结构

```
YOLO-Micro/
├── code/                      # 核心模块
│   ├── RCCA.py               # ResCCA 注意力模块
│   └── tasks.py              # 自定义任务模块
├── data/                      # 数据处理工具
│   ├── Augmentation.py       # 数据增强
│   ├── data_clean.py         # 数据清洗
│   ├── data_label.py         # 标注格式转换
│   └── random.py             # 数据集划分
├── user_data/                 # 模型配置与权重
│   ├── model_data/           # YAML 配置文件
│   └── model_pt/             # 训练好的模型
├── prediction_result/         # 实验结果
├── main.py                   # 训练脚本
├── test.py                   # 测试脚本
└── cpu_test.py              # CPU 性能测试
```

## 🚦 使用指南

### 1. 数据准备

下载 [GERALD 数据集](https://publications.rwth-aachen.de/record/980030/files/GERALD.zip)并解压到项目根目录。

### 2. 数据预处理

```bash
# 转换标注格式
python data/data_label.py

# 划分训练/验证集 (8:2)
python data/random.py

# 清洗低频类别
python data/data_clean.py

# 可选：数据增强
python data/Augmentation.py
```

### 3. 模型训练

```bash
# 使用默认配置训练
python main.py

# 自定义配置训练
python main.py --model user_data/model_data/Myolo11s.yaml --epochs 100 --batch 16
```

### 4. 模型测试

```bash
# 测试单张图片
python test.py --image path/to/image.jpg --model user_data/model_pt/best.pt

# CPU 性能测试
python cpu_test.py
```

### 5. 模型部署

```bash
# 转换模型格式
python transfer.py --input best.pt --output model_state_dict.pth
```

## 📊 实验结果

### 性能对比

| 模型 | 参数量 | GFLOPs | Precision | Recall | mAP50 | mAP50-95 |
|------|--------|---------|-----------|---------|--------|-----------|
| **YOLO11s** (960px) | 9.4M | 21.4 | 0.734 | 0.585 | 0.639 | 0.353 |
| **YOLO-Micro** (960px) | **4.6M** | **17.7** | **0.817** | **0.623** | **0.691** | **0.381** |
| YOLO11s (640px) | 9.4M | 21.4 | 0.748 | 0.460 | 0.518 | 0.286 |
| YOLO-Micro (640px) | 4.6M | 17.5 | 0.762 | 0.568 | 0.617 | 0.331 |

### 消融实验

| 配置 | mAP50 | mAP50-95 | 说明 |
|------|--------|-----------|------|
| YOLO-Micro (完整) | 0.617 | 0.331 | - |
| w/o Upsample | 0.498 | 0.262 | 移除上采样增强 |
| w/o Attention | 0.588 | 0.310 | 移除注意力模块 |

### CPU 推理性能

```
平台：macOS-14.2.1-arm64 (8 核心)
平均推理时间：0.2252 秒
内存占用：678.22 MB
```

## 🔍 技术细节

### 网络架构优化

1. **特征金字塔增强**
   - 三级上采样结构
   - 多尺度特征融合
   - 针对小目标的特征保留

2. **注意力机制选择**
   - 实验验证：全局注意力（如 CCA）在背景相似场景下效果不佳
   - 采用轻量级局部注意力模块
   - 计算效率与检测精度的平衡

3. **训练策略**
   - Warm-up 学习率调度
   - SGD 优化器 + 动量
   - Early stopping 防止过拟合
   - 多尺度训练增强泛化能力

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 📝 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@software{yolo-micro2024,
  author = {Jeanzhe Li},
  title = {YOLO-Micro: Lightweight Real-time Small Object Detection},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Jeanzhe-Li/YOLO-Micro}
}
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - 基础框架
- [GERALD Dataset](https://publications.rwth-aachen.de/record/980030) - 数据集提供
- 所有贡献者和支持者

---

<div align="center">

**如有问题或建议，欢迎提交 [Issue](https://github.com/Jeanzhe-Li/YOLO-Micro/issues)**

</div>
