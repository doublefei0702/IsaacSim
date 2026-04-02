# YOLO 分割数据集采集与验证指南

本文档介绍如何使用 Isaac Sim 生成 YOLO 分割格式的数据集，以及使用 FiftyOne 进行数据集验证。

## 目录

- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [数据集结构](#数据集结构)
- [验证工具](#验证工具)
- [常见问题](#常见问题)

---

## 环境要求

### Isaac Sim 环境

数据采集脚本需要在 Nvidia Isaac Sim 环境中运行。

- Isaac Sim 5.0+
- Python 3.10+
- CUDA 11.x+

### 验证工具环境（本地机器）

验证脚本可以在本地机器运行，无需 Isaac Sim。

```bash
# 安装 FiftyOne
pip install fiftyone

# 安装 OpenCV（可选，用于图像处理）
pip install opencv-python
```

---

## 快速开始

### 1. 数据采集

在 Isaac Sim 服务器上执行以下命令：

```bash
# 进入 Isaac Sim Python 环境
cd /root/gpufree-data/IsaacSim

# 运行数据采集脚本
./python.sh /root/gpufree-data/IsaacSim/scripts/battery_warehouse/generate_yolo_dataset.py
```

**预计耗时**: 200 张图片约 10-20 分钟（取决于 GPU 性能）

### 2. 数据集验证

将生成的数据集下载到本地后，使用 FiftyOne 进行验证：

```bash
# 安装 FiftyOne（首次使用）
pip install fiftyone

# 运行验证脚本
python scripts/battery_warehouse/verify_yolo_fiftyone.py --data_dir output_yolo
```

验证界面将在浏览器中自动打开：`http://localhost:5151`

---

## 数据集结构

生成的数据集遵循 YOLO 分割格式：

```
output_yolo/
├── images/
│   ├── train/           # 训练集图像
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   └── ...
│   └── val/             # 验证集图像
│       ├── 000003.jpg
│       └── ...
├── labels/
│   ├── train/           # 训练集标签
│   │   ├── 000000.txt
│   │   ├── 000001.txt
│   │   └── ...
│   └── val/             # 验证集标签
│       ├── 000003.txt
│       └── ...
└── data.yaml            # YOLO 训练配置文件
```

### 标签格式

每个 `.txt` 文件包含多行标注，每行对应一个实例：

```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

- `class_id`: 类别 ID（0-4）
- `x1 y1 ... xn yn`: 归一化的多边形顶点坐标（范围 0-1）

### 类别定义

| Class ID | 类别名称 | 描述 |
|----------|----------|------|
| 0 | crate | 散乱货箱 |
| 1 | crate_stack | 码垛整体 |
| 2 | person | 人体 |
| 3 | rack | 货架 |
| 4 | floor | 地面 |

### 配置文件示例

`data.yaml` 文件内容：

```yaml
path: /path/to/output_yolo
train: images/train
val: images/val

nc: 5  # 类别数量

names:
  0: crate
  1: crate_stack
  2: person
  3: rack
  4: floor
```

---

## 验证工具

### FiftyOne 可视化

FiftyOne 是一个强大的数据集可视化工具，支持交互式浏览、过滤和分析。

**功能**：
- 浏览所有图像及其标注
- 按类别过滤实例
- 检查标注质量
- 统计类别分布

**使用方法**：

```bash
# 基本用法
python verify_yolo_fiftyone.py --data_dir output_yolo

# 指定端口
python verify_yolo_fiftyone.py --data_dir output_yolo --port 8080
```

**FiftyOne 界面操作**：

1. **浏览样本**: 左侧面板显示所有样本缩略图
2. **查看标注**: 点击样本查看详细标注
3. **过滤类别**: 使用过滤面板选择特定类别
4. **统计数据**: 查看"Statistics"标签页了解分布

### 在线文档

- [FiftyOne 官方文档](https://docs.voxel51.com/)
- [YOLO 格式说明](https://docs.ultralytics.com/datasets/segment/)

---

## YOLO 训练

数据集生成后，可直接用于 YOLO 模型训练：

```bash
# 安装 Ultralytics
pip install ultralytics

# 开始训练
yolo segment train \
    data=output_yolo/data.yaml \
    model=yolov8n-seg.pt \
    epochs=100 \
    imgsz=640 \
    batch=16
```

**推荐配置**：

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `model` | yolov8n-seg.pt | nano 模型（快速验证）|
| `model` | yolov8s-seg.pt | small 模型（推荐）|
| `epochs` | 100-300 | 训练轮次 |
| `imgsz` | 640 | 输入图像尺寸 |
| `batch` | 16 | 批量大小（根据 GPU 调整）|

---

## 常见问题

### Q1: 数据采集时内存不足

**解决方案**: 减少采集帧数或降低图像分辨率：

```python
# 修改 generate_yolo_dataset.py
num_images_to_generate = 100  # 从 200 减少到 100
render_product = rep.create.render_product(camera, (512, 512))  # 从 1024 改为 512
```

### Q2: FiftyOne 无法启动

**检查端口占用**：

```bash
# Linux/macOS
lsof -i :5151

# Windows
netstat -ano | findstr 5151
```

**使用其他端口**：

```bash
python verify_yolo_fiftyone.py --data_dir output_yolo --port 8080
```

### Q3: 标注文件为空

**可能原因**：
1. 场景中缺少语义标签
2. 所有实例都被过滤（面积/尺寸过小）

**检查方法**：

```bash
# 查看标签文件
cat output_yolo/labels/train/000000.txt

# 如果文件为空，检查场景语义配置
```

### Q4: 训练时报错 "No labels found"

**解决方案**: 检查 `data.yaml` 中的路径是否正确：

```yaml
# 使用绝对路径
path: /absolute/path/to/output_yolo
```

---

## 脚本参数

### generate_yolo_dataset.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `scene_path` | /root/.../scene_01_completed.usd | USD 场景路径 |
| `output_dir` | /root/.../output_yolo | 输出目录 |
| `num_images_to_generate` | 200 | 采集帧数 |
| `TRAIN_RATIO` | 0.8 | 训练集比例 |
| `MIN_AREA` | 1600 | 最小实例面积 |
| `MIN_DIMENSION` | 40 | 最小实例宽高 |

### verify_yolo_fiftyone.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | output_yolo | 数据集目录 |
| `--port` | 5151 | FiftyOne 端口 |

---

## 联系与支持

如有问题，请检查：
1. Isaac Sim 日志输出
2. FiftyOne 控制台信息
3. 数据集目录结构是否正确

---

**最后更新**: 2026-04-02
