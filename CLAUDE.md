# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个 **Nvidia Isaac Sim** 自动化合成数据生成 (Synthetic Data Generation, SDG) 与 3D 重建项目，主要用于物流仓储场景的机器人感知训练。

### 核心功能

1. **合成数据生成 (SDG)**: 使用 Isaac Sim 的 Replicator 进行域随机化，生成用于训练的图像数据
2. **COCO 格式转换**: 将 Replicator 输出的实例分割数据转换为 SAM3 模型可用的 COCO 格式
3. **3D 重建数据采集**: 为 Nova Carter 机器人规划路径，采集双目视觉数据

## 目录结构

```
isaacsim/
├── CLAUDE.md
├── scenes/                         # 所有 USD 场景资产
│   ├── battery_warehouse/          # 工厂仓库合成数据场景
│   │   ├── scene_01_completed.usd  # 完整场景 (含货箱/码垛/语义)
│   │   ├── scene_01_6shelfves.usd # 6货架场景
│   │   ├── scene_01_complete.usd
│   │   └── sacene_plasticCrates.usd
│   └── my_warehouse/              # 3DGS 建图用仓库场景
│       └── my_warehouse_for_3DReconstrcion.usd
├── scripts/                        # 所有脚本统一管理
│   ├── battery_warehouse/         # 工厂仓库 SDG 脚本
│   │   ├── generate_dataset.py     # 主数据采集 (Replicator 域随机化)
│   │   ├── convert_to_sam3.py     # COCO 格式转换
│   │   ├── setup_scene_v2_cline.py # 场景初始化 (货箱/码垛/语义)
│   │   └── setup_scene_test.py    # 测试用场景设置
│   └── 3DGS/                      # 3DGS 建图数据采集脚本
│       ├── replicator_final.py     # 机器人路径规划与数据采集
│       ├── batch_extract_params.py # 批量提取参数
│       ├── check_params.py        # 参数检查
│       └── Get_ply.py             # 点云获取
├── verify/                        # 数据验证脚本
│   ├── verify_transforms.py
│   └── verify_mask.py
└── output_dataset/                 # 采集输出目录 (运行时生成)
    ├── rgb/                       # RGB 图像
    ├── instance_segmentation/     # 实例分割 mask + mapping JSON
    ├── semantic_segmentation/     # 语义分割数据
    └── bounding_box_2d_tight/    # 2D 边界框
```

## 运行方式

### 1. 数据采集 (需要 Isaac Sim 环境)

```bash
# 工厂仓库合成数据生成
python scripts/battery_warehouse/generate_dataset.py

# 3DGS 建图数据采集
python scripts/3DGS/replicator_final.py
```

### 2. COCO 格式转换 (可在本地运行)

```bash
cd output_dataset
python ../scripts/battery_warehouse/convert_to_sam3.py
```

转换输出到 `data/train/` 和 `data/valid/` 目录，包含:
- RGB 图像
- `_annotations.coco.json` - COCO 格式标注文件

### 3. 场景设置与调试

```bash
# 场景初始化 (生成货箱、码垛、语义标签)
python scripts/battery_warehouse/setup_scene_v2_cline.py
```

### 4. 数据验证

```bash
# 验证采集的姿态轨迹
python verify/verify_transforms.py

# 验证 mask 是否正确
python verify/verify_mask.py
```

## 数据流程

```
scenes/battery_warehouse/scene_01_completed.usd
    ↓
scripts/battery_warehouse/setup_scene_v2_cline.py (场景填充 + 语义打标)
    ↓
scripts/battery_warehouse/generate_dataset.py (Replicator 域随机化)
    ↓
output_dataset/ (rgb, instance_segmentation, semantic_segmentation, bbox)
    ↓
scripts/battery_warehouse/convert_to_sam3.py (COCO 格式转换)
    ↓
data/train/_annotations.coco.json + rgb images
    ↓
SAM3 模型训练
```

## 关键参数配置

### 域随机化 (generate_dataset.py)

- 相机位姿: `position=(-8.0~-10.0, 1.3~2.7)`, `look_at=(-5.0~5.0, 0.0~2.0)`
- 光照强度: `400~1200` (Dome Light)
- 采集数量: `num_images_to_generate = 200`

### 码垛生成 (setup_scene_v2_cline.py)

- 区域 1: `x=(-10,9), y=(-11,-4.5)`
- 区域 2: `x=(-10,9), y=(12.6,16)`
- 层数: `3~5层`, 列数: `2~4`, 行数: `2~3`
- 货箱间距: `X=0.02, Y=0.02, Z=0.5`

### 语义标签类别

| 标签 | 说明 |
|------|------|
| `crate` | 散乱货箱 |
| `crate_stack` | 码垛整体 |
| `person` | 人体 |
| `rack` | 货架 |
| `floor` | 地面 |
| `wall/ceiling/door` | 忽略类别 |

### COCO 转换过滤条件

- 最小面积: `1600` 像素
- 最小宽/高: `40` 像素

## 机器人路径规划 (replicator_final.py)

路径使用 `SEQUENCE_CONFIG` 配置，支持三种指令:
- `LINE_ABS`: 绝对坐标直线 `(start=(x,y), end=(x,y), frames=int)`
- `TURN`: 圆弧转弯 `(angle=度数, radius=米, frames=int)`
- `LINE_FWD`: 相对直线 `(distance=米, frames=int)`

机器人配置:
- 路径: `/World/my_warehouse_1_new/my_warehouse_1/nova_carter`
- 左相机: `/World/.../nova_carter/chassis_link/sensors/front_hawk/left/camera_left`
- 右相机: `/World/.../nova_carter/chassis_link/sensors/front_hawk/right/camera_right`

## 技术栈

- **仿真平台**: Nvidia Isaac Sim 5.0
- **数据生成**: Omni.Replicator
- **USD 场景**: Pixar USD 格式
- **标注格式**: COCO (via pycocotools)
- **路径规划**: 自定义 TrajectoryBuilder 类

## 注意事项

1. 这些脚本设计为在 **Isaac Sim 运行环境**中运行，非标准 Python 环境
2. USD 资产路径硬编码为 `/root/gpufree-data/...`，需要根据实际部署环境修改
3. `convert_to_sam3.py` 依赖 `pycocotools`，需单独安装: `pip install pycocotools opencv-python`
