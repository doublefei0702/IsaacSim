#!/usr/bin/env python3
"""
BasicWriter 数据集转换为 YOLO 格式

将 Isaac Sim BasicWriter 采集的数据转换为 YOLO 分割格式。
训练/验证集按 8:2 比例划分。

使用方法:
    python convert_to_yolo.py --data_dir /path/to/output_dataset

依赖:
    pip install opencv-python numpy
"""

import logging
import sys
from datetime import datetime

import os
import argparse
import random
import json
from pathlib import Path
from typing import Dict, List, Tuple
import shutil

import numpy as np
import cv2


# ============================================================================
# 配置区域
# ============================================================================

# 类别名称到 YOLO class_id 的映射
CLASS_NAME_TO_ID = {
    "crate": 0,
    "crate_stack": 1,
    "person": 2,
    "storage_rack": 3,
    "floor": 4,
}

# YOLO 类别名称（用于 data.yaml）
YOLO_CLASS_NAMES = {
    0: "crate",
    1: "crate_stack",
    2: "person",
    3: "rack",
    4: "floor",
}

# 忽略的类别
IGNORE_LABELS = {"BACKGROUND", "UNLABELLED", "wall", "ceiling", "fire_extinguisher", "door"}

# 过滤阈值
MIN_AREA = 6400          # 最小面积阈值（像素数）
MIN_DIMENSION = 80       # 最小宽高阈值（像素）

# 训练/验证集划分比例
TRAIN_RATIO = 0.8


# ============================================================================
# 核心转换函数
# ============================================================================

def mask_to_polygon(mask: np.ndarray) -> np.ndarray:
    """
    将 mask 转换为多边形轮廓。

    Args:
        mask: 二值掩码

    Returns:
        多边形点数组 [[x1, y1], [x2, y2], ...]
    """
    # 查找轮廓
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        logger = logging.getLogger(__name__)
        logger.debug("未找到任何轮廓")
        return None

    # 使用面积最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)

    # 简化轮廓，减少点数
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    return simplified_contour


def setup_logging(log_file: str = "convert_to_yolo.log"):
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def convert_to_yolo(
    data_dir: str,
    yolo_dir: str,
    train_ratio: float = TRAIN_RATIO
):
    """
    转换 BasicWriter 数据集为 YOLO 格式。

    Args:
        data_dir: BasicWriter 数据集目录
        yolo_dir: YOLO 输出目录
        train_ratio: 训练集比例
    """
    data_path = Path(data_dir)
    yolo_path = Path(yolo_dir)

    print("=" * 60)
    print("🔄 BasicWriter → YOLO 格式转换")
    print("=" * 60)

    # 创建输出目录结构
    (yolo_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (yolo_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_path / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # 读取 RGB 文件列表
    rgb_dir = data_path / "rgb"
    if not rgb_dir.exists():
        print(f"❌ 错误: 未找到 RGB 目录 {rgb_dir}")
        return

    rgb_files = sorted(list(rgb_dir.glob("*.jpg"))) + sorted(list(rgb_dir.glob("*.png")))
    print(f"\n📊 找到 {len(rgb_files)} 个 RGB 文件")

    # 随机划分训练/验证集
    random.seed(42)  # 固定随机种子，确保可复现
    all_indices = list(range(len(rgb_files)))
    random.shuffle(all_indices)

    train_count = int(len(all_indices) * train_ratio)
    train_indices = set(all_indices[:train_count])
    val_indices = set(all_indices[train_count:])

    print(f"\n📊 数据集划分 (8:2):")
    print(f"  - 训练集: {len(train_indices)} 张")
    print(f"  - 验证集: {len(val_indices)} 张")

    # 处理每张图像
    train_count_processed = 0
    val_count_processed = 0
    train_annotations = 0
    val_annotations = 0

    for idx, rgb_file in enumerate(rgb_files):
        # 确定数据集划分
        if idx in train_indices:
            split = "train"
            train_count_processed += 1
        elif idx in val_indices:
            split = "val"
            val_count_processed += 1
        else:
            continue

        # 读取 RGB 图像
        image = cv2.imread(str(rgb_file))
        if image is None:
            print(f"⚠️ 跳过无效图像: {rgb_file}")
            continue

        height, width = image.shape[:2]

        # 构建对应的 instance_segmentation 文件名
        instance_file = data_path / "instance_segmentation" / f"instance_segmentation_{idx:06d}.png"

        if not instance_file.exists():
            print(f"⚠️ 跳过无对应分割的图像: {rgb_file}")
            continue

        # 读取实例分割 mask
        mask_image = cv2.imread(str(instance_file), cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            print(f"⚠️ 跳过无效分割: {instance_file}")
            continue

        # 读取映射文件
        mapping_file = data_path / "instance_segmentation" / f"instance_segmentation_semantics_mapping_{idx:06d}.json"
        if not mapping_file.exists():
            print(f"⚠️ 跳过无映射的图像: {rgb_file}")
            continue

        import json
        with open(mapping_file, 'r') as f:
            mapping_data = json.load(f)

        # 解析映射数据
        id_to_labels = {}
        if isinstance(mapping_data, dict):
            id_to_labels = mapping_data
        else:
            # 兼容不同的映射格式
            for color_str, class_info in mapping_data.items():
                # 解析颜色字符串: "(R, G, B, A)"
                if color_str.startswith("("):
                    try:
                        values = [int(x.strip()) for x in color_str.strip("()").split(",")]
                        rgb_id = values[0] << 16 | values[0] << 8 | values[0]
                        g = (values[1] << 16 | values[1] << 8 | values[1]
                        b = (values[2] << 16 | values[2] << 8 | values[2]
                        a = values[3] if len(values) > 3 else 255

                        # 将 RGB 值转换为 int ID
                        if isinstance(color_str, str):
                            color_int = (rgb_id << 16) | (g << 8) | (b << 8) | a
                        else:
                            color_int = int(color_str)
                        id_to_labels[color_int] = class_info
                    else:
                        # 尝试直接解析为 int
                        try:
                            color_int = int(color_str)
                            id_to_labels[color_int] = class_info
                        except:
                            pass

        # 处理每个实例 ID
        yolo_lines = []

        for mask_id in np.unique(mask_image):
            # 跳过背景和未标签
            if mask_id == 0:
                continue

            # 获取类别信息
            label_info = id_to_labels.get(mask_id, {})
            if not label_info:
                continue

            # 解析类别名称
            if isinstance(label_info, dict):
                class_name = label_info.get("class", "")
            elif isinstance(label_info, str):
                class_name = label_info
            else:
                continue

            # 跳过忽略类别
            if class_name in IGNORE_LABELS:
                continue

            # 映射到 YOLO class_id
            yolo_class_id = CLASS_NAME_TO_ID.get(class_name)
            if yolo_class_id is None:
                continue

            # 提取二值掩码
            binary_mask = (mask_image == mask_id).astype(np.uint8)

            # 检查面积
            area = np.sum(binary_mask)
            if area < MIN_AREA:
                continue

            # 转换为多边形
            polygon = mask_to_polygon(binary_mask)
            if polygon is None:
                continue

            # 检查边界框尺寸
            x, y, w, h = cv2.boundingRect(polygon)
            if w < MIN_DIMENSION or h < MIN_DIMENSION:
                continue

            # 归一化坐标到 [0, 1]
            normalized_points = []
            for point in polygon:
                px = point[0] / width
                py = point[1] / height
                # 确保坐标在有效范围内
                px = max(0.0, min(1.0, px))
                py = max(0.0, min(1.0, py))
                normalized_points.extend([px, py])

            # 构建 YOLO 标签行: class_id x1 y1 x2 y2 ... xn yn
            label_line = f"{yolo_class_id} " + " ".join(f"{v:.6f}" for v in normalized_points)
            yolo_lines.append(label_line)

        # 写入 YOLO 标签文件
        if yolo_lines:
            # 保存图像
            img_filename = rgb_file.name
            dst_img_path = yolo_path / "images" / split / img_filename
            import shutil
            if not dst_img_path.exists():
                shutil.copy2(rgb_file, dst_img_path)

            # 保存标签
            label_filename = Path(img_filename).stem + ".txt"
            dst_label_path = yolo_path / "labels" / split / label_filename

            with open(dst_label_path, 'w') as f:
                f.write("\n".join(yolo_lines))

            # 更新标注计数
            if split == "train":
                train_annotations += len(yolo_lines)
            else:
                val_annotations += len(yolo_lines)

        # 进度显示
        if (idx + 1) % 50 == 0:
            print(f"  已处理 {idx + 1}/{len(rgb_files)} 张图像...")

    # 生成 data.yaml 配置文件
    generate_data_yaml(yolo_path)

    # 输出统计
    print("\n" + "=" * 60)
    print("✅ 转换完成！")
    print("=" * 60)
    print(f"\n📊 统计信息:")
    print(f"  训练集: {train_count_processed} 张图像, {train_annotations} 个标注")
    print(f"  验证集: {val_count_processed} 张图像, {val_annotations} 个标注")
    print(f"  总计: {train_count_processed + val_count_processed} 张图像, {train_annotations + val_annotations} 个标注")
    print(f"\n📂 输出目录: {yolo_dir}")


def generate_data_yaml(yolo_path: Path):
    """
    生成 YOLO 训练配置文件 data.yaml。
    """
    yaml_content = f"""# YOLO 分割数据集配置
# 自动生成

path: {yolo_path.absolute()}
train: images/train
val: images/val

# 类别数量
nc: {len(YOLO_CLASS_NAMES)}

# 类别名称
names:
"""
    for class_id, class_name in YOLO_CLASS_NAMES.items():
        yaml_content += f"  {class_id}: {class_name}\n"

    yaml_path_final = yolo_path / "data.yaml"
    with open(yaml_path_final, 'w', encoding='utf-8') as f:
        f.write(yaml_content)



def main():
    """主函数"""
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="BasicWriter 数据集转换为 YOLO 格式"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/root/gpufree-data/output_dataset",
        help="BasicWriter 数据集目录",
    )
    parser.add_argument(
        "--yolo_dir",
        type=str,
        default="/root/gpufree-data/output_yolo",
        help="YOLO 输出目录",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=TRAIN_RATIO,
        help="训练集比例 (0.0-1.0)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    args = parser.parse_args()

    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info(f"开始运行 convert_to_yolo.py")
    logger.info(f"参数: data_dir={args.data_dir}, yolo_dir={args.yolo_dir}, train_ratio={args.train_ratio}")

    convert_to_yolo(
        data_dir=args.data_dir,
        yolo_dir=args.yolo_dir,
        train_ratio=args.train_ratio
    )

    logger.info("程序执行成功完成")


if __name__ == "__main__":
    main()
