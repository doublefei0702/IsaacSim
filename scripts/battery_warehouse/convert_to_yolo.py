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
from typing import Dict, List, Tuple, Optional
import shutil
import re

import numpy as np
import cv2


# ============================================================================
# 配置区域
# ============================================================================

# YOLO 类别名称（用于 data.yaml）
YOLO_CLASS_NAMES = {
    0: "crate",
    1: "crate_stack",
    2: "person",
    3: "rack",
    4: "floor",
}

# 忽略的类别（原始路径中包含这些关键词则忽略）
IGNORE_KEYWORDS = {"BACKGROUND", "UNLABELLED", "wall", "ceiling", "fire_extinguisher", "door"}

# 过滤阈值
MIN_AREA = 1600          # 最小面积阈值（像素数）
MIN_DIMENSION = 40       # 最小宽高阈值（像素）

# 训练/验证集划分比例
TRAIN_RATIO = 0.8


# ============================================================================
# 核心转换函数
# ============================================================================

def parse_class_info(class_info) -> Optional[str]:
    """
    从映射值解析出类别名称。

    Args:
        class_info: 可以是字符串（如 "/World/Generated/Crate_new_38/..."）
                   或字典（如 {"class": "crate"}）

    Returns:
        类别名称如 "crate", "person", "rack", "floor"，或 None 如果应忽略
    """
    # 处理字典格式: {"class": "crate"}
    if isinstance(class_info, dict):
        class_name = class_info.get("class", "")
        if class_name in IGNORE_KEYWORDS:
            return None
        return class_name

    # 处理字符串格式: "/World/Generated/Crate_new_38/SM_CratePlasticNote_A_01"
    if not isinstance(class_info, str):
        return None

    upper_path = class_info.upper()

    # 忽略列表
    for ignore in IGNORE_KEYWORDS:
        if ignore.upper() in upper_path:
            return None

    # 解析类别
    if "CRATE_NEW" in upper_path:
        return "crate"
    elif "CRATE_STACK" in upper_path:
        return "crate_stack"
    elif "HUMAN_" in upper_path:
        return "person"
    elif "RACKSHELF_" in upper_path:
        return "rack"
    elif "/FLOOR" in upper_path or class_info.endswith("/floor"):
        return "floor"
    else:
        # 其他类别，返回 None 表示忽略
        logging.getLogger(__name__).warning(f"未识别的类别路径: {class_info}")
        return None


def mask_to_polygon(mask: np.ndarray) -> Optional[np.ndarray]:
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


def setup_logging(log_file: str = "convert_to_yolo.log", log_level: str = "DEBUG"):
    """设置日志配置

    Args:
        log_file: 日志文件名
        log_level: 日志级别，默认 DEBUG 以输出所有日志
    """
    level = getattr(logging, log_level.upper(), logging.DEBUG)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # 确保根 logger 级别一致
    logging.getLogger().setLevel(level)
    return logging.getLogger(__name__)


def convert_to_yolo(
    data_dir: str,
    yolo_dir: str,
    train_ratio: float = TRAIN_RATIO
):
    """
    转换 BasicWriter 数据集为 YOLO 格式。

    Args:
        data_dir: BasicWriter 数据集目录（所有文件在同一目录下）
        yolo_dir: YOLO 输出目录
        train_ratio: 训练集比例
    """
    logger = setup_logging(log_level="INFO")
    data_path = Path(data_dir)
    yolo_path = Path(yolo_dir)

    logger.info("=" * 60)
    logger.info("BasicWriter → YOLO 格式转换开始")
    logger.info("=" * 60)
    logger.info(f"输入数据目录: {data_path}")
    logger.info(f"输出 YOLO 目录: {yolo_path}")
    logger.info(f"训练集比例: {train_ratio}")

    # 创建输出目录结构
    (yolo_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (yolo_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
    logger.info("✅ 成功创建输出目录结构")

    # 读取 RGB 文件列表（直接在 data_dir 下）
    rgb_files = sorted(list(data_path.glob("rgb_*.jpg"))) + sorted(list(data_path.glob("rgb_*.png")))
    logger.info(f"📊 找到 {len(rgb_files)} 个 RGB 文件")

    if not rgb_files:
        logger.error("未找到 rgb_*.jpg 或 rgb_*.png 文件")
        raise FileNotFoundError("未找到 RGB 文件")

    # 随机划分训练/验证集
    random.seed(42)  # 固定随机种子，确保可复现
    all_indices = list(range(len(rgb_files)))
    random.shuffle(all_indices)

    train_count = int(len(all_indices) * train_ratio)
    train_indices = set(all_indices[:train_count])
    val_indices = set(all_indices[train_count:])

    logger.info(f"📊 数据集划分 (8:2):")
    logger.info(f"  - 训练集: {len(train_indices)} 张")
    logger.info(f"  - 验证集: {len(val_indices)} 张")

    # 预处理：建立文件索引字典
    # instance_segmentation_XXXX.png → 文件路径
    instance_files = {}
    for f in data_path.glob("instance_segmentation_*.png"):
        # 从文件名提取索引 XXXX
        match = re.search(r'(\d+)', f.stem)
        if match:
            idx = match.group(1)
            instance_files[idx] = f

    # mapping 文件: instance_segmentation_mapping_XXXX.json 或 instance_segmentation_semantics_mapping_XXXX.json
    mapping_files = {}
    semantics_mapping_files = {}

    for f in data_path.glob("instance_segmentation_semantics_mapping_*.json"):
        match = re.search(r'(\d+)', f.stem)
        if match:
            semantics_mapping_files[match.group(1)] = f

    for f in data_path.glob("instance_segmentation_mapping_*.json"):
        match = re.search(r'(\d+)', f.stem)
        if match:
            idx = match.group(1)
            # 优先使用 semantics_mapping
            if idx not in mapping_files:
                mapping_files[idx] = f

    # semantics_mapping 优先级更高
    for idx, f in semantics_mapping_files.items():
        mapping_files[idx] = f

    logger.info(f"📊 文件统计: {len(instance_files)} 个分割文件, {len(mapping_files)} 个映射文件")

    # 统计信息
    stats = {
        'total_masks': 0,
        'filtered_by_area': 0,
        'filtered_by_dimension': 0,
        'filtered_by_class': 0,
        'failed_conversions': 0,
        'skipped_images': 0,
        'processed_images': 0,
    }

    logger.info(f"🔄 开始处理 {len(rgb_files)} 张图像...")

    for idx, rgb_file in enumerate(rgb_files):
        # 确定数据集划分
        if idx in train_indices:
            split = "train"
        elif idx in val_indices:
            split = "val"
        else:
            continue

        # 从 rgb_0000.jpg 提取索引 0000
        match = re.search(r'(\d+)', rgb_file.stem)
        if not match:
            logger.warning(f"无法从 {rgb_file.name} 提取索引，跳过")
            stats['skipped_images'] += 1
            continue
        idx_match = match.group(1)

        logger.info(f"📷 处理图像 {idx + 1}/{len(rgb_files)}: {rgb_file.name} [{split}]")

        # 查找分割文件
        if idx_match not in instance_files:
            logger.warning(f"   ⚠️ 未找到分割文件: instance_segmentation_{idx_match}.png")
            stats['skipped_images'] += 1
            continue

        # 查找映射文件
        if idx_match not in mapping_files:
            logger.warning(f"   ⚠️ 未找到映射文件: instance_segmentation_mapping_{idx_match}.json")
            stats['skipped_images'] += 1
            continue

        # 读取 RGB 图像
        image = cv2.imread(str(rgb_file))
        if image is None:
            logger.warning(f"跳过无效图像: {rgb_file}")
            stats['skipped_images'] += 1
            continue

        height, width = image.shape[:2]

        # 读取实例分割 mask（灰度图，mask_id 是整数）
        instance_file = instance_files[idx_match]
        mask_image = cv2.imread(str(instance_file), cv2.IMREAD_UNCHANGED)
        if mask_image is None:
            logger.warning(f"跳过无效分割: {instance_file}")
            stats['skipped_images'] += 1
            continue

        # 检查 mask 和图像尺寸是否匹配
        if mask_image.shape[:2] != (height, width):
            logger.warning(f"Mask 尺寸 {mask_image.shape[:2]} 与图像尺寸 {width}x{height} 不匹配，跳过")
            stats['skipped_images'] += 1
            continue

        # 读取映射文件（格式: {int_mask_id: class_path_string}）
        mapping_file = mapping_files[idx_match]
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)

        # 处理每个实例 ID
        yolo_lines = []
        unique_mask_ids = np.unique(mask_image)
        stats['total_masks'] += len(unique_mask_ids) - 1  # 减去背景 (0)

        for mask_id in unique_mask_ids:
            if mask_id == 0:
                continue  # 跳过背景

            # 获取类别路径
            class_path = mapping_data.get(str(mask_id)) or mapping_data.get(mask_id)
            if not class_path:
                logger.debug(f"未找到 mask_id={mask_id} 的映射信息")
                continue

            # 解析类别名称
            class_name = parse_class_info(class_path)
            if class_name is None:
                stats['filtered_by_class'] += 1
                logger.debug(f"忽略类别: {class_path}")
                continue

            # 检查是否在定义的类别中
            if class_name not in YOLO_CLASS_NAMES.values():
                stats['filtered_by_class'] += 1
                logger.debug(f"未知类别: {class_name}")
                continue

            # 获取 YOLO class_id
            yolo_class_id = None
            for cid, cname in YOLO_CLASS_NAMES.items():
                if cname == class_name:
                    yolo_class_id = cid
                    break
            if yolo_class_id is None:
                continue

            # 提取二值掩码
            binary_mask = (mask_image == mask_id).astype(np.uint8)

            # 检查面积
            area = np.sum(binary_mask)
            if area < MIN_AREA:
                stats['filtered_by_area'] += 1
                logger.debug(f"过滤: mask_id={mask_id} 面积 {area} < {MIN_AREA}")
                continue

            # 转换为多边形
            polygon = mask_to_polygon(binary_mask)
            if polygon is None:
                logger.warning(f"无法为 mask_id={mask_id} 生成多边形")
                stats['failed_conversions'] += 1
                continue

            # 检查边界框尺寸
            _, _, w, h = cv2.boundingRect(polygon)
            if w < MIN_DIMENSION or h < MIN_DIMENSION:
                stats['filtered_by_dimension'] += 1
                logger.debug(f"过滤: mask_id={mask_id} 边界框 {w}x{h} < {MIN_DIMENSION}")
                continue

            # 归一化坐标到 [0, 1]
            normalized_points = []
            polygon = polygon.reshape(-1, 2)
            for point in polygon:
                px = float(point[0]) / width
                py = float(point[1]) / height
                px = max(0.0, min(1.0, px))
                py = max(0.0, min(1.0, py))
                normalized_points.extend([px, py])

            # 构建 YOLO 标签行: class_id x1 y1 x2 y2 ... xn yn
            label_line = f"{yolo_class_id} " + " ".join(f"{v:.6f}" for v in normalized_points)
            yolo_lines.append(label_line)
            logger.debug(f"   ✅ mask_id={mask_id} -> {class_name} ({len(normalized_points)//2} 个顶点)")

        # 写入 YOLO 标签文件
        if yolo_lines:
            # 保存图像
            dst_img_path = yolo_path / "images" / split / rgb_file.name
            shutil.copy2(rgb_file, dst_img_path)

            # 保存标签
            label_filename = rgb_file.stem + ".txt"
            dst_label_path = yolo_path / "labels" / split / label_filename

            with open(dst_label_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(yolo_lines))

            stats['processed_images'] += 1
            logger.info(f"   ✅ 保存: {len(yolo_lines)} 个标注")
        else:
            # 没有有效标注时，也复制图像但不创建标签文件
            dst_img_path = yolo_path / "images" / split / rgb_file.name
            shutil.copy2(rgb_file, dst_img_path)
            logger.debug(f"   ⚠️ 无有效标注，复制图像但不创建标签文件")

    # 生成 data.yaml 配置文件
    generate_data_yaml(yolo_path)

    # 输出统计
    logger.info("\n" + "=" * 60)
    logger.info("✅ 转换完成！")
    logger.info("=" * 60)
    logger.info(f"\n📊 统计信息:")
    logger.info(f"  处理图像: {stats['processed_images']} 张")
    logger.info(f"  跳过图像: {stats['skipped_images']} 张")
    logger.info(f"\n🔍 过滤统计:")
    logger.info(f"  总掩码数: {stats['total_masks']}")
    logger.info(f"  按面积过滤: {stats['filtered_by_area']}")
    logger.info(f"  按尺寸过滤: {stats['filtered_by_dimension']}")
    logger.info(f"  按类别过滤: {stats['filtered_by_class']}")
    logger.info(f"  转换失败: {stats['failed_conversions']}")
    logger.info(f"\n📂 输出目录: {yolo_dir}")


def generate_data_yaml(yolo_path: Path):
    """
    生成 YOLO 训练配置文件 data.yaml。
    """
    yaml_content = f"""# YOLO 分割数据集配置
# 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

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
    logging.getLogger(__name__).info(f"✅ 已生成 data.yaml")


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
