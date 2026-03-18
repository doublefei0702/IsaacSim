#!/usr/bin/env python3
"""
Replicator to SAM3 Dataset Converter

This script converts Replicator-generated warehouse data to SAM3 format.
It creates COCO-style annotations with polygon masks for instance segmentation.

Requirements:
1. COCO format JSON with polygon annotations
2. RGB images in data/train and data/valid directories
3. Text prompts derived from cleaned class names
4. No background or ignored classes in annotations
"""

import os
import json
import shutil
import random
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Any
import glob
from pycocotools import mask as maskUtils

# Configuration
DATA_ROOT = Path("data")
TRAIN_DIR = DATA_ROOT / "train"
VALID_DIR = DATA_ROOT / "valid"

# 【修复 2】：使用地道的英文自然语义
CLASS_MAPPING = {
    "crate": "crate",
    "crate_stack": "crate stack",       # 改为地道的单词
    "person": "person",
    "rack": "storage rack",        # 增加前缀，让文本特征更明确
    "floor": "floor"
}

# 2. 忽略列表：模型不需要学习这些
IGNORE_CLASSES = ["BACKGROUND", "UNLABELLED", "wall", "ceiling","fire_extinguisher"]

# 下面的 COCO_CATEGORIES 也要同步修改！
COCO_CATEGORIES = [
    {"id": 1, "name": "crate"},
    {"id": 2, "name": "crate stack"},
    {"id": 3, "name": "person"},
    {"id": 4, "name": "storage rack"},
    {"id": 5, "name": "floor"}
]

CATEGORY_ID_MAP = {cat["name"]: cat["id"] for cat in COCO_CATEGORIES}


def init_coco_format() -> Dict[str, Any]:
    """Initialize COCO format dictionary."""
    return {
        "info": {"description": "Battery Warehouse SAM3 Dataset"},
        "images": [],
        "annotations": [],
        "categories": COCO_CATEGORIES
    }


def get_color_key(color_array: np.ndarray) -> str:
    """Convert numpy color array to string key for JSON lookup."""
    return f"[{color_array[0]} {color_array[1]} {color_array[2]} {color_array[3]}]"


def process_frame(frame_idx: int, split: str, image_id_counter: int, 
                 annotation_id_counter: int, coco_data: Dict[str, Any]) -> Tuple[int, int]:
    """Process a single frame and add to COCO data."""
    
    # File paths
    frame_str = f"{frame_idx:04d}"
    rgb_path = Path(f"rgb/rgb_{frame_str}.png")
    mask_path = Path(f"instance_segmentation/instance_segmentation_{frame_str}.png")
    mapping_path = Path(f"instance_segmentation/instance_segmentation_semantics_mapping_{frame_str}.json")
    
    # Check if all required files exist
    if not all(p.exists() for p in [rgb_path, mask_path, mapping_path]):
        print(f"Warning: Missing files for frame {frame_idx}, skipping...")
        return image_id_counter, annotation_id_counter
    
    # Read mapping JSON
    with open(mapping_path, 'r') as f:
        mapping_data = json.load(f)
    
    # Read RGB image for dimensions
    rgb_image = cv2.imread(str(rgb_path))
    if rgb_image is None:
        print(f"Warning: Could not read RGB image for frame {frame_idx}, skipping...")
        return image_id_counter, annotation_id_counter
    
    height, width = rgb_image.shape[:2]
    
    # Copy RGB image to appropriate directory
    target_dir = TRAIN_DIR if split == "train" else VALID_DIR
    target_rgb_path = target_dir / f"rgb_{frame_str}.png"
    shutil.copy2(rgb_path, target_rgb_path)
    
    # Add image entry to COCO
    image_entry = {
        "id": image_id_counter,
        "file_name": f"rgb_{frame_str}.png",
        "width": width,
        "height": height
    }
    coco_data["images"].append(image_entry)
    
    # Read instance segmentation mask
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask_image is None:
        print(f"Warning: Could not read mask image for frame {frame_idx}, skipping...")
        return image_id_counter, annotation_id_counter
    
    # 【修复 1】：将 OpenCV 默认的 BGRA 转回正常的 RGBA，使其与 JSON 对齐
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGRA2RGBA)
    
    # Process each color-class mapping
    for color_str, class_info in mapping_data.items():
        # Extract class name from the nested structure
        if isinstance(class_info, dict) and "class" in class_info:
            class_name = class_info["class"]
        else:
            class_name = class_info
        
        # Skip ignored classes
        if class_name in IGNORE_CLASSES:
            continue
        
        # Map class name using CLASS_MAPPING
        clean_class_name = CLASS_MAPPING.get(class_name, class_name)
        if clean_class_name not in CATEGORY_ID_MAP:
            print(f"Warning: Unknown class '{clean_class_name}' for frame {frame_idx}, skipping...")
            continue
        
        category_id = CATEGORY_ID_MAP[clean_class_name]
        
        # Parse color from string
        try:
            # Remove parentheses and split by comma
            color_values = [int(x.strip()) for x in color_str.strip('()').split(',')]
            color = np.array(color_values, dtype=np.uint8)
        except:
            print(f"Warning: Invalid color format '{color_str}' for frame {frame_idx}, skipping...")
            continue
        
        # Create binary mask for this color
        mask = np.all(mask_image == color, axis=2)
        
        # Skip if no pixels found for this color
        if not np.any(mask):
            continue
        
        # 【修复 6】：使用 pycocotools 生成 RLE 编码，完美处理遮挡和镂空
        # 将二维 boolean mask 转换为 uint8 类型的 Fortran 连续数组（pycocotools 硬性要求）
        fortran_mask = np.asfortranarray(mask.astype(np.uint8))
        
        # 生成 RLE 编码
        rle = maskUtils.encode(fortran_mask)
        # JSON 无法直接序列化 bytes 类型，必须解码
        rle['counts'] = rle['counts'].decode('utf-8')
        
        # 获取面积和 BBox
        area = float(maskUtils.area(rle))
        bbox = maskUtils.toBbox(rle).tolist()
        x, y, w, h = bbox
        
        # ==========================================
        # 核心过滤逻辑：双保险拦截无效碎片
        # ==========================================
        MIN_AREA = 1600      # 最小面积阈值（像素数）
        MIN_DIMENSION = 40  # 最小宽高阈值（像素长度）
        
        # 如果面积太小，或者宽高中有任意一边太窄，直接无视这个目标
        if area < MIN_AREA or w < MIN_DIMENSION or h < MIN_DIMENSION:
            continue
        # ==========================================
        
        # 生成唯一 Annotation：只生成一个 annotation_entry
        annotation_entry = {
            "id": annotation_id_counter,
            "image_id": image_id_counter,
            "category_id": category_id,
            "segmentation": rle,  # 使用 RLE 编码
            "area": area,
            "bbox": [int(x), int(y), int(w), int(h)],  # 精确的全局边界框
            "iscrowd": 0
        }
        
        coco_data["annotations"].append(annotation_entry)
        annotation_id_counter += 1  # 只自增 1 次
    
    return image_id_counter + 1, annotation_id_counter


def main():
    """Main conversion function."""
    print("Starting Replicator to SAM3 conversion...")
    
    # Initialize COCO data
    coco_train = init_coco_format()
    coco_valid = init_coco_format()
    
    # Create directories
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VALID_DIR.mkdir(parents=True, exist_ok=True)
    
    # 【修复 4】：动态获取图片数量
    rgb_files = glob.glob("rgb/rgb_*.png")
    total_frames = len(rgb_files)
    if total_frames == 0:
        print("Error: 找不到任何 rgb 图片，请确保在 output_dataset 根目录下运行！")
        return
    train_ratio = 0.8
    train_frames = int(total_frames * train_ratio)
    
    # Create frame list and shuffle for random split
    frame_indices = list(range(total_frames))
    random.shuffle(frame_indices)
    
    train_indices = frame_indices[:train_frames]
    valid_indices = frame_indices[train_frames:]
    
    print(f"Total frames: {total_frames}")
    print(f"Train frames: {len(train_indices)}")
    print(f"Valid frames: {len(valid_indices)}")
    
    # Process training frames
    print("\nProcessing training frames...")
    image_id = 1
    annotation_id = 1
    
    for i, frame_idx in enumerate(train_indices):
        if i % 10 == 0:
            print(f"  Processed {i}/{len(train_indices)} training frames")
        
        image_id, annotation_id = process_frame(
            frame_idx, "train", image_id, annotation_id, coco_train
        )
    
    # Process validation frames
    print("\nProcessing validation frames...")
    valid_image_id = 1
    valid_annotation_id = 1
    
    for i, frame_idx in enumerate(valid_indices):
        if i % 10 == 0:
            print(f"  Processed {i}/{len(valid_indices)} validation frames")
        
        valid_image_id, valid_annotation_id = process_frame(
            frame_idx, "valid", valid_image_id, valid_annotation_id, coco_valid
        )
    
    # Save COCO annotations
    print("\nSaving annotations...")
    with open(DATA_ROOT / "train" / "_annotations.coco.json", 'w') as f:
        json.dump(coco_train, f, indent=2)
    
    with open(DATA_ROOT / "valid" / "_annotations.coco.json", 'w') as f:
        json.dump(coco_valid, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"Training set: {len(coco_train['images'])} images, {len(coco_train['annotations'])} annotations")
    print(f"Validation set: {len(coco_valid['images'])} images, {len(coco_valid['annotations'])} annotations")
    print(f"RGB images copied to {TRAIN_DIR} and {VALID_DIR}")
    print(f"Annotations saved to {DATA_ROOT / 'train' / '_annotations.coco.json'}")
    print(f"and {DATA_ROOT / 'valid' / '_annotations.coco.json'}")


if __name__ == "__main__":
    main()