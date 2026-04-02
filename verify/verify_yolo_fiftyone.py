#!/usr/bin/env python3
"""
YOLO 分割数据集 FiftyOne 验证工具

使用 FiftyOne 进行数据集可视化和质量检查，支持 mask 检验。

安装依赖:
    pip install fiftyone opencv-python pyyaml

使用方法:
    python verify_yolo_fiftyone.py --data_dir /path/to/output_yolo
    python verify_yolo_fiftyone.py --data_dir /path/to/output_yolo --no-viz  # 仅验证不启动界面
"""

import argparse
import logging
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# 默认类别名称映射
DEFAULT_CLASS_NAMES = {
    0: "crate",
    1: "crate_stack",
    2: "person",
    3: "rack",
    4: "floor",
}


def load_class_names(data_yaml: Path) -> dict:
    """从 data.yaml 加载类别名称映射。"""
    if not data_yaml.exists():
        logger.warning(f"未找到 data.yaml ({data_yaml})，使用默认类别名称")
        return DEFAULT_CLASS_NAMES

    with open(data_yaml) as f:
        config = yaml.safe_load(f)

    names = config.get("names", {})
    if not names:
        return DEFAULT_CLASS_NAMES

    return {int(k): v for k, v in names.items()}


def parse_yolo_polygon(line: str) -> Optional[tuple]:
    """
    解析 YOLO 分割标签的一行。

    Returns:
        (class_id, points) 其中 points 是归一化坐标的 [(x, y), ...] 列表
    """
    parts = line.strip().split()
    if len(parts) < 7:  # 至少 1 个 class_id + 3 个点 (6 坐标)
        return None

    class_id = int(parts[0])
    coords = [float(x) for x in parts[1:]]

    if len(coords) % 2 != 0:
        return None

    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    return class_id, points


def rasterize_polygons_to_mask(
    polygons: list, class_ids: list, height: int, width: int, ignore_index: int = 255
) -> np.ndarray:
    """
    将多边形光栅化为 mask。

    Args:
        polygons: 多边形点列表 [[(x,y), ...], ...]
        class_ids: 每个多边形对应的类别 ID 列表
        height: 图像高度
        width: 图像宽度
        ignore_index: 背景填充值

    Returns:
        HxW numpy array，像素值为类别 ID
    """
    mask = np.full((height, width), ignore_index, dtype=np.uint8)

    for pts, cls_id in zip(polygons, class_ids):
        # 转换为整数坐标（像素位置）
        pts_int = np.array(pts, dtype=np.int32)
        # 调整坐标到像素空间（YOLO 是归一化的 x,y，mask 需要 col,row 即 x,y）
        pts_int[:, 0] = np.clip(pts_int[:, 0] * width, 0, width - 1)
        pts_int[:, 1] = np.clip(pts_int[:, 1] * height, 0, height - 1)

        # 填充多边形
        cv2.fillPoly(mask, [pts_int], int(cls_id))

    return mask


def compute_bbox(points: list) -> list:
    """计算归一化边界框 [x, y, w, h]."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def validate_polygon(points: list, min_area: float = 1e-6) -> tuple:
    """
    验证多边形有效性（使用 OpenCV）。

    Returns:
        (is_valid, error_msg, area)
    """
    if len(points) < 3:
        return False, "点数少于 3 个", 0.0

    # 转换为 numpy 数组并确保是 float 类型
    pts = np.array(points, dtype=np.float32)

    # 检查是否有重复点（会导致面积为 0）
    if len(pts) != len(np.unique(pts, axis=0)):
        return False, "包含重复点", 0.0

    # 使用 OpenCV 计算面积和轮廓
    area = cv2.contourArea(pts)

    if abs(area) < min_area:
        return False, f"面积过小: {abs(area):.2e}", abs(area)

    # 检查边界框有效性
    x, y = pts[:, 0], pts[:, 1]
    w, h = x.max() - x.min(), y.max() - y.min()
    if w <= 0 or h <= 0:
        return False, "边界框无效", abs(area)

    # 检查自相交（简化检查：凸包面积应接近原始面积）
    hull = cv2.convexHull(pts)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0 and area / hull_area < 0.1:
        return False, f"可能自相交 (凸包比={area/hull_area:.2f})", abs(area)

    return True, "", abs(area)


def process_single_image(args: tuple) -> dict:
    """
    处理单个图像文件（供并行调用）。

    Returns:
        dict with keys: img_path, label_path, split, detections, errors
    """
    img_path, label_path, class_names, split = args
    result = {
        "img_path": str(img_path),
        "label_path": str(label_path),
        "split": split,
        "detections": [],
        "errors": [],
    }

    if not label_path.exists():
        result["errors"].append(f"缺少标签文件: {label_path.name}")
        return result

    try:
        with open(label_path) as f:
            lines = f.readlines()
    except Exception as e:
        result["errors"].append(f"读取标签文件失败: {e}")
        return result

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parsed = parse_yolo_polygon(line)
        if parsed is None:
            result["errors"].append(f"{label_path.name}:{line_num} - 解析失败或点数不足")
            continue

        class_id, points = parsed
        label_name = class_names.get(class_id, f"class_{class_id}")

        is_valid, error_msg, _ = validate_polygon(points)
        if not is_valid:
            result["errors"].append(f"{label_path.name}:{line_num} [{label_name}] - {error_msg}")
            continue

        bbox = compute_bbox(points)
        result["detections"].append({
            "label": label_name,
            "class_id": class_id,
            "bbox": bbox,
            "polygon": points,
        })

    return result


def verify_yolo_dataset(
    data_dir: str,
    port: int = 5151,
    skip_viz: bool = False,
    num_workers: int = None,
    min_polygon_area: float = 1e-6,
):
    """
    使用 FiftyOne 加载并可视化 YOLO 分割数据集，同时进行 mask/polygon 验证。

    Args:
        data_dir: 数据集根目录
        port: FiftyOne 服务端口
        skip_viz: 跳过可视化，仅进行验证
        num_workers: 并行处理的工作进程数，None=自动
        min_polygon_area: 多边形最小面积阈值
    """
    data_path = Path(data_dir)

    # 检查目录结构
    if not (data_path / "images").exists():
        logger.error(f"未找到 images 目录: {data_path / 'images'}")
        return

    if not (data_path / "labels").exists():
        logger.error(f"未找到 labels 目录: {data_path / 'labels'}")
        return

    print("=" * 60)
    print("FiftyOne YOLO 分割数据集验证工具")
    print("=" * 60)

    # 加载类别名称
    class_names = load_class_names(data_path / "data.yaml")

    print(f"\n📂 数据集: {data_dir}")
    print(f"📋 类别: {list(class_names.values())}")

    # 并行处理参数
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    # 收集所有图像文件
    tasks = []
    for split in ("train", "val"):
        img_dir = data_path / "images" / split
        lbl_dir = data_path / "labels" / split

        if not img_dir.exists():
            continue

        for img_file in img_dir.glob("*.[jp][pn]g"):
            label_file = lbl_dir / f"{img_file.stem}.txt"
            tasks.append((img_file, label_file, class_names, split))

    if not tasks:
        logger.error("未找到任何图像文件")
        return

    print(f"\n🔄 找到 {len(tasks)} 个图像文件，使用 {num_workers} 个工作进程")
    print(f"📐 最小多边形面积阈值: {min_polygon_area:.2e}")

    # 并行处理
    results = []
    with mp.Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap(process_single_image, tasks), 1):
            results.append(result)
            if i % 50 == 0 or i == len(tasks):
                print(f"  处理进度: {i}/{len(tasks)}", end="\r")

    print()  # 换行

    # 汇总统计
    total_errors = sum(len(r["errors"]) for r in results)
    total_detections = sum(len(r["detections"]) for r in results)

    print("\n" + "=" * 60)
    print("Mask/多边形验证结果")
    print("=" * 60)

    # 按错误类型分组
    error_types = defaultdict(list)
    for r in results:
        for err in r["errors"]:
            error_type = err.split(" - ")[-1] if " - " in err else err
            error_types[error_type].append(err)

    if error_types:
        print("\n❌ 发现以下问题:")
        for error_type, errors in sorted(error_types.items(), key=lambda x: -len(x[1])):
            print(f"\n  [{error_type}] ({len(errors)} 处)")
            for err in errors[:5]:
                print(f"    - {err}")
            if len(errors) > 5:
                print(f"    ... 还有 {len(errors) - 5} 处")
    else:
        print("\n✅ 所有多边形验证通过")

    # 统计信息
    print("\n" + "=" * 60)
    print("📊 数据集统计")
    print("=" * 60)

    split_counts = defaultdict(int)
    class_counts = defaultdict(int)

    for r in results:
        split_counts[r["split"]] += 1
        for det in r["detections"]:
            class_counts[det["label"]] += 1

    print(f"\n  训练集样本数: {split_counts.get('train', 0)}")
    print(f"  验证集样本数: {split_counts.get('val', 0)}")
    print(f"  总样本数: {len(results)}")
    print(f"  总实例数: {total_detections}")
    print(f"  错误数: {total_errors}")

    print("\n📈 类别分布:")
    for label, count in sorted(class_counts.items()):
        print(f"  {label}: {count} 个实例")

    if skip_viz:
        print("\n✅ 验证完成 (skip viz mode)")
        return

    # 尝试导入 FiftyOne
    try:
        import fiftyone as fo
        from fiftyone import ViewField as F
    except Exception as e:
        logger.error(f"无法导入 FiftyOne: {e}")
        logger.error("请使用 --no-viz 选项仅进行验证")
        return

    # 构建 FiftyOne 数据集
    print("\n" + "=" * 60)
    print("🚀 构建 FiftyOne 数据集")
    print("=" * 60)

    dataset_name = "yolo_seg_dataset"
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name)

    for r in results:
        sample = fo.Sample(filepath=r["img_path"])

        # 读取图像获取尺寸
        img = cv2.imread(r["img_path"])
        if img is None:
            logger.warning(f"无法读取图像: {r['img_path']}")
            continue
        h, w = img.shape[:2]

        # 构建 detections（用于边界框和 polygon 标注）
        detections = []
        polygons = []
        class_ids_for_mask = []

        for det in r["detections"]:
            detection = fo.Detection(
                label=det["label"],
                bounding_box=det["bbox"],
                polygon=det["polygon"],
            )
            detections.append(detection)
            polygons.append(det["polygon"])
            class_ids_for_mask.append(det["class_id"])

        sample["ground_truth"] = fo.Detections(detections=detections)
        sample["split"] = r["split"]

        # 光栅化为 mask（单通道，值为类别 ID）
        if polygons:
            mask = rasterize_polygons_to_mask(
                polygons, class_ids_for_mask, h, w, ignore_index=255
            )
            sample["mask"] = fo.Segmentation(mask=mask)

        if r["errors"]:
            sample["has_errors"] = True
            sample["error_summary"] = "; ".join(r["errors"][:3])
        else:
            sample["has_errors"] = False

        dataset.add_sample(sample)

    # 启动 FiftyOne 可视化
    print(f"\n📌 浏览器访问: http://localhost:{port}")
    print("📌 按 Ctrl+C 退出\n")

    session = fo.launch_app(dataset, port=port)
    session.wait()


def main():
    parser = argparse.ArgumentParser(
        description="FiftyOne YOLO 分割数据集验证工具"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="output_yolo",
        help="YOLO 数据集根目录",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5151,
        help="FiftyOne 服务端口",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="仅进行 mask 验证，不启动可视化界面",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行工作进程数 (默认: CPU 核数 - 1)",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=1e-6,
        help="多边形最小面积阈值 (默认: 1e-6)",
    )

    args = parser.parse_args()

    verify_yolo_dataset(
        args.data_dir,
        port=args.port,
        skip_viz=args.no_viz,
        num_workers=args.workers,
        min_polygon_area=args.min_area,
    )


if __name__ == "__main__":
    main()