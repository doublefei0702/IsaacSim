#!/usr/bin/env python3
"""
YOLO 分割数据集 FiftyOne 验证工具

使用 FiftyOne 进行数据集可视化和质量检查。

安装依赖:
    pip install fiftyone

使用方法:
    python verify_yolo_fiftyone.py --data_dir /path/to/output_yolo
"""

import argparse
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F


# 类别名称映射
CLASS_NAMES = {
    0: "crate",
    1: "crate_stack",
    2: "person",
    3: "rack",
    4: "floor",
}


def verify_yolo_dataset(data_dir: str, port: int = 5151):
    """
    使用 FiftyOne 加载并可视化 YOLO 分割数据集。

    Args:
        data_dir: 数据集根目录
        port: FiftyOne 服务端口
    """
    data_path = Path(data_dir)

    # 检查目录结构
    if not (data_path / "images").exists():
        print(f"❌ 错误: 未找到 images 目录: {data_path / 'images'}")
        return

    if not (data_path / "labels").exists():
        print(f"❌ 错误: 未找到 labels 目录: {data_path / 'labels'}")
        return

    print("=" * 60)
    print("🔍 FiftyOne YOLO 分割数据集验证工具")
    print("=" * 60)

    # 构建类别名称列表（按 ID 顺序）
    max_class_id = max(CLASS_NAMES.keys())
    classes = [CLASS_NAMES.get(i, f"class_{i}") for i in range(max_class_id + 1)]

    print(f"\n📂 加载数据集: {data_dir}")
    print(f"📋 类别列表: {classes}")

    # 加载数据集
    # FiftyOne 要求 YOLO 格式有特定的目录结构
    # 我们需要分别加载 train 和 val，然后合并

    dataset = fo.Dataset("yolo_seg_dataset")

    # 加载训练集
    train_images = data_path / "images" / "train"
    train_labels = data_path / "labels" / "train"

    if train_images.exists():
        print(f"\n📥 加载训练集: {train_images}")

        # 手动构建样本
        for img_file in train_images.glob("*.[jp][pn]g"):
            label_file = train_labels / f"{img_file.stem}.txt"

            if not label_file.exists():
                print(f"  ⚠️ 缺少标签: {label_file}")
                continue

            # 创建样本
            sample = fo.Sample(filepath=str(img_file))

            # 解析 YOLO 分割标签
            detections = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 7:
                        continue

                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]

                    # 构建 FiftyOne 多边形格式
                    # coords 是归一化的 [x1, y1, x2, y2, ...]
                    points = []
                    for i in range(0, len(coords), 2):
                        points.append([coords[i], coords[i + 1]])

                    # 计算边界框 (用于显示)
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]

                    detection = fo.Detection(
                        label=CLASS_NAMES.get(class_id, f"class_{class_id}"),
                        bounding_box=bbox,
                        polygon=points,
                    )
                    detections.append(detection)

            sample["ground_truth"] = fo.Detections(detections=detections)
            sample["split"] = "train"
            dataset.add_sample(sample)

    # 加载验证集
    val_images = data_path / "images" / "val"
    val_labels = data_path / "labels" / "val"

    if val_images.exists():
        print(f"\n📥 加载验证集: {val_images}")

        for img_file in val_images.glob("*.[jp][pn]g"):
            label_file = val_labels / f"{img_file.stem}.txt"

            if not label_file.exists():
                print(f"  ⚠️ 缺少标签: {label_file}")
                continue

            sample = fo.Sample(filepath=str(img_file))

            detections = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 7:
                        continue

                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]

                    points = []
                    for i in range(0, len(coords), 2):
                        points.append([coords[i], coords[i + 1]])

                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]

                    detection = fo.Detection(
                        label=CLASS_NAMES.get(class_id, f"class_{class_id}"),
                        bounding_box=bbox,
                        polygon=points,
                    )
                    detections.append(detection)

            sample["ground_truth"] = fo.Detections(detections=detections)
            sample["split"] = "val"
            dataset.add_sample(sample)

    # 数据集统计
    print("\n" + "=" * 60)
    print("📊 数据集统计")
    print("=" * 60)

    train_count = len(dataset.match(F("split") == "train"))
    val_count = len(dataset.match(F("split") == "val"))

    print(f"\n  训练集样本数: {train_count}")
    print(f"  验证集样本数: {val_count}")
    print(f"  总样本数: {len(dataset)}")

    # 类别分布统计
    print("\n📈 类别分布:")
    class_counts = {}
    for sample in dataset:
        for det in sample.ground_truth.detections:
            label = det.label
            class_counts[label] = class_counts.get(label, 0) + 1

    for label, count in sorted(class_counts.items()):
        print(f"  {label}: {count} 个实例")

    # 启动 FiftyOne 可视化界面
    print("\n" + "=" * 60)
    print("🚀 启动 FiftyOne 可视化界面")
    print("=" * 60)
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

    args = parser.parse_args()

    verify_yolo_dataset(args.data_dir, args.port)


if __name__ == "__main__":
    main()
