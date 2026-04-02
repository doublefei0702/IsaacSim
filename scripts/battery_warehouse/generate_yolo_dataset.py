#!/usr/bin/env python3
"""
YOLO 分割格式数据集生成脚本

直接输出 YOLO 分割模型训练所需的数据格式：
- images/train/*.jpg / images/val/*.jpg
- labels/train/*.txt / labels/val/*.txt

标签格式：class_id x1 y1 x2 y2 ... xn yn (归一化坐标)
"""

import os
import time
import json
import random
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

print("[CheckPoint 1] 🚀 开始启动 SimulationApp (无头模式)...")
t_start = time.time()
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
print(f"[CheckPoint 2] ✅ SimulationApp 启动完成！耗时: {time.time() - t_start:.2f} 秒")

print("[CheckPoint 3] 📦 开始导入底层模块...")
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.utils.semantics import add_labels
print("[CheckPoint 4] ✅ 模块导入完毕。")


# ============================================================================
# 配置区域
# ============================================================================

# 类别映射：语义标签 -> YOLO class_id
CLASS_MAPPING = {
    "crate": 0,
    "crate_stack": 1,
    "person": 2,
    "rack": 3,
    "floor": 4,
}

# 忽略的类别（不写入标注）
IGNORE_CLASSES = {"BACKGROUND", "UNLABELLED", "wall", "ceiling", "fire_extinguisher"}

# 过滤阈值
MIN_AREA = 1600          # 最小面积阈值（像素数）
MIN_DIMENSION = 60       # 最小宽高阈值（像素）

# 训练/验证集划分比例
TRAIN_RATIO = 0.8


# ============================================================================
# 自定义 YOLO 分割 Writer
# ============================================================================

class YOLOSegWriter(rep.Writer):
    """
    自定义 Writer，直接输出 YOLO 分割格式。

    输出目录结构：
    output_dir/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    """

    def __init__(self):
        """
        初始化 Writer，指定需要的 annotators。
        """
        # 必须指定需要的 annotators
        self.annotators = ["rgb", "instance_segmentation_fast"]

        # Writer 版本
        self.version = "1.0.0"

        # 初始化内部状态
        self._output_dir: str = ""
        self._frame_id: int = 0
        self._train_indices: List[int] = []
        self._valid_indices: List[int] = []
        self._total_frames: int = 0
        self._initialized: bool = False

        # 缓存数据，等待最后写入
        self._cached_data: List[Tuple[int, str, np.ndarray, Dict]] = []

        # 日志文件
        self._log_file = None
        self._write_call_count = 0

    def _log(self, msg: str):
        """
        输出日志到控制台和文件。
        """
        print(msg)
        if self._log_file:
            self._log_file.write(msg + "\n")
            self._log_file.flush()

    def initialize(self, **kwargs):
        """
        初始化 Writer 参数。

        Args:
            output_dir: 输出目录路径
            total_frames: 总帧数（用于划分训练/验证集）
            train_ratio: 训练集比例（默认 0.8）
        """
        self._output_dir = kwargs.get("output_dir", "./output_yolo")
        self._total_frames = kwargs.get("total_frames", 100)
        train_ratio = kwargs.get("train_ratio", TRAIN_RATIO)

        # 创建输出目录结构
        os.makedirs(os.path.join(self._output_dir, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(self._output_dir, "images", "val"), exist_ok=True)
        os.makedirs(os.path.join(self._output_dir, "labels", "train"), exist_ok=True)
        os.makedirs(os.path.join(self._output_dir, "labels", "val"), exist_ok=True)

        # 创建日志文件
        log_path = os.path.join(self._output_dir, "writer_debug.log")
        self._log_file = open(log_path, 'w', encoding='utf-8')

        # 生成训练/验证集索引
        all_indices = list(range(self._total_frames))
        random.shuffle(all_indices)
        train_count = int(self._total_frames * train_ratio)
        self._train_indices = set(all_indices[:train_count])
        self._valid_indices = set(all_indices[train_count:])

        self._frame_id = 0
        self._initialized = True
        self._cached_data = []
        self._write_call_count = 0

        self._log(f"[YOLOSegWriter] 初始化完成:")
        self._log(f"  - 输出目录: {self._output_dir}")
        self._log(f"  - 总帧数: {self._total_frames}")
        self._log(f"  - 训练集: {len(self._train_indices)} 帧")
        self._log(f"  - 验证集: {len(self._valid_indices)} 帧")
        self._log(f"  - 日志文件: {log_path}")

    def write(self, data: dict):
        """
        每帧调用一次，处理 RGB 和实例分割数据。

        Args:
            data: 包含 annotator 数据的字典
                  格式: {"rgb": {"data": np.ndarray},
                         "instance_segmentation_fast": {"data": np.ndarray, "info": dict}}
        """
        self._write_call_count += 1

        # 首次调用时输出 data 结构
        if self._write_call_count == 1:
            self._log(f"\n[DEBUG] write() 第 1 次调用")
            self._log(f"[DEBUG] data 类型: {type(data)}")
            self._log(f"[DEBUG] data 顶层键: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")

            # 详细输出 data 结构
            if isinstance(data, dict):
                for key, value in data.items():
                    self._log(f"[DEBUG]   data['{key}'] 类型: {type(value)}")
                    if isinstance(value, dict):
                        self._log(f"[DEBUG]     子键: {list(value.keys())}")

        if not self._initialized:
            self._log("[YOLOSegWriter] 错误: Writer 未初始化！")
            return

        try:
            # 提取 RGB 数据
            rgb_data = self._extract_rgb_data(data)
            if rgb_data is None:
                self._log(f"[YOLOSegWriter] 警告: 帧 {self._frame_id} 无法获取 RGB 数据")
                if self._write_call_count <= 3:
                    self._log(f"[DEBUG] RGB 提取失败，data 键: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
                self._frame_id += 1
                return

            # 提取实例分割数据
            instance_data, instance_info = self._extract_instance_data(data)
            if instance_data is None:
                self._log(f"[YOLOSegWriter] 警告: 帧 {self._frame_id} 无法获取实例分割数据")
                if self._write_call_count <= 3:
                    self._log(f"[DEBUG] 实例分割提取失败")
                self._frame_id += 1
                return

            # 首次成功提取时输出信息
            if self._write_call_count <= 3:
                self._log(f"[DEBUG] 成功提取数据: RGB shape={rgb_data.shape}, instance shape={instance_data.shape}")
                self._log(f"[DEBUG] instance_info keys: {list(instance_info.keys()) if instance_info else 'None'}")

            # 确定当前帧属于训练集还是验证集
            split = "train" if self._frame_id in self._train_indices else "val"

            # 缓存数据
            self._cached_data.append((
                self._frame_id,
                split,
                rgb_data,
                {"data": instance_data, "info": instance_info}
            ))

            # 进度提示
            if (self._frame_id + 1) % 20 == 0:
                self._log(f"[YOLOSegWriter] 已缓存 {self._frame_id + 1}/{self._total_frames} 帧...")

            self._frame_id += 1

        except Exception as e:
            self._log(f"[YOLOSegWriter] 处理帧 {self._frame_id} 时出错: {e}")
            import traceback
            self._log(traceback.format_exc())
            self._frame_id += 1

    def on_final_frame(self):
        """
        所有帧处理完成后调用，写入所有缓存的数据。
        """
        self._log(f"\n[YOLOSegWriter] on_final_frame() 被调用")
        self._log(f"[YOLOSegWriter] write() 总调用次数: {self._write_call_count}")
        self._log(f"[YOLOSegWriter] 缓存数据量: {len(self._cached_data)} 帧")

        if len(self._cached_data) == 0:
            self._log("[YOLOSegWriter] ⚠️ 警告: 没有缓存任何数据！")
            self._log("[YOLOSegWriter] 可能的原因:")
            self._log("  1. write() 方法没有被 Replicator 调用")
            self._log("  2. 数据提取失败（RGB 或 instance_segmentation 为空）")
            self._log("  3. Writer 没有正确挂载到 render_product")

        self._log(f"\n[YOLOSegWriter] 开始写入 {len(self._cached_data)} 帧数据...")

        for frame_id, split, rgb_data, instance_dict in self._cached_data:
            try:
                self._write_single_frame(frame_id, split, rgb_data, instance_dict)
            except Exception as e:
                self._log(f"[YOLOSegWriter] 写入帧 {frame_id} 时出错: {e}")
                import traceback
                self._log(traceback.format_exc())

        # 输出统计信息
        train_count = sum(1 for _, s, _, _ in self._cached_data if s == "train")
        val_count = sum(1 for _, s, _, _ in self._cached_data if s == "val")

        self._log(f"\n[YOLOSegWriter] ✅ 数据集生成完成!")
        self._log(f"  - 训练集: {train_count} 帧")
        self._log(f"  - 验证集: {val_count} 帧")
        self._log(f"  - 输出目录: {self._output_dir}")

        # 生成 data.yaml 配置文件
        self._generate_data_yaml()

        # 关闭日志文件
        if self._log_file:
            self._log_file.close()

    def _extract_rgb_data(self, data: dict) -> Optional[np.ndarray]:
        """
        从 data 字典中提取 RGB 图像数据。
        """
        # 尝试不同的键名
        for key in ["rgb", "LdrColor"]:
            if key in data:
                rgb_dict = data[key]
                if isinstance(rgb_dict, dict) and "data" in rgb_dict:
                    return rgb_dict["data"]
                elif isinstance(rgb_dict, np.ndarray):
                    return rgb_dict

        # 检查嵌套的 renderProduct 结构
        if "renderProducts" in data:
            for rp_path, rp_data in data["renderProducts"].items():
                for key in ["rgb", "LdrColor"]:
                    if key in rp_data:
                        rgb_dict = rp_data[key]
                        if isinstance(rgb_dict, dict) and "data" in rgb_dict:
                            return rgb_dict["data"]
                        elif isinstance(rgb_dict, np.ndarray):
                            return rgb_dict

        # 检查 annotators 结构
        if "annotators" in data:
            for anno_name, anno_data in data["annotators"].items():
                if anno_name in ["rgb", "LdrColor"]:
                    if isinstance(anno_data, dict):
                        # 可能是 render_product -> data 结构
                        for rp_path, rp_data in anno_data.items():
                            if isinstance(rp_data, dict) and "data" in rp_data:
                                return rp_data["data"]
                        # 或者直接有 data 键
                        if "data" in anno_data:
                            return anno_data["data"]

        return None

    def _extract_instance_data(self, data: dict) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        从 data 字典中提取实例分割数据。

        Returns:
            (instance_mask, id_to_labels) 或 (None, None)
        """
        # 尝试不同的键名
        for key in ["instance_segmentation_fast", "instance_segmentation"]:
            if key in data:
                inst_dict = data[key]
                if isinstance(inst_dict, dict):
                    inst_data = inst_dict.get("data")
                    inst_info = inst_dict.get("info", {})
                    id_to_labels = inst_info.get("idToLabels", {})
                    return inst_data, id_to_labels

        # 检查嵌套的 renderProduct 结构
        if "renderProducts" in data:
            for rp_path, rp_data in data["renderProducts"].items():
                for key in ["instance_segmentation_fast", "instance_segmentation"]:
                    if key in rp_data:
                        inst_dict = rp_data[key]
                        if isinstance(inst_dict, dict):
                            inst_data = inst_dict.get("data")
                            inst_info = inst_dict.get("info", {})
                            id_to_labels = inst_info.get("idToLabels", {})
                            return inst_data, id_to_labels

        # 检查 annotators 结构
        if "annotators" in data:
            for anno_name, anno_data in data["annotators"].items():
                if anno_name in ["instance_segmentation_fast", "instance_segmentation"]:
                    if isinstance(anno_data, dict):
                        # 可能是 render_product -> data 结构
                        for rp_path, rp_data in anno_data.items():
                            if isinstance(rp_data, dict):
                                inst_data = rp_data.get("data")
                                inst_info = rp_data.get("info", {})
                                id_to_labels = inst_info.get("idToLabels", {})
                                if inst_data is not None:
                                    return inst_data, id_to_labels
                        # 或者直接有 data 键
                        if "data" in anno_data:
                            inst_info = anno_data.get("info", {})
                            id_to_labels = inst_info.get("idToLabels", {})
                            return anno_data["data"], id_to_labels

        return None, None

    def _write_single_frame(self, frame_id: int, split: str,
                            rgb_data: np.ndarray, instance_dict: dict):
        """
        写入单帧的 RGB 图像和 YOLO 格式标签。
        """
        frame_str = f"{frame_id:06d}"

        # 保存 RGB 图像
        image_dir = os.path.join(self._output_dir, "images", split)
        image_path = os.path.join(image_dir, f"{frame_str}.jpg")

        # RGB 数据可能是 RGBA，转换为 RGB
        if rgb_data.shape[2] == 4:
            rgb_data = rgb_data[:, :, :3]

        # 保存为 JPEG（YOLO 常用格式）
        cv2.imwrite(image_path, cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR))

        # 处理实例分割，生成 YOLO 标签
        instance_mask = instance_dict["data"]
        id_to_labels = instance_dict["info"]

        height, width = instance_mask.shape

        # 收集所有有效实例的标注行
        label_lines: List[str] = []

        for inst_id_str, label_info in id_to_labels.items():
            inst_id = int(inst_id_str)

            # 解析类别名称
            if isinstance(label_info, dict):
                class_name = label_info.get("class", "")
            else:
                class_name = str(label_info)

            # 跳过忽略类别
            if class_name in IGNORE_CLASSES:
                continue

            # 获取 YOLO class_id
            yolo_class_id = CLASS_MAPPING.get(class_name)
            if yolo_class_id is None:
                continue

            # 提取该实例的二值掩码
            binary_mask = (instance_mask == inst_id).astype(np.uint8)

            # 跳过空掩码
            if not np.any(binary_mask):
                continue

            # 查找轮廓
            contours, _ = cv2.findContours(
                binary_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            # 使用最大轮廓
            contour = max(contours, key=cv2.contourArea)

            # 计算边界框用于过滤
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            # 过滤太小的目标
            if area < MIN_AREA or w < MIN_DIMENSION or h < MIN_DIMENSION:
                continue

            # 将轮廓点归一化到 [0, 1]
            normalized_points = []
            for point in contour:
                px = point[0][0] / width
                py = point[0][1] / height
                # 确保坐标在有效范围内
                px = max(0.0, min(1.0, px))
                py = max(0.0, min(1.0, py))
                normalized_points.extend([px, py])

            # 构建 YOLO 标签行: class_id x1 y1 x2 y2 ... xn yn
            label_line = f"{yolo_class_id} " + " ".join(f"{v:.6f}" for v in normalized_points)
            label_lines.append(label_line)

        # 写入标签文件
        label_dir = os.path.join(self._output_dir, "labels", split)
        label_path = os.path.join(label_dir, f"{frame_str}.txt")

        with open(label_path, 'w') as f:
            f.write("\n".join(label_lines))

    def _generate_data_yaml(self):
        """
        生成 YOLO 训练配置文件 data.yaml。
        """
        yaml_content = f"""# YOLO 分割数据集配置
# 自动生成于 {time.strftime("%Y-%m-%d %H:%M:%S")}

path: {os.path.abspath(self._output_dir)}
train: images/train
val: images/val

# 类别数量
nc: {len(CLASS_MAPPING)}

# 类别名称
names:
  0: crate
  1: crate_stack
  2: person
  3: rack
  4: floor
"""
        yaml_path = os.path.join(self._output_dir, "data.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)

        print(f"[YOLOSegWriter] 生成配置文件: {yaml_path}")


# ============================================================================
# 主函数
# ============================================================================

def capture_yolo_dataset():
    """
    主函数：采集 YOLO 分割格式数据集。
    """
    # 场景路径（根据实际环境修改）
    scene_path = "/root/gpufree-data/IsaacSim/scenes/battery_warehouse/scene_01_completed.usd"

    # 输出目录
    output_dir = "/root/gpufree-data/output_yolo"

    # 采集帧数
    num_images_to_generate = 200

    print(f"[CheckPoint 5] 📂 准备加载场景文件: {scene_path}")
    t_load = time.time()
    omni.usd.get_context().open_stage(scene_path)
    stage = omni.usd.get_context().get_stage()
    print(f"[CheckPoint 6] ✅ 场景加载成功！耗时: {time.time() - t_load:.2f} 秒")

    print("[CheckPoint 7] 📷 开始配置相机与渲染视口...")
    camera = rep.create.camera()
    render_product = rep.create.render_product(camera, (1024, 1024))

    print("[CheckPoint 8] 🎲 开始构建域随机化计算图...")

    with rep.trigger.on_frame(num_frames=num_images_to_generate):
        # --- A. 相机位姿随机化 ---
        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform((-8.0, -10.0, 1.3), (8.0, 16.0, 2.7)),
                look_at=rep.distribution.uniform((-5.0, -5.0, 0.0), (5.0, 10.0, 2.0))
            )

        # --- B. 光照域随机化 ---
        dome_light = rep.create.light(light_type="Dome")
        with dome_light:
            rep.modify.attribute("inputs:intensity", rep.distribution.uniform(400.0, 1200.0))
            rep.modify.attribute("inputs:color", rep.distribution.uniform((0.7, 0.7, 0.7), (1.0, 1.0, 1.0)))

        # --- C. 材质色彩随机化 ---
        crates = rep.get.prims(path_pattern="/World/Generated/Crate_new_*")
        with crates:
            rep.randomizer.color(colors=rep.distribution.uniform((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))

    print("[CheckPoint 9] ✅ 随机化图构建完成。")

    print("[CheckPoint 10] 💾 初始化 YOLOSegWriter...")

    # 注册自定义 Writer
    rep.writers.register_writer(YOLOSegWriter)

    # 获取 Writer 实例
    writer = rep.writers.get("YOLOSegWriter")

    # 初始化 Writer
    writer.initialize(
        output_dir=output_dir,
        total_frames=num_images_to_generate,
        train_ratio=TRAIN_RATIO
    )

    # 挂载到 render_product（必须传入列表）
    writer.attach([render_product])

    print(f"[DEBUG] Writer annotators: {writer.annotators}")
    print("[CheckPoint 11] ✅ Writer 挂载完成。")

    print(f"📂 数据将保存到: {output_dir}")
    print(f"   - images/train/  (训练集图像)")
    print(f"   - images/val/    (验证集图像)")
    print(f"   - labels/train/  (训练集标签)")
    print(f"   - labels/val/    (验证集标签)")

    print(f"\n🚀 开始执行批量数据采集！预计生成 {num_images_to_generate} 张图片。")

    print("[CheckPoint 12] ⚡ 调用 rep.orchestrator.run() 触发采集任务...")
    rep.orchestrator.run()

    print("[CheckPoint 13] ⏳ 等待采集完成...")
    rep.orchestrator.wait_until_complete()

    print("\n[CheckPoint 14] ✅ 数据集采集完美结束！")


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    capture_yolo_dataset()

    print("[CheckPoint 15] 🛑 准备关闭 SimulationApp...")
    simulation_app.close()
    print("[CheckPoint 16] 🎉 程序完全退出。")
