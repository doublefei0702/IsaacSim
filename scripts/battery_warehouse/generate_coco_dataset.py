#!/usr/bin/env python3
"""
COCO 格式数据集采集脚本

使用 Isaac Sim Replicator 的 CocoWriter 采集 COCO 格式数据。
然后可以使用单独的脚本将 COCO 转换为 YOLO 格式。

输出格式：
output_coco/
├── rgb/                    # RGB 图像
├── instance_segmentation/   # 实例分割 mask
├── semantic_segmentation/   # 语义分割 mask
├── bounding_box_2d_tight/   # 2D 边界框
└── _annotations.coco.json     # COCO 标注文件
"""

import os
import time
import json

print("[CheckPoint 1] 🚀 开始启动 SimulationApp (无头模式)...")
t_start = time.time()
from isaacsim import SimulationApp
# 启动无头模式 (Headless: True)，后台极速渲染，不弹界面
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

# COCO 类别定义（必须按照 COCO 格式）
# 参考: https://cocodataset.org/#format-data
COCO_CATEGORIES = [
    {"id": 1, "name": "crate", "supercategory": "object"},
    {"id": 2, "name": "crate_stack", "supercategory": "object"},
    {"id": 3, "name": "person", "supercategory": "person"},
    {"id": 4, "name": "storage_rack", "supercategory": "furniture"},
    {"id": 5, "name": "floor", "supercategory": "background"},
]

# 要考虑的语义类型
SEMANTIC_TYPES = ["class"]

# 忽略的类别（不写入 COCO 标注）
IGNORE_LABELS = ["BACKGROUND", "UNLABELLED", "wall", "ceiling", "fire_extinguisher", "door"]

# 过滤阈值（最小面积，像素）
MIN_ANNOTATION_AREA = 1600


# ============================================================================
# 主函数
# ============================================================================

def capture_coco_dataset():
    """
    主函数：使用 CocoWriter 采集 COCO 格式数据集。
    """
    # 场景路径（根据实际环境修改）
    scene_path = "/root/gpufree-data/IsaacSim/scenes/battery_warehouse/scene_01_completed.usd"

    # 输出目录
    output_dir = "/root/gpufree-data/output_coco"

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

    print("[CheckPoint 10] 💾 初始化 CocoWriter...")

    # 获取 CocoWriter 实例
    coco_writer = rep.writers.get("CocoWriter")

    # 初始化 CocoWriter
    # 参考: omni.replicator.core.writers_default.CocoWriter
    coco_writer.initialize(
        output_dir=output_dir,
        coco_categories=COCO_CATEGORIES,
        semantic_types=SEMANTIC_TYPES,
        rgb=True,                      # 输出 RGB 图像
        semantic_segmentation=True,        # 输出语义分割
        instance_segmentation=True,        # 输出实例分割
        bounding_box_2d_tight=True,     # 输出 2D 边界框
        image_output_format="jpg",         # 图像格式
    )

    # 挂载到 render_product
    coco_writer.attach([render_product])

    print("[CheckPoint 11] ✅ Writer 挂载完成。")

    print(f"📂 数据将保存到: {output_dir}")
    print(f"   - rgb/                    (RGB 图像)")
    print(f"   - instance_segmentation/   (实例分割 mask)")
    print(f"   - semantic_segmentation/   (语义分割 mask)")
    print(f"   - bounding_box_2d_tight/   (2D 边界框)")
    print(f"   - _annotations.coco.json    (COCO 标注文件)")

    print(f"\n🚀 开始执行批量数据采集！预计生成 {num_images_to_generate} 张图片。")

    print("[CheckPoint 12] ⚡ 调用 rep.orchestrator.run() 触发采集任务...")
    rep.orchestrator.run()

    print("[CheckPoint 13] ⏳ 进入 wait_until_complete() 阻塞等待，请检查硬盘是否有文件生成...")
    rep.orchestrator.wait_until_complete()

    print("\n[CheckPoint 14] ✅ 数据集采集完美结束！")

    # 检查生成的文件
    annotation_file = os.path.join(output_dir, "_annotations.coco.json")
    if os.path.exists(annotation_file):
        print(f"\n📄 生成的标注文件: {annotation_file}")

        # 统计信息
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        print(f"\n📊 数据集统计:")
        print(f"   - 图像数量: {len(coco_data.get('images', []))}")
        print(f"   - 标注数量: {len(coco_data.get('annotations', []))}")
        print(f"   - 类别数量: {len(coco_data.get('categories', []))}")
    else:
        print(f"\n⚠️ 警告: 未找到标注文件 {annotation_file}")


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    capture_coco_dataset()

    print("[CheckPoint 15] 🛑 准备关闭 SimulationApp...")
    simulation_app.close()
    print("[CheckPoint 16] 🎉 程序完全退出。")
