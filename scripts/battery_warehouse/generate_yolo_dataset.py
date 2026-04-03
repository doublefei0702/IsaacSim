#!/usr/bin/env python3
"""
Isaac Sim Replicator YOLO 格式数据集采集脚本

使用 YOLOWriter 采集 YOLO 格式的检测和实例分割数据。

输出格式（YOLO标准结构）:
output_yolo_dataset/
├── detection/                  # 目标检测数据集
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── dataset.yaml
└── segmentation/               # 实例分割数据集
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── dataset.yaml
"""

import os
import time

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

# 导入 YOLOWriter
from yolo_writer import YOLOWriter


# ============================================================================
# 配置区域
# ============================================================================

# 采集帧数
num_images_to_generate = 800

# 场景路径
scene_path = "/root/gpufree-data/IsaacSim/scenes/battery_warehouse/scene_01_completed.usd"

# 输出目录
output_dir = "/root/gpufree-data/output_yolo_dataset"

# YOLO 类别映射（根据场景语义标签定义）
# 参考 generate_dataset.py 中的语义标签类别
class_mapping = {
    "crate": 0,         # 散乱货箱
    "crate_stack": 1,   # 码垛整体
    "person": 2,        # 人体
    "rack": 3,          # 货架
    "floor": 4,         # 地面
}
# 注意: wall/ceiling/door 等忽略类别不在映射中


# ============================================================================
# 主函数
# ============================================================================

def capture_yolo_dataset():
    """
    主函数：采集 YOLO 格式数据集。
    """
    print(f"[CheckPoint 5] 📂 准备加载场景文件: {scene_path}")
    t_load = time.time()
    omni.usd.get_context().open_stage(scene_path)
    stage = omni.usd.get_context().get_stage()
    print(f"[CheckPoint 6] ✅ 场景加载成功！耗时: {time.time() - t_load:.2f} 秒")

    # ============================================================================
    # 调试：检查场景中的 prim 结构
    # ============================================================================
    print("\n" + "="*60)
    print("[调试] 检查场景中的 /World/Generated 路径结构...")
    print("="*60)

    from isaacsim.core.utils.prims import find_matching_prim_paths

    # 查找所有 Generated 下的 prims
    all_generated = find_matching_prim_paths("/World/Generated/*")
    print(f"[调试] /World/Generated/* 找到 {len(all_generated)} 个直接子节点:")
    for p in sorted(all_generated)[:20]:
        print(f"       {p}")

    # 查找散乱货箱
    crate_paths = find_matching_prim_paths("/World/Generated/Crate_new_*")
    print(f"\n[调试] 散乱货箱 Crate_new_*: 找到 {len(crate_paths)} 个")

    # 查找码垛 Pallet
    pallet_paths = find_matching_prim_paths("/World/Generated/Pallet_*")
    print(f"[调试] 码垛 Pallet_*: 找到 {len(pallet_paths)} 个")
    for p in sorted(pallet_paths)[:10]:
        # 查找该 Pallet 下的 Crate
        crate_in_pallet = find_matching_prim_paths(f"{p}/Crate_*")
        print(f"       {p} -> 子节点: {len(crate_in_pallet)} 个")

    # 检查语义标签
    print("\n[调试] 检查场景语义标签...")
    from pxr import Usd
    all_prims = [p for p in stage.Traverse() if p.HasAPI('UsdShade')]
    print(f"[调试] 遍历到 {len(list(stage.Traverse()))} 个 prims")

    print("="*60 + "\n")

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

        # --- D. 码垛整体(crate_stack)材质色彩随机化 ---
        # Pallet_* 是码垛整体的父节点(Xform)，包含多个货箱子节点
        # 注意: Replicator path_pattern 不支持中间路径通配符，需逐个 Pallet 处理
        for pallet_idx in range(5):  # 根据调试已知有 5 个 Pallet (0-4)
            crate_in_pallet = rep.get.prims(path_pattern=f"/World/Generated/Pallet_{pallet_idx}/Crate_*")
            with crate_in_pallet:
                rep.randomizer.color(colors=rep.distribution.uniform((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))

    print("[CheckPoint 9] ✅ 随机化图构建完成。")

    print("[CheckPoint 10] 💾 初始化 YOLOWriter（YOLO 格式输出）...")

    # 注册并初始化 YOLOWriter
    rep.WriterRegistry.register(YOLOWriter)

    writer = rep.WriterRegistry.get("YOLOWriter")
    writer.initialize(
        output_dir=output_dir,
        rgb=True,
        bounding_box_2d_tight=True,     # 启用目标检测
        instance_segmentation=True,    # 启用实例分割
        class_mapping=class_mapping,
        train_val_split=0.8,            # 80% 训练, 20% 验证
        image_output_format="jpg",
        min_bbox_area=0.004,           # 最小边界框面积 (0.1%)
        min_mask_area=0.004,          # 最小 mask 面积 (0.1%)
        max_points=200,               # 多边形最大顶点数
    )

    # 挂载到 render_product
    writer.attach([render_product])

    print("[CheckPoint 11] ✅ Writer 挂载完成。")

    print(f"📂 YOLO 数据将保存到: {output_dir}")
    print(f"   - detection/  (目标检测: images/, labels/, dataset.yaml)")
    print(f"   - segmentation/  (实例分割: images/, labels/, dataset.yaml)")

    print(f"\n🚀 开始执行批量数据采集！预计生成 {num_images_to_generate} 张图片。")

    print("[CheckPoint 12] ⚡ 调用 rep.orchestrator.run() 触发采集任务...")
    rep.orchestrator.run()

    print("[CheckPoint 13] ⏳ 进入 wait_until_complete() 阻塞等待，请检查硬盘是否有文件生成...")
    rep.orchestrator.wait_until_complete()

    print("\n[CheckPoint 14] ✅ 数据集采集完美结束！")

    # 检查生成的文件
    print("\n📊 检查生成的文件:")
    det_img_train = len(os.listdir(os.path.join(output_dir, "detection", "images", "train"))) if os.path.exists(os.path.join(output_dir, "detection", "images", "train")) else 0
    det_img_val = len(os.listdir(os.path.join(output_dir, "detection", "images", "val"))) if os.path.exists(os.path.join(output_dir, "detection", "images", "val")) else 0
    seg_img_train = len(os.listdir(os.path.join(output_dir, "segmentation", "images", "train"))) if os.path.exists(os.path.join(output_dir, "segmentation", "images", "train")) else 0
    seg_img_val = len(os.listdir(os.path.join(output_dir, "segmentation", "images", "val"))) if os.path.exists(os.path.join(output_dir, "segmentation", "images", "val")) else 0

    print(f"   检测-训练集图像: {det_img_train}")
    print(f"   检测-验证集图像: {det_img_val}")
    print(f"   分割-训练集图像: {seg_img_train}")
    print(f"   分割-验证集图像: {seg_img_val}")


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    capture_yolo_dataset()

    print("[CheckPoint 15] 🛑 准备关闭 SimulationApp...")
    simulation_app.close()
    print("[CheckPoint 16] 🎉 程序完全退出。")