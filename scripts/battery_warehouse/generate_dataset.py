#!/usr/bin/env python3
"""
Isaac Sim Replicator 数据集采集脚本（简化版）

使用单个 BasicWriter 采集 RGB + 实例分割数据。

输出格式：
output_dataset/
├── rgb/                      # RGB 图像 (JPG)
├── instance_segmentation/       # 实例分割 mask (PNG)
└── instance_segmentation_semantics_mapping_*.json  # 映射文件
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


# ============================================================================
# 配置区域
# ============================================================================

# 采集帧数
num_images_to_generate = 20

# 场景路径
scene_path = "/root/gpufree-data/IsaacSim/scenes/battery_warehouse/scene_01_completed.usd"

# 输出目录（所有数据写入同一文件夹）
output_dir = "/root/gpufree-data/output_dataset"


# ============================================================================
# 主函数
# ============================================================================

def capture_dataset():
    """
    主函数：采集数据集。
    """
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

    print("[CheckPoint 10] 💾 初始化 BasicWriter（RGB + 实例分割）...")

    # 只初始化一个 Writer，输出所有数据到同一目录
    writer = rep.writers.get("BasicWriter")
    writer.initialize(
        output_dir=output_dir,
        rgb=True,
        instance_segmentation=True,
        colorize_instance_segmentation=False,  # 输出原始灰度 mask，便于后续处理
        image_output_format="jpg"
    )

    # 挂载到 render_product
    writer.attach([render_product])

    print("[CheckPoint 11] ✅ Writer 挂载完成。")

    print(f"📂 数据将保存到: {output_dir}")
    print(f"   - rgb/                      (RGB 图像，JPG 格式)")
    print(f"   - instance_segmentation/       (实例分割 mask，PNG 格式)")

    print(f"\n🚀 开始执行批量数据采集！预计生成 {num_images_to_generate} 张图片。")

    print("[CheckPoint 12] ⚡ 调用 rep.orchestrator.run() 触发采集任务...")
    rep.orchestrator.run()

    print("[CheckPoint 13] ⏳ 进入 wait_until_complete() 阻塞等待，请检查硬盘是否有文件生成...")
    rep.orchestrator.wait_until_complete()

    print("\n[CheckPoint 14] ✅ 数据集采集完美结束！")

    # 检查生成的文件
    print("\n📊 检查生成的文件:")
    rgb_count = len(os.listdir(os.path.join(output_dir, "rgb"))) if os.path.exists(os.path.join(output_dir, "rgb")) else 0
    instance_count = len(os.listdir(os.path.join(output_dir, "instance_segmentation"))) if os.path.exists(os.path.join(output_dir, "instance_segmentation")) else 0

    print(f"   RGB 文件数: {rgb_count}")
    print(f"   实例分割文件数: {instance_count}")


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    capture_dataset()

    print("[CheckPoint 15] 🛑 准备关闭 SimulationApp...")
    simulation_app.close()
    print("[CheckPoint 16] 🎉 程序完全退出。")
