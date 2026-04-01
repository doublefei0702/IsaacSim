import os
import time

print("[CheckPoint 1] 🚀 开始启动 SimulationApp (无头模式)...")
t_start = time.time()
from isaacsim import SimulationApp
# 启动无头模式 (Headless: True)，后台极速渲染，不弹界面
simulation_app = SimulationApp({"headless": True})
print(f"[CheckPoint 2] ✅ SimulationApp 启动完成！耗时: {time.time() - t_start:.2f} 秒")

print("[CheckPoint 3] 📦 开始导入底层模块 (omni.replicator, omni.usd)...")
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.utils.semantics import add_labels
print("[CheckPoint 4] ✅ 模块导入完毕。")

def capture_sam3_dataset():
    # 你的场景路径
    scene_path = "/root/gpufree-data/battery_warehouse_SDG/assets/scene_01_completed.usd"
    
    print(f"[CheckPoint 5] 📂 准备加载场景文件: {scene_path}")
    t_load = time.time()
    omni.usd.get_context().open_stage(scene_path)
    stage = omni.usd.get_context().get_stage()
    print(f"[CheckPoint 6] ✅ 场景加载成功！耗时: {time.time() - t_load:.2f} 秒")

    print("[CheckPoint 7] 📷 开始配置相机与渲染视口...")
    camera = rep.create.camera()
    render_product = rep.create.render_product(camera, (1024, 1024))

    print("[CheckPoint 8] 🎲 开始构建域随机化 (Domain Randomization) 计算图...")
    num_images_to_generate = 200
    
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

    print("[CheckPoint 10] 💾 开始初始化 BasicWriter 并配置输出格式...")
    base_out_dir = "/root/gpufree-data/battery_warehouse_SDG/output_dataset"
    
    # 创建各个类型的输出文件夹
    rgb_dir = os.path.join(base_out_dir, "rgb")
    instance_dir = os.path.join(base_out_dir, "instance_segmentation")
    semantic_dir = os.path.join(base_out_dir, "semantic_segmentation")
    bbox_dir = os.path.join(base_out_dir, "bounding_box_2d_tight")
    colorize_dir = os.path.join(base_out_dir, "colorize_semantic_segmentation")
    
    # 创建独立的 writers 分别写入不同文件夹
    writer_rgb = rep.writers.get("BasicWriter")
    writer_rgb.initialize(output_dir=rgb_dir, rgb=True)
    writer_rgb.attach([render_product])
    
    writer_instance = rep.writers.get("BasicWriter")
    writer_instance.initialize(output_dir=instance_dir, instance_segmentation=True)
    writer_instance.attach([render_product])
    
    writer_semantic = rep.writers.get("BasicWriter")
    writer_semantic.initialize(output_dir=semantic_dir, semantic_segmentation=True)
    writer_semantic.attach([render_product])
    
    writer_bbox = rep.writers.get("BasicWriter")
    writer_bbox.initialize(output_dir=bbox_dir, bounding_box_2d_tight=True)
    writer_bbox.attach([render_product])
    
    writer_colorize = rep.writers.get("BasicWriter")
    writer_colorize.initialize(output_dir=colorize_dir, colorize_semantic_segmentation=True)
    writer_colorize.attach([render_product])
    
    print("[CheckPoint 11] ✅ Writer 挂载完成。")
    
    print(f"📂 数据将分别保存在:")
    print(f"   - RGB: {rgb_dir}")
    print(f"   - Instance Segmentation: {instance_dir}")
    print(f"   - Semantic Segmentation: {semantic_dir}")
    print(f"   - Bounding Box 2D: {bbox_dir}")
    print(f"   - Colorize Semantic: {colorize_dir}")
    
    print(f"\n🚀 开始执行批量数据采集！预计生成 {num_images_to_generate} 张图片。")
    
    print("[CheckPoint 12] ⚡ 调用 rep.orchestrator.run() 触发采集任务...")
    rep.orchestrator.run()
    
    print("[CheckPoint 13] ⏳ 进入 wait_until_complete() 阻塞等待，请检查硬盘是否有文件生成...")
    rep.orchestrator.wait_until_complete()
    
    print("\n[CheckPoint 14] ✅ 数据集采集完美结束！")

capture_sam3_dataset()

print("[CheckPoint 15] 🛑 准备关闭 SimulationApp...")
simulation_app.close()
print("[CheckPoint 16] 🎉 程序完全退出。")
