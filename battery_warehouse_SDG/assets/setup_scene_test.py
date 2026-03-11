import random
from isaacsim import SimulationApp
import math # 记得在文件顶部 import math
# 1. 启动仿真环境
simulation_app = SimulationApp({"headless": False})
import omni.timeline
import omni.usd
from isaacsim.core.utils.prims import create_prim
from isaacsim.core.utils.semantics import add_labels
from pxr import UsdGeom

def populate_warehouse():
    
    # 场景路径
    scene_path = "/root/gpufree-data/battery_warehouse_SDG/assets/scene_01_6shelfves.usd"
    omni.usd.get_context().open_stage(scene_path)
    stage = omni.usd.get_context().get_stage()
    
    # 官方资产库
    crate_urls = [
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/Props//SM_CratePlasticNote_A_01.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/Props//SM_CratePlastic_A_01.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/Props//SM_CratePlastic_C_02.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/Props//SM_CratePlastic_E_02.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/Props//SM_CratePlastic_B_01.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_D_02.usd"
    ]
    human_urls = [
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/People/Characters/male_adult_construction_03/male_adult_construction_03.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/People/Characters/male_adult_construction_01_new/male_adult_construction_01_new.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/People/Characters/male_adult_construction_05_new/male_adult_construction_05_new.usd"
    ]

    # 货架高度层级 (Z轴)
    shelf_z_levels = [0.0, 1.3, 2.8, 4.09]

    # 定义 6 个货架的信息：(中心X, 中心Y, X方向宽度, Y方向长度)
    # 前四个沿 Y 轴 (X宽1m, Y长15m)，后两个沿 X 轴 (X宽16m, Y长1m)
    shelves_data = [
        (-4.7, 4, 1.0, 15.0),
        (0.0,  4, 1.0, 15.0),
        (4.5,  4, 1.0, 15.0),
        (8.6,  4, 1.0, 15.0),
        (0.0,  16.8, 15.0, 1.0),
        (0.0, -11.7, 15.0, 1.0)
    ]

    # 为了防止货箱边缘悬空或掉落，设置一个边缘安全边距 (Margin)
    margin = 0.05 # 5cm 的安全边距，确保货箱完全在货架范围内    
    # ==========================================
    # NEW: Minimum distance between bin centers to avoid overlap
    # 0.5m is a good starting point based on typical bin sizes.
    # Increase this if bins still overlap, decrease if they are too far apart.
    min_distance = 0.5 
    # ==========================================
    print("开始生成货箱...")
    crate_counter = 0
    # 1. 批量放置货箱
    for shelf_idx, (center_x, center_y, size_x, size_y) in enumerate(shelves_data):
        # 计算该货架的有效随机生成区间
        x_min = center_x - (size_x / 2.0) + margin
        x_max = center_x + (size_x / 2.0) - margin
        y_min = center_y - (size_y / 2.0) + margin
        y_max = center_y + (size_y / 2.0) - margin

        for z_height in shelf_z_levels:
            # Number of crates to attempt placing on this layer
            num_crates_to_place = random.randint(25, 35)
            
            # ==========================================
            # NEW: List to keep track of placed positions on THIS layer
            # ==========================================
            placed_positions = []
            
            for _ in range(num_crates_to_place):
                selected_crate = random.choice(crate_urls)
                scale_xy = random.uniform(1.0, 1.2)  # X 和 Y 轴保持 1 到 1.2 倍的随机缩放
                scale_z = random.uniform(1.3, 1.6)   # Z 轴单独拉大，例如 1.3 到 1.6 倍（你可以根据视觉效果微调这个范围）
                
                # ==========================================
                # NEW: Rejection Sampling Logic
                # ==========================================
                best_pos = None
                max_attempts = 50 # Give up after 50 tries if the shelf is full
                
                for attempt in range(max_attempts):
                    # 1. Pick a random candidate position
                    candidate_x = random.uniform(x_min, x_max)
                    candidate_y = random.uniform(y_min, y_max)
                    
                    # 2. Check distance against all previously placed bins on this layer
                    is_valid = True
                    for (placed_x, placed_y) in placed_positions:
                        # Calculate Euclidean distance
                        dist = math.sqrt((candidate_x - placed_x)**2 + (candidate_y - placed_y)**2)
                        if dist < min_distance:
                            is_valid = False # Too close! Reject.
                            break
                    
                    # 3. If valid, accept it and stop trying
                    if is_valid:
                        best_pos = (candidate_x, candidate_y)
                        break
                
                # If we couldn't find a valid position after max_attempts, skip this bin
                if best_pos is None:
                    # print(f"Warning: Could not find a slot for a bin on shelf {shelf_idx}, layer {z_height} after {max_attempts} attempts.")
                    continue 
                    
                pos_x, pos_y = best_pos
                # Add the successful position to our list for future checks
                placed_positions.append((pos_x, pos_y))
                # ==========================================

                prim_path = f"/World/Generated/Crate_new_{crate_counter}"
                prim = create_prim(
                    prim_path=prim_path,
                    position=[pos_x, pos_y, z_height],
                    scale=[scale_xy, scale_xy, scale_z],
                    usd_path=selected_crate
                )
                
                add_labels(prim, labels=["plastic_crate"], instance_name="class")
                crate_counter += 1

    print("开始生成人物...")
    # 2. 放置过道上的人
    aisle_x_positions = [-2.35, 2.25, 6.75]
    aisle_y_range = (0.0, 14.0) 
    
    for i in range(5): 
        selected_human = random.choice(human_urls)
        
        # 1. 位置随机化：在中心线基础上，左右随机偏移 0.6 米
        base_x = random.choice(aisle_x_positions)
        h_pos_x = base_x + random.uniform(-0.6, 0.6) 
        h_pos_y = random.uniform(*aisle_y_range)
        
        # 2. 朝向随机化：生成 0 到 360 度的随机旋转角
        angle = random.uniform(0, 2 * math.pi)
        # 将欧拉角转换为四元数 [w, x, y, z] 绕 Z 轴旋转
        qw = math.cos(angle / 2)
        qz = math.sin(angle / 2)
        random_orientation = [qw, 0.0, 0.0, qz]
        
        human_path = f"/World/Generated/Human_{i}"
        human_prim = create_prim(
            prim_path=human_path,
            prim_type="Xform",
            position=[h_pos_x, h_pos_y, 0.0], 
            orientation=random_orientation, # 应用随机朝向
            usd_path=selected_human
        )
        add_labels(human_prim, labels=["person"], instance_name="class")
        
    print(f"✅ 场景填充完毕！共在 6 个货架上生成了 {crate_counter} 个货箱，以及 5 个人物。")

populate_warehouse()
# 在代码末尾，启动时间线让人物播放默认的自然站立动画

#omni.timeline.get_timeline_interface().play()
while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
