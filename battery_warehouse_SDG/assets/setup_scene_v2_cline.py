import random
from isaacsim import SimulationApp
import math
# 1. 启动仿真环境
simulation_app = SimulationApp({"headless": False})
import omni.timeline
import omni.usd
from isaacsim.core.utils.prims import create_prim
from isaacsim.core.utils.semantics import add_labels, remove_all_semantics, remove_labels
from pxr import UsdGeom, Usd


# ==========================================
# 码垛生成参数 - 可在此处调整
# ==========================================

# 码垛生成区域
PALLET_ZONES = [
    {"x_range": (-4, 6), "y_range": (13, 16)},   # 区域1: -4<x<6, 13<y<16
    {"x_range": (-6, 6), "y_range": (-18, -14)}  # 区域2: -6<x<6, -18<y<-14
]

# 码垛规模参数
PALLET_MIN_LAYERS = 3   # 最少层数
PALLET_MAX_LAYERS = 5   # 最多层数
PALLET_MIN_COLS = 2     # 最少每层列数
PALLET_MAX_COLS = 4     # 最多每层列数
PALLET_MIN_ROWS = 2     # 最少每层行数
PALLET_MAX_ROWS = 3     # 最多每层行数

# 货箱尺寸参数
PALLET_CRATE_SCALE_XY = 1.0   # XY轴缩放
PALLET_CRATE_SCALE_Z = 1.3    # Z轴缩放

# 货箱间距参数
PALLET_CRATE_SPACING_X = 0.02  # 水平X方向间距
PALLET_CRATE_SPACING_Y = 0.02  # 水平Y方向间距
PALLET_LAYER_SPACING_Z = 0.5   # 垂直层间距

# 码垛数量控制
PALLET_NUM_PALLETS_ZONE1 = 3   # 区域1码垛数量
PALLET_NUM_PALLETS_ZONE2 = 3   # 区域2码垛数量
PALLET_PLACE_PROBABILITY = 0.8  # 放置概率

# 语义标签
PALLET_SEMANTIC_LABEL = "crate_stack"  # 码垛整体的语义标签
CRATE_SEMANTIC_LABEL = "plastic_crate"  # 散乱货箱的语义标签


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
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/Props//SM_CratePlastic_B_01.usd"
    ]
    human_urls = [
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/People/Characters/male_adult_construction_03/male_adult_construction_03.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/People/Characters/male_adult_construction_01_new/male_adult_construction_01_new.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/People/Characters/male_adult_construction_05_new/male_adult_construction_05_new.usd"
    ]

    # 货架高度层级 (Z轴)
    shelf_z_levels = [0.0, 1.3, 2.8, 4.09]

    # 定义 6 个货架的信息：(中心X, 中心Y, X方向宽度, Y方向长度)
    shelves_data = [
        (-4.7, 4, 1.0, 15.0),
        (0.0,  4, 1.0, 15.0),
        (4.5,  4, 1.0, 15.0),
        (8.6,  4, 1.0, 15.0),
        (0.0,  16.8, 15.0, 1.0),
        (0.0, -11.7, 15.0, 1.0)
    ]

    # 安全边距
    margin = 0.05
    
    # 网格化排列参数
    grid_spacing = 0.55
    small_random_offset = 0.03
    
    print("开始生成货箱（网格化排列）...")
    crate_counter = 0
    
    # 1. 批量放置货箱 - 网格化排列（散乱货箱）
    for shelf_idx, (center_x, center_y, size_x, size_y) in enumerate(shelves_data):
        x_min = center_x - (size_x / 2.0) + margin
        x_max = center_x + (size_x / 2.0) - margin
        y_min = center_y - (size_y / 2.0) + margin
        y_max = center_y + (size_y / 2.0) - margin

        grid_count_x = max(1, int((x_max - x_min) / grid_spacing))
        grid_count_y = max(1, int((y_max - y_min) / grid_spacing))

        for z_height in shelf_z_levels:
            for grid_x in range(grid_count_x):
                if random.random() < 0.15:
                    continue
                for grid_y in range(grid_count_y):
                    if random.random() < 0.15:
                        continue
                    
                    grid_pos_x = x_min + (grid_x + 0.5) * (x_max - x_min) / grid_count_x
                    grid_pos_y = y_min + (grid_y + 0.5) * (y_max - y_min) / grid_count_y
                    
                    offset_x = random.uniform(-small_random_offset, small_random_offset)
                    offset_y = random.uniform(-small_random_offset, small_random_offset)
                    
                    pos_x = grid_pos_x + offset_x
                    pos_y = grid_pos_y + offset_y
                    
                    selected_crate = random.choice(crate_urls)
                    scale_xy = random.uniform(1.0, 1.2)
                    scale_z = random.uniform(1.3, 1.6)
                    
                    prim_path = f"/World/Generated/Crate_new_{crate_counter}"
                    prim = create_prim(
                        prim_path=prim_path,
                        position=[pos_x, pos_y, z_height],
                        scale=[scale_xy, scale_xy, scale_z],
                        usd_path=selected_crate
                    )
                    
                    # 散乱货箱的语义标签
                    add_labels(prim, labels=[CRATE_SEMANTIC_LABEL], instance_name="class")
                    crate_counter += 1

    # ==========================================
    # 在指定区域生成大规模码垛（带整体语义标签）
    # ==========================================
    print("开始生成指定区域码垛...")
    pallet_counter = 0
    
    # 为每个区域生成码垛
    for zone_idx, zone in enumerate(PALLET_ZONES):
        x_range = zone["x_range"]
        y_range = zone["y_range"]
        
        # 确定该区域生成多少个码垛
        num_pallets = PALLET_NUM_PALLETS_ZONE1 if zone_idx == 0 else PALLET_NUM_PALLETS_ZONE2
        
        for pallet_idx in range(num_pallets):
            # 随机决定是否放置
            if random.random() > PALLET_PLACE_PROBABILITY:
                continue
            
            # 在区域内随机生成码垛中心位置
            pallet_x = random.uniform(x_range[0] + 0.5, x_range[1] - 0.5)
            pallet_y = random.uniform(y_range[0] + 0.5, y_range[1] - 0.5)
            
            # 随机生成码垛规模
            num_layers = random.randint(PALLET_MIN_LAYERS, PALLET_MAX_LAYERS)
            num_cols = random.randint(PALLET_MIN_COLS, PALLET_MAX_COLS)
            num_rows = random.randint(PALLET_MIN_ROWS, PALLET_MAX_ROWS)
            
            # 计算码垛整体边界（用于创建Xform组）
            total_width_x = num_cols * (0.5 * PALLET_CRATE_SCALE_XY + PALLET_CRATE_SPACING_X)
            total_width_y = num_rows * (0.5 * PALLET_CRATE_SCALE_XY + PALLET_CRATE_SPACING_Y)
            total_height = num_layers * PALLET_LAYER_SPACING_Z
            
            # 创建Xform组作为码垛的整体父节点
            pallet_group_path = f"/World/Generated/Pallet_{pallet_counter}"
            
            # 计算码垛中心（基于堆叠方式，调整为从底部开始计算）
            # 这里将pallet_x, pallet_y作为码垛的基准点
            pallet_center_x = pallet_x
            pallet_center_y = pallet_y
            
            # 创建Xform组
            pallet_xform = create_prim(
                prim_path=pallet_group_path,
                prim_type="Xform",
                position=[pallet_center_x, pallet_center_y, 0.0]
            )
            
            # 在Xform组内生成货箱
            # 计算起始位置（从左下角开始）
            start_x = -total_width_x / 2
            start_y = -total_width_y / 2
            
            for layer in range(num_layers):
                z_height = layer * PALLET_LAYER_SPACING_Z
                
                for col in range(num_cols):
                    for row in range(num_rows):
                        # 计算每个货箱的位置（相对于码垛组）
                        pos_x = start_x + col * (0.5 * PALLET_CRATE_SCALE_XY + PALLET_CRATE_SPACING_X) + 0.25 * PALLET_CRATE_SCALE_XY
                        pos_y = start_y + row * (0.5 * PALLET_CRATE_SCALE_XY + PALLET_CRATE_SPACING_Y) + 0.25 * PALLET_CRATE_SCALE_XY
                        
                        selected_crate = random.choice(crate_urls)
                        
                        # 货箱在Xform组内的绝对位置
                        abs_pos_x = pallet_center_x + pos_x
                        abs_pos_y = pallet_center_y + pos_y
                        
                        prim_path = f"/World/Generated/Pallet_{pallet_counter}/Crate_{crate_counter}"
                        prim = create_prim(
                            prim_path=prim_path,
                            position=[pos_x, pos_y, z_height],  # 相对于父节点的位置
                            scale=[PALLET_CRATE_SCALE_XY, PALLET_CRATE_SCALE_XY, PALLET_CRATE_SCALE_Z],
                            usd_path=selected_crate
                        )
                        
                        # 货箱本身不再添加任何标签，因为父节点 Pallet 会有整体标签
                        crate_counter += 1
            
            # ==========================================
            # 核心修复：递归清洗 + 整体打标
            # ==========================================
            # 1. 递归清洗：一键剥离组内所有子孙节点自带的旧版和新版语义
            pallet_prim = stage.GetPrimAtPath(pallet_group_path)
            remove_all_semantics(pallet_prim, recursive=True)
            remove_labels(pallet_prim, include_descendants=True)
            
            # 2. 整体打标：只给最顶层的父节点重新赋予我们想要的整体标签
            add_labels(pallet_prim, labels=[PALLET_SEMANTIC_LABEL], instance_name="class")
            # ==========================================
            
            pallet_counter += 1
            print(f"  区域{zone_idx+1}: 生成码垛 {pallet_counter}, 规模 {num_layers}层 x {num_cols}列 x {num_rows}行")

    print("开始生成人物...")
    # 2. 放置过道上的人
    aisle_x_positions = [-2.35, 2.25, 6.75]
    aisle_y_range = (0.0, 14.0) 
    
    for i in range(5): 
        selected_human = random.choice(human_urls)
        
        base_x = random.choice(aisle_x_positions)
        h_pos_x = base_x + random.uniform(-0.6, 0.6) 
        h_pos_y = random.uniform(*aisle_y_range)
        
        angle = random.uniform(0, 2 * math.pi)
        qw = math.cos(angle / 2)
        qz = math.sin(angle / 2)
        random_orientation = [qw, 0.0, 0.0, qz]
        
        human_path = f"/World/Generated/Human_{i}"
        human_prim = create_prim(
            prim_path=human_path,
            prim_type="Xform",
            position=[h_pos_x, h_pos_y, 0.0], 
            orientation=random_orientation,
            usd_path=selected_human
        )
        add_labels(human_prim, labels=["person"], instance_name="class")
        
    print(f"✅ 场景填充完毕！共生成了 {crate_counter} 个货箱（散乱货箱 + 码垛货箱），{pallet_counter} 个码垛，以及 5 个人物。")
    print(f"📍 码垛区域: 区域1({PALLET_ZONES[0]}), 区域2({PALLET_ZONES[1]})")
    print(f"📦 码垛参数: {PALLET_MIN_LAYERS}-{PALLET_MAX_LAYERS}层, {PALLET_MIN_COLS}-{PALLET_MAX_COLS}列, {PALLET_MIN_ROWS}-{PALLET_MAX_ROWS}行")

populate_warehouse()

#omni.timeline.get_timeline_interface().play()
while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
