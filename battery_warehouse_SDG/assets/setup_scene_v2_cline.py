import random
from isaacsim import SimulationApp
import math
# 1. 启动仿真环境
simulation_app = SimulationApp({"headless": False})
import omni.timeline
import omni.usd
from isaacsim.core.utils.prims import create_prim
from isaacsim.core.utils.semantics import add_labels, remove_all_semantics, remove_labels
from isaacsim.core.utils.prims import find_matching_prim_paths
from pxr import UsdGeom, Usd, Sdf
import omni.usd


# ==========================================
# 码垛生成参数 - 可在此处调整
# ==========================================

# 码垛生成区域
PALLET_ZONES = [
    {"x_range": (-10, 9), "y_range": (-11, -4.5)},   # 区域1: -10<x<9, -11<y<-4.5
    {"x_range": (-5, 9), "y_range": (12.6, 16)}  # 区域2: -5<x<9, 12.6<y<16
]

# 码垛规模参数
PALLET_MIN_LAYERS = 3   # 最少层数
PALLET_MAX_LAYERS = 5   # 最多层数
PALLET_MIN_COLS = 2     # 最少每层列数
PALLET_MAX_COLS = 4     # 最多每层列数
PALLET_MIN_ROWS = 2     # 最少每层行数
PALLET_MAX_ROWS = 4     # 最多每层行数

# 货箱尺寸参数
PALLET_CRATE_SCALE_XY = 1.0   # XY轴缩放
PALLET_CRATE_SCALE_Z = 1.3    # Z轴缩放

# 货箱间距参数
PALLET_CRATE_SPACING_X = 0.02  # 水平X方向间距
PALLET_CRATE_SPACING_Y = 0.02  # 水平Y方向间距
PALLET_LAYER_SPACING_Z = 0.5   # 垂直层间距

# 货箱随机旋转参数
PALLET_CRATE_RANDOM_ROTATION = True  # 是否启用货箱随机180°旋转
PALLET_CRATE_ROTATION_PROBABILITY = 0.5  # 货箱180°旋转的概率

# 码垛数量控制
PALLET_NUM_PALLETS_ZONE1 = 4   # 区域1码垛数量
PALLET_NUM_PALLETS_ZONE2 = 3   # 区域2码垛数量
PALLET_PLACE_PROBABILITY = 0.8  # 放置概率

# 语义标签
PALLET_SEMANTIC_LABEL = "crate_stack"  # 码垛整体的语义标签
CRATE_SEMANTIC_LABEL = "plastic_crate"  # 散乱货箱的语义标签


def apply_semantic_labels_to_groups(stage):
    """
    语义清洗与打标：
    获取 /Root 路径下的 Group，并根据组名进行语义打标
    - RackShelf_00 到 0X → "rack"
    - floor 相关的组 → "floor"
    - ceiling 相关的组 → "ceiling"
    - 其他组根据名称自动识别
    """
    print("\n" + "="*60)
    print("[语义打标] 开始获取 /Root 下的 Group 并进行语义打标...")
    print("="*60)
    
    # 获取 /Root 路径下的所有子节点
    root_prim = stage.GetPrimAtPath("/Root")
    if not root_prim:
        print("[错误] 无法获取 /Root 根节点")
        return
    
    child_prims = list(root_prim.GetChildren())
    print(f"[语义打标] /Root 下共有 {len(child_prims)} 个直接子节点")
    
    # 过滤出 Group 类型的节点
    group_prims = []
    for prim in child_prims:
        prim_type = prim.GetTypeName()
        if prim_type in ["Group", "Xform", "Scope"]:
            group_prims.append(prim)
    
    print(f"[语义打标] 找到 {len(group_prims)} 个 Group/Xform/Scope 节点")
    
    # 调试输出：列出所有子节点路径和类型
    print("\n--- /Root 下的所有子节点 ---")
    for prim in child_prims:
        prim_path = prim.GetPath().pathString
        prim_type = prim.GetTypeName()
        print(f"  [{prim_type}] {prim_path}")
    print("-" * 40)
    
    # 定义标签映射规则
    def get_label_from_path(path_str):
        """根据路径确定语义标签"""
        path_lower = path_str.lower()
        
        # RackShelf_00 到 0X → rack
        if "rackshelf" in path_lower or "rack" in path_lower or "shelf" in path_lower:
            return "rack"
        
        # floor/ground → floor
        if "floor" in path_lower or "ground" in path_lower:
            return "floor"
        
        # ceiling → ceiling
        if "ceiling" in path_lower or "roof" in path_lower:
            return "ceiling"
        
        # wall → wall
        if "wall" in path_lower:
            return "wall"
        
        # door → door
        if "door" in path_lower:
            return "door"
        
        # 默认返回 None，表示不处理
        return None
    
    # 统计信息
    labeled_count = 0
    skipped_count = 0
    
    # 对每个 Group 进行语义清洗与打标
    for prim in group_prims:
        prim_path = prim.GetPath().pathString
        prim_name = prim_path.split("/")[-1]
        
        # 确定标签
        label = get_label_from_path(prim_path)
        
        if label is None:
            print(f"[跳过] {prim_path} (无匹配标签)")
            skipped_count += 1
            continue
        
        try:
            # 1. 递归清洗子节点自带的语义
            remove_all_semantics(prim, recursive=True)
            remove_labels(prim, include_descendants=True)
            
            # 2. 整体打标
            add_labels(prim, labels=[label], instance_name="class")
            
            print(f"[打标] ✅ {prim_path} → {label}")
            labeled_count += 1
            
        except Exception as e:
            print(f"[错误] 处理 {prim_path} 失败: {e}")
    
    print("\n" + "="*60)
    print(f"[语义打标] 完成！共处理 {labeled_count} 个组，跳过 {skipped_count} 个")
    print("="*60)


def populate_warehouse():
    
    # 场景路径
    scene_path = "/root/gpufree-data/battery_warehouse_SDG/assets/scene_01_6shelfves.usd"
    omni.usd.get_context().open_stage(scene_path)
    stage = omni.usd.get_context().get_stage()
    
    # ==========================================
    # 语义清洗与打标
    # ==========================================
    apply_semantic_labels_to_groups(stage)
    
    # 官方资产库
    crate_urls = [
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/Props//SM_CratePlasticNote_A_01.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/Props//SM_CratePlastic_A_01.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/Props//SM_CratePlastic_C_02.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/Props//SM_CratePlastic_E_02.usd"
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
        (-9.3,4.35,1.0, 16.0),
        (-4.7, 4.35, 1.0, 16.0),
        (0.0,  4.35, 1.0, 16.0),
        (4.5,  4.35, 1.0, 16.0),
        (8.6,  4.35, 1.0, 16.0),
        (0.0,  16.8, 16.0, 1.0),
        (0.0, -11.7, 16.0, 1.0)
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
                        
                        # 计算货箱方向：随机180°旋转（绕Z轴）
                        crate_orientation = [1.0, 0.0, 0.0, 0.0]  # 默认无旋转 (w=1, x=y=z=0)
                        if PALLET_CRATE_RANDOM_ROTATION and random.random() < PALLET_CRATE_ROTATION_PROBABILITY:
                            # 180°旋转：四元数 [w, x, y, z] = [0, 0, 0, 1]
                            crate_orientation = [0.0, 0.0, 0.0, 1.0]
                        
                        prim_path = f"/World/Generated/Pallet_{pallet_counter}/Crate_{crate_counter}"
                        prim = create_prim(
                            prim_path=prim_path,
                            translation=[pos_x, pos_y, z_height],  # 相对于父节点的位置
                            scale=[PALLET_CRATE_SCALE_XY, PALLET_CRATE_SCALE_XY, PALLET_CRATE_SCALE_Z],
                            orientation=crate_orientation,
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
    # 2. 放置过道上的人 - 避开码垛区域
    # 码垛区域1: x_range=(-10,9), y_range=(-11,-4.5)
    # 码垛区域2: x_range=(-5,9), y_range=(12.6,16)
    # 安全区域: x=(-10,9), y=(-4.5,12.6) 即货架之间的过道
    
    # 调整过道位置，避开码垛区域
    # 货架位置在 x=-9.3, -4.7, 0.0, 4.5, 8.6 附近，过道应在两排货架之间
    aisle_x_positions = [-7.0, -2.35, 2.25, 6.75]  # 扩展过道位置
    # 使用安全的y范围：(-4.5, 12.6) 避开两个码垛区域
    aisle_y_range = (-2.0, 10.0)  # 避开码垛区域1(y<-4.5)和区域2(y>12.6)
    
    # 调试输出人物生成位置
    print(f"[人物生成] 过道X位置: {aisle_x_positions}")
    print(f"[人物生成] Y轴范围: {aisle_y_range}")
    
    for i in range(6): 
        selected_human = random.choice(human_urls)
        
        base_x = random.choice(aisle_x_positions)
        h_pos_x = base_x + random.uniform(-0.6, 0.6) 
        h_pos_y = random.uniform(*aisle_y_range)
        
        print(f"[人物生成] Human_{i}: 位置=({h_pos_x:.2f}, {h_pos_y:.2f})")
        
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
