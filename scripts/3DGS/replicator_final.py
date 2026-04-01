import omni.replicator.core as rep
import omni.usd
from pxr import Usd, UsdGeom, Gf
import math
import os
import numpy as np

# ================= 1. 全局配置区域 =================

# 输出路径
OUTPUT_DIR = "/root/gpufree-data/3Dreconstruction/dataset/sequence_1"

# 资产路径
ROBOT_PATH = "/World/my_warehouse_1_new/my_warehouse_1/nova_carter"
CAM_LEFT_PATH = "/World/my_warehouse_1_new/my_warehouse_1/nova_carter/chassis_link/sensors/front_hawk/left/camera_left"
CAM_RIGHT_PATH = "/World/my_warehouse_1_new/my_warehouse_1/nova_carter/chassis_link/sensors/front_hawk/right/camera_right"

# --- 路径规划剧本 (按顺序执行) ---
# 指令说明：
# 1. "LINE_ABS": 绝对坐标直线。参数: start=(x,y), end=(x,y), frames=int
# 2. "TURN":     圆弧转弯。    参数: angle=度数(正左负右), radius=米, frames=int
# 3. "LINE_FWD": 相对直线。    参数: distance=米(沿当前朝向继续走), frames=int
SEQUENCE_CONFIG = [
    # 第一段：直线行驶 (给定起点和终点)
    {
        "type": "LINE_ABS", 
        "start": (-84.0, 30.0), 
        "end":   (-67.0, 30.0), 
        "frames": 34
    },
    # 第二段：右转 90度 (半径1米)
    {
        "type": "TURN", 
        "angle": -90,    # 90=左转, -90=右转
        "radius": 1.0, 
        "frames": 10
    },
    # 第三段：转弯后继续直行 15米 (推荐用相对距离，不用算坐标)
    {
        "type": "LINE_FWD", 
        "distance": 20.0, 
        "frames": 40
    },
    {
        "type": "TURN", 
        "angle": -90, 
        "radius": 1.0, 
        "frames": 10
    },
    {
        "type": "LINE_FWD", 
        "distance": 3.0, 
        "frames": 6
    },
    {
        "type": "TURN", 
        "angle": -90, 
        "radius": 1.0, 
        "frames": 10
    },
    {
        "type": "LINE_FWD", 
        "distance": 40.0, 
        "frames": 80
    },
    {
        "type": "TURN", 
        "angle": 90, 
        "radius": 1.0, 
        "frames": 10
    },
    {
        "type": "LINE_FWD", 
        "distance": 4.0, 
        "frames": 8
    },
    {
        "type": "TURN", 
        "angle": 90, 
        "radius": 1.0, 
        "frames": 10
    },
    {
        "type": "LINE_FWD", 
        "distance": 40.0, 
        "frames": 80
    },
    {
        "type": "TURN", 
        "angle": -90, 
        "radius": 1.0, 
        "frames": 10
    },
    {
        "type": "LINE_FWD", 
        "distance": 3.0, 
        "frames": 6
    },
    {
        "type": "TURN", 
        "angle": -90, 
        "radius": 1.0, 
        "frames": 10
    },
    {
        "type": "LINE_FWD", 
        "distance": 40.0, 
        "frames": 80
    },
    {
        "type": "TURN", 
        "angle": 90, 
        "radius": 1.0, 
        "frames": 10
    },
    {
        "type": "LINE_FWD", 
        "distance": 3.0, 
        "frames": 6
    },
    {
        "type": "TURN", 
        "angle": 90, 
        "radius": 1.0, 
        "frames": 10
    },
    {
        "type": "LINE_FWD", 
        "distance": 19.0, 
        "frames": 38
    },
    {
        "type": "TURN", 
        "angle": 90, 
        "radius": 1.0, 
        "frames": 10
    },
    {
        "type": "LINE_FWD", 
        "distance": 23.0, 
        "frames": 46
    },
    {
        "type": "TURN", 
        "angle": -90, 
        "radius": 1.0, 
        "frames": 10
    },
    {
        "type": "LINE_FWD", 
        "distance": 6.0, 
        "frames": 12
    }
]

# ================= 2. 核心轨迹生成类 =================
class TrajectoryBuilder:
    def __init__(self):
        self.positions = []
        self.rotations = []
        # 记录当前状态，用于连续路径规划
        self.current_pos = (0, 0, 0)
        self.current_yaw = 0.0 # 度数

    def add_line_segment(self, start_pt, end_pt, num_frames):
        """添加绝对坐标直线"""
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        yaw = math.degrees(math.atan2(dy, dx))
        
        # 更新内部状态
        self.current_yaw = yaw
        
        for i in range(num_frames):
            t = i / float(num_frames)
            x = start_pt[0] + dx * t
            y = start_pt[1] + dy * t
            self.positions.append((x, y, 0.15))
            self.rotations.append((0, 0, yaw))
        
        self.current_pos = end_pt
        return end_pt

    def add_line_forward(self, distance, num_frames):
        """添加相对直线 (沿当前朝向向前走)"""
        start_pt = self.current_pos
        rad = math.radians(self.current_yaw)
        
        dx = distance * math.cos(rad)
        dy = distance * math.sin(rad)
        end_pt = (start_pt[0] + dx, start_pt[1] + dy)
        
        return self.add_line_segment(start_pt, end_pt, num_frames)

    def add_turn(self, angle_deg, radius, num_frames):
        """
        通用转弯逻辑
        
        根据当前朝向和转弯方向(angle_deg)自动计算圆心
        """
        start_pt = self.current_pos
        start_yaw = self.current_yaw
        
        # 1. 计算圆心位置
        # 左转(+): 圆心在当前朝向的左侧 (+90度方向)
        # 右转(-): 圆心在当前朝向的右侧 (-90度方向)
        
        # 如果是左转 (angle > 0)，圆心方向是当前yaw + 90
        # 如果是右转 (angle < 0)，圆心方向是当前yaw - 90
        direction_sign = 1 if angle_deg >= 0 else -1
        center_angle_rad = math.radians(start_yaw + 90 * direction_sign)
        
        center_x = start_pt[0] + radius * math.cos(center_angle_rad)
        center_y = start_pt[1] + radius * math.sin(center_angle_rad)
        
        # 2. 计算起始角度和终止角度 (相对于圆心)
        # 向量 center -> start 的角度
        vec_x = start_pt[0] - center_x
        vec_y = start_pt[1] - center_y
        start_theta = math.atan2(vec_y, vec_x) # 弧度
        
        # 总转过的弧度
        sweep_rad = math.radians(angle_deg)
        
        for i in range(num_frames):
            t = (i+1) / float(num_frames)
            theta = start_theta + sweep_rad * t # 当前在圆上的角度
            
            # 计算位置
            x = center_x + radius * math.cos(theta)
            y = center_y + radius * math.sin(theta)
            
            # 计算切线朝向 (机器人的yaw)
            # 左转: 切线是半径角度 + 90
            # 右转: 切线是半径角度 - 90
            tangent_yaw = math.degrees(theta) + (90 * direction_sign)
            
            self.positions.append((x, y, 0.05))
            self.rotations.append((0, 0, tangent_yaw))
            
            # 更新最后状态
            if i == num_frames - 1:
                self.current_pos = (x, y)
                self.current_yaw = tangent_yaw

# ================= 3. 主逻辑 =================

# 3.1 生成轨迹数据
builder = TrajectoryBuilder()
print("🤖 正在根据配置生成路径...")

for step in SEQUENCE_CONFIG:
    stype = step["type"]
    frames = step["frames"]
    
    if stype == "LINE_ABS":
        print(f" -> 添加直线: {step['start']} -> {step['end']}")
        builder.add_line_segment(step["start"], step["end"], frames)
        
    elif stype == "TURN":
        print(f" -> 添加转弯: {step['angle']}度, 半径 {step['radius']}m")
        builder.add_turn(step["angle"], step["radius"], frames)
        
    elif stype == "LINE_FWD":
        print(f" -> 添加前行: {step['distance']}m")
        builder.add_line_forward(step["distance"], frames)

print(f"✅ 路径生成完毕，总计 {len(builder.positions)} 帧")

# 3.2 Replicator 配置
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 创建相机 RenderProduct
cam_left = rep.create.render_product(CAM_LEFT_PATH, (1024, 1024), name="Cam_Left")
try:
    cam_right = rep.create.render_product(CAM_RIGHT_PATH, (1024, 1024), name="Cam_Right")
    products = [cam_left, cam_right]
except:
    print("⚠️ 未找到右相机，仅使用左相机")
    products = [cam_left]

# 初始化 Writer (包含所有你需要的数据)
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir=OUTPUT_DIR,
    rgb=True,
    camera_params=True,            # 内参 + 外参 (Pose)
    # 分割设置 (Raw ID 模式，适合算法处理)
    semantic_segmentation=True,    
    instance_segmentation=False,    
    colorize_semantic_segmentation=False, 
    colorize_instance_segmentation=False,
    
    # 原始点云 (用于生成 Global Map)
    pointcloud=False,
    
    bounding_box_2d_tight=False
)
writer.attach(products)

# 3.3 执行运动
def move_robot_sequence():
    robot_group = rep.create.group([ROBOT_PATH])
    with rep.trigger.on_frame(num_frames=len(builder.positions)):
        with robot_group:
            rep.modify.pose(
                position=rep.distribution.sequence(builder.positions),
                rotation=rep.distribution.sequence(builder.rotations)
            )

print(f"🚀 开始采集...")
move_robot_sequence()
rep.orchestrator.run()