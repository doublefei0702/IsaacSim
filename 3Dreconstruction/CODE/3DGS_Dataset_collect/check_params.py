import json
import numpy as np

# ================= 配置 =================
# 挑同一帧的左右两个 json 文件路径
json_left_path  = "/root/gpufree-data/3Dreconstruction/dataset/sequence_collect_step1/Replicator_03/camera_params/camera_params_0011.json"
json_right_path = "/root/gpufree-data/3Dreconstruction/dataset/sequence_collect_step1/Replicator_04/camera_params/camera_params_0011.json" 
# (注：如果还没改名，先用 03/04 测；如果改名了用 Cam_Left/Cam_Right)
# =======================================

def get_matrix(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Replicator 输出的是 4x4 行主序矩阵 (Row-Major)
    # cameraViewTransform 通常是 World -> Camera (视图矩阵)
    # 但 Isaac Sim 有时输出的是 Camera -> World (Pose)，我们需要检查一下平移部分
    pose_flat = data['cameraViewTransform']
    pose_mat = np.array(pose_flat).reshape(4, 4)
    return pose_mat

T_left = get_matrix(json_left_path)
T_right = get_matrix(json_right_path)

# Extract Translation (最后一行前三列，因为是行主序)
# Isaac Sim 的 cameraViewTransform 物理含义通常是 World-to-Camera
# 所以相机在世界坐标系的位置 (Camera Position) 需要求逆矩阵，或者直接看平移分量逻辑
# 但在 Replicator 中，为了方便，我们先假设最后一行是平移 (如果是 Pose 矩阵)
# 或者直接通过矩阵求逆获取中心点
pos_left = np.linalg.inv(T_left)[3, :3]
pos_right = np.linalg.inv(T_right)[3, :3]

dist = np.linalg.norm(pos_left - pos_right)

print(f"Left Camera Pos: {pos_left}")
print(f"Right Camera Pos: {pos_right}")
print(f"---------------------------")
print(f"Calculated Baseline (距离): {dist:.4f} meters")

# 验证标准：
# Nova Carter 的双目基线通常是 0.28m 或 0.1m (取决于具体型号)
# 如果结果是 0，说明两个相机重叠了（路径配错）。
# 如果结果很大，说明配错了传感器。