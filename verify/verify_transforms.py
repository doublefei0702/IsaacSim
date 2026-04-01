import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ================= 配置 =================
JSON_PATH = "/root/gpufree-data/3Dreconstruction/dataset/sequence_1/Cam_Left_07/processed_params/transforms.json"

def plot_camera(ax, c2w, scale=0.5, color='blue'):
    """在 3D 图中画一个相机 (金字塔形状)"""
    # 相机局部坐标系的 5 个关键点: 原点 + 图像平面的 4 个角
    # 假设 OpenCV 坐标系: +Z 向前, +X 向右, +Y 向下
    w, h = 1.0, 0.75
    z = 1.0
    
    # 局部坐标系下的点
    points_local = np.array([
        [0, 0, 0],          # 0: 光心
        [-w, -h, z],        # 1: 左上
        [w, -h, z],         # 2: 右上
        [w, h, z],          # 3: 右下
        [-w, h, z]          # 4: 左下
    ]).T * scale # (3, 5)

    # 变换到世界坐标系
    # P_world = R * P_local + T
    # c2w 矩阵是 4x4, 我们取前 3x3 作为 R, 第 4 列作为 T
    # 或者直接用矩阵乘法 (需转为齐次坐标)
    
    # 简单做法: 旋转 + 平移
    R = c2w[:3, :3]
    t = c2w[:3, 3].reshape(3, 1)
    
    points_world = (R @ points_local) + t
    
    # 绘制线条连接成金字塔
    # 0-1, 0-2, 0-3, 0-4 (四条棱)
    # 1-2-3-4-1 (底面)
    pw = points_world
    
    # 画光心到四角
    for i in range(1, 5):
        ax.plot([pw[0,0], pw[0,i]], [pw[1,0], pw[1,i]], [pw[2,0], pw[2,i]], color=color, linewidth=0.5)
    
    # 画底面框 (指示方向)
    # 为了区分上下，底面我们用不同颜色或者加粗顶边
    ax.plot([pw[0,1], pw[0,2]], [pw[1,1], pw[1,2]], [pw[2,1], pw[2,2]], color='red', linewidth=1) # 顶部(对于OpenCV是上边)
    ax.plot([pw[0,2], pw[0,3]], [pw[1,2], pw[1,3]], [pw[2,2], pw[2,3]], color=color, linewidth=0.5)
    ax.plot([pw[0,3], pw[0,4]], [pw[1,3], pw[1,4]], [pw[2,3], pw[2,4]], color=color, linewidth=0.5)
    ax.plot([pw[0,4], pw[0,1]], [pw[1,4], pw[1,1]], [pw[2,4], pw[2,1]], color=color, linewidth=0.5)

def verify_transforms():
    if not os.path.exists(JSON_PATH):
        print("❌ 找不到 transforms.json")
        return

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    frames = data['frames']
    print(f"正在绘制 {len(frames)} 个相机位姿...")
    
    positions = []

    for i, frame in enumerate(frames):
        # 读取 4x4 矩阵
        c2w = np.array(frame['transform_matrix'])
        positions.append(c2w[:3, 3])
        
        # 为了不让图太乱，每隔几帧画一个相机
        if i % 5 == 0:
            # 起点用绿色，终点用红色，中间用蓝色
            color = 'blue'
            if i == 0: color = 'green'
            if i == len(frames) - 1: color = 'red'
            
            plot_camera(ax, c2w, scale=0.3, color=color)

    # 绘制轨迹连线
    pos_arr = np.array(positions)
    ax.plot(pos_arr[:,0], pos_arr[:,1], pos_arr[:,2], 'k--', alpha=0.5, label='Trajectory')

    # 设置轴标签
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(f"Transforms Verification\nStart(Green) -> End(Red)\nRed Line on Camera = Top of Image")
    
    # 强制比例一致 (关键！否则L型看着像直的)
    # Matplotlib 3D axis scaling is tricky, creating a bounding box
    max_range = np.array([pos_arr[:,0].max()-pos_arr[:,0].min(), 
                          pos_arr[:,1].max()-pos_arr[:,1].min(), 
                          pos_arr[:,2].max()-pos_arr[:,2].min()]).max() / 2.0
    mid_x = (pos_arr[:,0].max()+pos_arr[:,0].min()) * 0.5
    mid_y = (pos_arr[:,1].max()+pos_arr[:,1].min()) * 0.5
    mid_z = (pos_arr[:,2].max()+pos_arr[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.legend()
    plt.savefig("check_transforms.png")
    print("✅ 验证图已保存至: check_transforms.png")

if __name__ == "__main__":
    verify_transforms()