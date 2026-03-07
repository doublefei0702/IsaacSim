import json
import numpy as np
import os
import glob
from pathlib import Path

# ================= 配置区域 =================
# 输入文件夹 (Replicator 生成的 camera_params 所在目录)
INPUT_DIR = "/root/gpufree-data/3Dreconstruction/dataset/sequence_1/Cam_Right_07/camera_params"

# 输出文件夹 (提取后的干净数据存放处)
OUTPUT_DIR = "/root/gpufree-data/3Dreconstruction/dataset/sequence_1/Cam_Right_07/processed_params"

# ================= 核心处理逻辑 =================

def process_single_json(file_path):
    """读取单个 Isaac Sim JSON 并转换为 SLAM 格式"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 1. 基础信息
    W = data['renderProductResolution'][0]
    H = data['renderProductResolution'][1]
    
    # 2. 计算内参 (Intrinsics)
    # Isaac Sim: f_pixel = image_size * (f_mm / aperture_mm)
    f_mm = data['cameraFocalLength']
    aperture_w = data['cameraAperture'][0]
    aperture_h = data['cameraAperture'][1]

    fx = W * (f_mm / aperture_w)
    fy = H * (f_mm / aperture_h)
    cx = W / 2.0
    cy = H / 2.0

    # 3. 计算外参 (Extrinsics)
    # Replicator 输出: 4x4 Row-Major, View Matrix (World -> Camera, OpenGL coords)
    view_flat = data['cameraViewTransform']
    isaac_view_matrix = np.array(view_flat).reshape(4, 4)

    # 计算 Pose (Camera -> World)
    isaac_pose_matrix = np.linalg.inv(isaac_view_matrix)

    # 4. 坐标系转换 
    # Isaac (OpenGL): X右, Y上, -Z前
    # SLAM (OpenCV):  X右, Y下, +Z前
    # 变换矩阵: 绕 X 轴旋转 180 度
    T_isaac_to_opencv = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])

    # 转换后的 Pose (C2W) -> 用于 SLAM 轨迹 / 3DGS
    c2w_opencv = isaac_pose_matrix @ T_isaac_to_opencv
    
    # 转换后的 View (W2C) -> 用于投影
    w2c_opencv = T_isaac_to_opencv @ isaac_view_matrix

    return {
        "file_path": str(file_path),
        "w": W,
        "h": H,
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "k1": 0, "k2": 0, "p1": 0, "p2": 0, # 仿真默认为无畸变针孔
        "camera_model": "PINHOLE",
        # 保存为列表以便 JSON 序列化
        "transform_matrix": c2w_opencv.tolist(), # 习惯存 C2W
        "view_matrix": w2c_opencv.tolist()       # 备用
    }

def main():
    # 1. 准备目录
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 错误: 输入目录不存在: {INPUT_DIR}")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. 查找文件
    # 匹配 camera_params_*.json
    search_pattern = os.path.join(INPUT_DIR, "camera_params_*.json")
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        print(f"⚠️ 警告: 在 {INPUT_DIR} 下没有找到 json 文件")
        return

    print(f"📂 找到 {len(files)} 个文件，开始处理...")
    
    # 用于生成总的 transforms.json (Nerfstudio/3DGS 标准格式)
    frames_summary = []

    # 3. 循环处理
    for i, file_path in enumerate(files):
        try:
            # 提取数据
            res = process_single_json(file_path)
            
            # 构建文件名 (保持帧号对应)
            # 例如: camera_params_0000.json -> frame_0000.json
            frame_name = Path(file_path).stem.replace("camera_params_", "frame_")
            out_name = f"{frame_name}.json"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            
            # 保存单个文件的结果
            with open(out_path, 'w') as f:
                json.dump(res, f, indent=4)
            
            # 添加到汇总列表 (适配 Nerfstudio 格式)
            frames_summary.append({
                "file_path": f"./images/{frame_name.replace('frame_', 'rgb_')}.png", # 假设对应的图片路径
                "transform_matrix": res["transform_matrix"]
            })
            
            if i % 10 == 0:
                print(f" -> 已处理: {frame_name}")
                
        except Exception as e:
            print(f"❌ 处理 {file_path} 失败: {e}")

    # 4. 保存汇总文件 transforms.json
    # 这是绝大多数 3DGS 训练代码直接支持的格式
    summary_data = {
        "camera_model": "PINHOLE",
        "fl_x": res["fl_x"], # 假设焦距变焦不大，取最后一帧的
        "fl_y": res["fl_y"],
        "cx": res["cx"],
        "cy": res["cy"],
        "w": res["w"],
        "h": res["h"],
        "frames": frames_summary
    }
    
    summary_path = os.path.join(OUTPUT_DIR, "transforms.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=4)

    print("-" * 30)
    print(f"✅ 处理完成！")
    print(f"1. 单帧 JSON 已保存至: {OUTPUT_DIR}")
    print(f"2. 汇总训练文件已保存至: {summary_path}")

if __name__ == "__main__":
    main()