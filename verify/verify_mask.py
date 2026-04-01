import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt

# ================= 配置 =================
RGB_DIR = "/root/gpufree-data/3Dreconstruction/dataset/sequence_1/Cam_Left_07/rgb"
MASK_DIR = "/root/gpufree-data/3Dreconstruction/dataset/sequence_1/Cam_Left_07/semantic_segmentation"
FRAME_ID = "0237"  # 检查第 0 帧

# ================= 逻辑 =================
def verify_mask_overlay():
    # 1. 读取 RGB
    rgb_path = os.path.join(RGB_DIR, f"rgb_{FRAME_ID}.png")
    rgb = cv2.imread(rgb_path)
    
    # 2. 读取 Instance Mask (Raw ID)
    # 注意：Replicator 输出的 Raw Mask 通常是 uint32 或 uint16，存为 png 时可能看起来是黑的
    mask_path = os.path.join(MASK_DIR, f"semantic_segmentation_{FRAME_ID}.png")
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) # 必须用 UNCHANGED 读取原始值
    
    if rgb is None or mask is None:
        print("❌ 未找到文件")
        return

    # 3. 简单的着色 (为了可视化)
    # 将 ID 映射为随机颜色
    np.random.seed(42)
    unique_ids = np.unique(mask)
    colored_mask = np.zeros_like(rgb)
    
    for uid in unique_ids:
        if uid == 0: continue # 背景保持黑色
        color = np.random.randint(0, 255, (3,)).tolist()
        colored_mask[mask == uid] = color

    # 4. 叠加 (Add Weighted)
    # RGB * 0.6 + Mask * 0.4
    overlay = cv2.addWeighted(rgb, 0.6, colored_mask, 0.4, 0)
    
    # 5. 保存结果用于检查
    out_path = "verify_mask_result.png"
    cv2.imwrite(out_path, overlay)
    print(f"✅ Mask 验证图已保存至: {out_path}。请打开查看物体边缘是否完美贴合。")

if __name__ == "__main__":
    verify_mask_overlay()