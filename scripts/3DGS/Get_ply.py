import omni.usd
from pxr import Usd, UsdGeom, Vt,Semantics
import numpy as np
import trimesh
import os
import random
import json

# ================= 配置区域 =================
# 保存路径
save_path = "/root/gpufree-data/3Dreconstruction/dataset/ply/Global_map/Global_Map_no_label.ply"
ply_output_path = "/root/gpufree-data/3Dreconstruction/dataset/ply/Global_map/Global_Map_dense.ply" 
json_output_path="/root/gpufree-data/3Dreconstruction/dataset/ply/Global_map/semantic_mapping.json"
# 采样密度 (每平方米产生多少个点)
# 建议：1000~5000 适合做 SLAM/3DGS 初始化
# 如果场景很大，适当调小，否则文件会很大
SAMPLE_DENSITY = 100

# 最小点数 (防止小物体没点)
MIN_POINTS_PER_MESH = 50
# ===========================================
# 预定义语义映射 (可选)
# 如果你想强制规定 ID，可以在这里写。否则脚本会自动生成。
# 格式: {"class_name": id_int}
FORCE_MAPPING = {
    "unlabeled": 0,
    "background": 0
}
# ===========================================
def get_world_transform_matrix(prim):
    """获取 Numpy 格式的世界变换矩阵 (4x4)"""
    xform = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default()
    world_transform = xform.ComputeLocalToWorldTransform(time)
    return np.array(world_transform).T # 转置以适配 Numpy


def get_semantic_data_isaac(prim):
    """
    递归查找语义标签：如果当前 Prim 没有，就向上查找父级。
    """
    # 设定查找上限，防止死循环 (例如只向上查 5 层)
    max_depth = 5
    current_prim = prim
    
    for _ in range(max_depth):
        # 1. 如果已经到了根节点，停止
        if not current_prim.IsValid() or current_prim.IsPseudoRoot():
            #print("已到达根节点，停止查找语义标签。")
            break
            
        # 2. 检查是否有语义 API (这是最标准的判断方式)
        if current_prim.HasAPI(Semantics.SemanticsAPI, "Semantics"):
            #print("找到 SemanticsAPI，尝试获取语义标签...")
            sem_api = Semantics.SemanticsAPI.Get(current_prim, "Semantics")
            #print("SemanticsAPI:", sem_api)
            # 获取所有绑定的语义 (一个物体可能有多个语义，我们取第一个 type 为 class 的)
            # SemanticsAPI 存储结构比较复杂，通常包含 direct properties
            
            # 属性名为: semantic:Semantics:params:semanticData
            type_attr = current_prim.GetAttribute("semantic:Semantics:params:semanticType")
            data_attr = current_prim.GetAttribute("semantic:Semantics:params:semanticData")

            #print("语义有效属性：",type_attr.IsValid(), data_attr.IsValid())
            if type_attr.IsValid() and data_attr.IsValid():
                if type_attr.Get() == "class":
                    #print(f"✅ 成功找到语义! Prim: {current_prim.GetPath()}, Label: {data_attr.Get()}")
                    return data_attr.Get()
            
            # --- 方法 B: 针对新版 USD Semantics Schema 的标准写法 ---
            # 如果上面的不行，尝试用 Schema 接口获取
            # 注意: 这里简化处理，只针对简单的 class 标签
            pass 

        # 3. 如果当前节点没找到，在这个循环里将 current_prim 指向父级
        #print("当前 Prim 未找到语义标签，尝试向上查找父级...")
        current_prim = current_prim.GetParent()
        
    return "unlabeled"

def process_scene():
    stage = omni.usd.get_context().get_stage()
    
    all_points = []
    all_colors = []
    all_labels = [] # 存储整数 ID
    
    # 动态构建映射表
    label_map = FORCE_MAPPING.copy()
    next_id = max(label_map.values()) + 1 if label_map else 1
    
    print(f"开始语义采样...")
    
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            #print("处理 Mesh:", prim.GetPath())
            mesh_prim = UsdGeom.Mesh(prim)
            prim_path = prim.GetPath()
            # --- A. 获取语义标签 ---
            # 优先检查当前 Prim，如果没有，往上找一级 (通常 Mesh 是 Xform 的子级)
            sem_name = get_semantic_data_isaac(prim)
            #print("语义标签:", sem_name)
            if sem_name == "unlabeled":
                #print("尝试从父级获取语义...")
                sem_name = get_semantic_data_isaac(prim.GetParent())                
                
                
            # 注册 ID
            if sem_name not in label_map:
                label_map[sem_name] = next_id
                next_id += 1
            
            current_id = label_map[sem_name]
            
            # --- B. 获取颜色 (DisplayColor) ---
            # 新的策略: 根据 current_id 生成一个伪随机颜色
            # 这样同一个类别的物体颜色永远一样
            import random
            random.seed(current_id) # 确保颜色固定
            
            # 生成鲜艳的 RGB (避免太黑)
            r = random.randint(50, 255)
            g = random.randint(50, 255)
            b = random.randint(50, 255)
            
            base_color = [r, g, b]
            #print(f"  > 语义: {sem_name} (ID: {current_id}) -> 分配颜色: {base_color}")
            
            # --- C. 构建 Trimesh 并采样 ---
            points_attr = mesh_prim.GetPointsAttr().Get()
            if not points_attr: continue
            vertices = np.array(points_attr)
            
            face_indices = mesh_prim.GetFaceVertexIndicesAttr().Get()
            face_counts = mesh_prim.GetFaceVertexCountsAttr().Get()
            if not face_indices or not face_counts: continue
            
            indices = np.array(face_indices)
            counts = np.array(face_counts) # 转为 numpy 数组以便比较

            # =========== [修复核心] 面索引处理逻辑 ===========
            faces = None
            try:
                if np.all(counts == 3):
                    faces = indices.reshape(-1, 3)
                elif np.all(counts == 4):
                    # 修复：增加对四边形的支持
                    faces = indices.reshape(-1, 4)
                else:
                    # 混合面处理：Trimesh 其实可以直接根据 counts 解析，但需要更复杂的构建方式
                    # 这里尝试最简单的 fallback，如果不规则，可能会失败
                    print(f"⚠️ 跳过复杂拓扑 Mesh (非纯三角/四边形): {prim_path}")
                    continue
            except Exception as e:
                print(f"❌ Reshape Error on {prim_path}: {e}")
                continue
            # ===================================================

            try:
                # Trimesh 会自动将 Quads 处理为 Triangles 用于计算面积
                tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                
                # 应用世界变换
                mat = get_world_transform_matrix(prim)
                tm.apply_transform(mat)
                
                # 计算采样数
                area = tm.area
                n_samples = max(int(area * SAMPLE_DENSITY), MIN_POINTS_PER_MESH)
                
                # 采样
                samples, _ = trimesh.sample.sample_surface(tm, n_samples)
                
                if len(samples) > 0:
                    all_points.append(samples)
                    
                    # 填充颜色
                    c_stack = np.tile(base_color, (len(samples), 1))
                    all_colors.append(c_stack)
                    
                    # 填充语义 ID
                    l_stack = np.full((len(samples), 1), current_id, dtype=np.uint16)
                    all_labels.append(l_stack)
                else:
                    pass
                    # print(f"Mesh {prim_path} 面积太小，未生成点")
                    
            except Exception as e:
                print(f"Error processing mesh {prim_path}: {e}")
                pass

    if not all_points:
        print("❌ 未提取到点云数据。请检查上面的错误日志。")
        return

    # --- D. 合并数据 ---
    final_points = np.vstack(all_points).astype(np.float32)
    final_colors = np.vstack(all_colors).astype(np.uint8)
    final_labels = np.vstack(all_labels).astype(np.uint16) 

    if final_colors.shape[1] > 3:
        final_colors = final_colors[:, :3]

    print(f"采样完成: {len(final_points)} points.")
    print(f"检测到的语义类别: {label_map}")

    # --- E. 保存 JSON ---
    with open(json_output_path, 'w') as f:
        save_map = {str(v): k for k, v in label_map.items()}
        json.dump(save_map, f, indent=4)

    # --- F. 保存 PLY ---
    with open(ply_output_path, 'wb') as f:
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(final_points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property ushort label
end_header
"""
        f.write(header.encode('ascii'))
        
        dt = np.dtype([
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('label', 'u2')
        ])
        
        vertex_data = np.empty(len(final_points), dtype=dt)
        vertex_data['x'] = final_points[:, 0]
        vertex_data['y'] = final_points[:, 1]
        vertex_data['z'] = final_points[:, 2]
        vertex_data['red'] = final_colors[:, 0]
        vertex_data['green'] = final_colors[:, 1]
        vertex_data['blue'] = final_colors[:, 2]
        vertex_data['label'] = final_labels[:, 0]
        
        f.write(vertex_data.tobytes())

    print(f"成功保存: {ply_output_path}")

def save_dense_ply(filepath):
    stage = omni.usd.get_context().get_stage()
    all_points = []
    all_colors = []
    
    print(f"开始稠密采样 (密度: {SAMPLE_DENSITY} points/m^2)...")
    
    # 遍历场景
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh_prim = UsdGeom.Mesh(prim)
            
            # --- 1. 获取几何数据 ---
            # 顶点
            points_attr = mesh_prim.GetPointsAttr().Get()
            if points_attr is None: continue
            vertices = np.array(points_attr)
            
            # 面索引 (Face Indices)
            face_indices = mesh_prim.GetFaceVertexIndicesAttr().Get()
            face_counts = mesh_prim.GetFaceVertexCountsAttr().Get()
            
            if face_indices is None or face_counts is None: continue
            
            # --- 2. 转换为 Trimesh 对象 ---
            # Trimesh 需要 (N, 3) 的 faces。
            # USD 可能是混合的多边形 (Quads + Triangles)。
            # 这里我们需要手动处理索引，或者简单起见，假设是三角面或者让 trimesh 尝试解析
            # 为保证鲁棒性，我们先构建 trimesh，再应用变换
            
            # 转换 face_indices 为 numpy
            indices = np.array(face_indices)
            
            # 简易处理：如果全是三角形 (counts全是3)
            counts = np.array(face_counts)
            if np.all(counts == 3):
                faces = indices.reshape(-1, 3)
            elif np.all(counts == 4):
                # 如果是四边形，Trimesh 也能处理，或者我们简单切分
                faces = indices.reshape(-1, 4)
            else:
                # 混合多边形比较麻烦，这里尝试直接传给 trimesh (它有容错机制)
                # 如果报错，通常是因为网格拓扑太奇怪，这里跳过
                print(f"跳过复杂拓扑网格: {prim.GetPath()}")
                continue

            # 创建 Trimesh 对象 (Local Space)
            try:
                tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            except Exception as e:
                print(f"Trimesh 构建失败 {prim.GetPath()}: {e}")
                continue

            # --- 3. 应用世界坐标变换 ---
            # 我们直接变换 Mesh 的顶点，这样采样就在世界坐标系下了
            mat = get_world_transform_matrix(prim)
            tm.apply_transform(mat)
            
            # --- 4. 计算采样数量 ---
            area = tm.area
            n_samples = int(area * SAMPLE_DENSITY)
            n_samples = max(n_samples, MIN_POINTS_PER_MESH) # 保证至少有几个点
            
            # --- 5. 执行泊松盘采样或随机采样 ---
            # sample_surface 返回 (points, face_index)
            try:
                samples, _ = trimesh.sample.sample_surface(tm, n_samples)
            except Exception as e:
                # 极少数情况下面积计算错误
                continue
            
            if len(samples) > 0:
                all_points.append(samples)
                
                # --- 6. 颜色处理 ---
                # 获取物体颜色
                color_attr = mesh_prim.GetDisplayColorAttr().Get()
                if color_attr and len(color_attr) > 0:
                    c = np.array(color_attr[0]) * 255
                    # 广播颜色到所有采样点
                    point_colors = np.tile(c, (len(samples), 1))
                else:
                    point_colors = np.full((len(samples), 3), 200) # 默认灰
                
                all_colors.append(point_colors)

    if not all_points:
        print("未提取到任何点，请检查场景是否包含 Mesh。")
        return

    # --- 7. 合并并保存 ---
    final_points = np.vstack(all_points).astype(np.float32)
    final_colors = np.vstack(all_colors).astype(np.uint8)
    
    print(f"采样完成！共生成 {len(final_points)} 个点。")
    print(f"正在保存至 {filepath} ...")
    
    # 写入 PLY
    with open(filepath, 'wb') as f:
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(final_points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        f.write(header.encode('ascii'))
        data = np.hstack([final_points.view(np.uint8), final_colors])
        f.write(data.tobytes())
        
    print("保存成功！")


#save_dense_ply(save_path)
process_scene()