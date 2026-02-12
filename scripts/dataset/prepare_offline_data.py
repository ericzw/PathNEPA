import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
import glob

def process_offline(h5_path, save_dir, target_size=32):
    try:
        slide_id = os.path.basename(h5_path).replace('.h5', '')
        save_path = os.path.join(save_dir, f"{slide_id}.pt")

        # 1. 读取数据
        with h5py.File(h5_path, 'r') as f:
            # 适配 (1, N, 1536) 结构
            features = f['features'][0][:]  
            coords = f['coords'][0][:]
        
        if len(features) == 0:
            return

        # 2. 生成 32x32 Grid
        c_min, c_max = coords.min(axis=0), coords.max(axis=0)
        # 避免除以 0
        denom = np.where((c_max - c_min) == 0, 1.0, (c_max - c_min))
        # 归一化并映射到 0~31
        norm_coords = (coords - c_min) / denom * (target_size - 1)
        grid_indices = norm_coords.astype(int)

        grid_features = torch.zeros((target_size, target_size, 1536))
        valid_mask = torch.zeros((target_size, target_size), dtype=torch.bool)

        for i in range(len(features)):
            # 注意：病理坐标通常是 (x, y)，x对应列(W)，y对应行(H)
            gx, gy = grid_indices[i]
            # 填入特征 (gy 是行，gx 是列)
            grid_features[gy, gx] = torch.from_numpy(features[i])
            valid_mask[gy, gx] = True

        # 3. 保存数据
        torch.save({
            'grid_32': grid_features.view(-1, 1536),        # (1024, 1536)
            'mask_32': valid_mask.view(-1),                # (1024,)
            'full_seq': torch.from_numpy(features).float(), # (N, 1536)
            'full_coords': torch.from_numpy(coords).float() # (N, 2)
        }, save_path)
        
    except Exception as e:
        print(f"\n 处理 {h5_path} 出错: {e}")

def main():
    # --- 配置区域 ---
    H5_DIR = "/root/autodl-tmp/nepa/datas/TCGA/TCGA-BLCA"
    SAVE_DIR = "/root/autodl-tmp/nepa/datas/TCGA/TCGA-BLCA-PT" 
    TARGET_GRID = 32
    # ----------------

    # 创建输出文件夹
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 获取所有 h5 文件
    h5_files = glob.glob(os.path.join(H5_DIR, "*.h5"))
    print(f" 找到 {len(h5_files)} 个 H5 文件")

    # 开始遍历处理
    for h5_file in tqdm(h5_files, desc="Converting H5 to PT"):
        process_offline(h5_file, SAVE_DIR, TARGET_GRID)

    print(f"\n 转换完成！数据保存在: {SAVE_DIR}")

if __name__ == "__main__":
    main()