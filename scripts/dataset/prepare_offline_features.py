import os
import glob
import h5py
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

def prepare_offline_data(raw_h5_dir, output_dir, max_crops_to_save=32, crop_size=32, valid_ratio=0.5):
    os.makedirs(output_dir, exist_ok=True)
    h5_files = glob.glob(os.path.join(raw_h5_dir, "**", "*.h5"), recursive=True)
    
    threshold_count = (crop_size * crop_size) * valid_ratio
    kernel = torch.ones((1, 1, crop_size, crop_size), dtype=torch.float32)
    
    print(f"🚀 开始离线榨汁！共发现 {len(h5_files)} 个原始 H5 文件。")
    print(f"💾 输出目录: {output_dir} (将保存为极致纯净的 H5 格式)")
    
    for h5_path in tqdm(h5_files, desc="Processing WSIs"):
        file_name = os.path.basename(h5_path)
        save_path = os.path.join(output_dir, file_name) 
        
        if os.path.exists(save_path):
            continue
            
        try:
            with h5py.File(h5_path, 'r') as f:
                coords = f['coords_patching'][:]
                features = f['features'][:] 
                
            if features.ndim == 3 and features.shape[0] == 1:
                features = features[0]
            
            N, D = features.shape
            
            x_unique = np.unique(coords[:, 0])
            x_diffs = np.diff(np.sort(x_unique))
            valid_diffs = x_diffs[x_diffs > 0]
            patch_size = int(np.min(valid_diffs)) if len(valid_diffs) > 0 else 256
            
            grid_coords = coords // patch_size
            min_x, min_y = grid_coords.min(axis=0)
            max_x, max_y = grid_coords.max(axis=0)
            W, H = int(max_x - min_x + 1), int(max_y - min_y + 1)
            
            index_grid = np.full((H, W), -1, dtype=np.int32)
            norm_coords = (grid_coords - [min_x, min_y]).astype(int)
            index_grid[norm_coords[:, 1], norm_coords[:, 0]] = np.arange(N)
            
            pad_h = max(0, crop_size - H)
            pad_w = max(0, crop_size - W)
            if pad_h > 0 or pad_w > 0:
                index_grid = np.pad(index_grid, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=-1)
                
            valid_mask = (index_grid != -1)
            mask_tensor = torch.from_numpy(valid_mask).float().unsqueeze(0).unsqueeze(0)
            
            valid_sums = F.conv2d(mask_tensor, kernel, stride=crop_size).squeeze(0).squeeze(0)
            valid_positions = torch.nonzero(valid_sums >= threshold_count)
            
            if len(valid_positions) == 0:
                valid_positions = torch.nonzero(valid_sums == valid_sums.max())
                
            if len(valid_positions) > max_crops_to_save:
                indices = torch.randperm(len(valid_positions))[:max_crops_to_save]
                valid_positions = valid_positions[indices]
                
            M = len(valid_positions)
            crops_feat = np.zeros((M, crop_size * crop_size, D), dtype=np.float16) 
            
            for i, pos in enumerate(valid_positions):
                y_start = pos[0].item() * crop_size
                x_start = pos[1].item() * crop_size
                crop_idx = index_grid[y_start : y_start + crop_size, x_start : x_start + crop_size].flatten()
                
                valid_locs = np.where(crop_idx != -1)[0]
                real_indices = crop_idx[valid_locs]
                
                if len(real_indices) > 0:
                    crops_feat[i, valid_locs, :] = features[real_indices].astype(np.float16)
                    
            # ================= 🚀 核心修改：保存为 HDF5 =================
            with h5py.File(save_path, 'w') as f_out:
                f_out.create_dataset('features', data=crops_feat, dtype=np.float16, compression='lzf')
            # ============================================================
            
        except Exception as e:
            print(f"\n❌ 处理 {file_name} 时出错: {e}")

if __name__ == "__main__":
    # 请替换为你的实际路径
    RAW_TRAIN_DIR = "/data2/mengzibing/Amedicine/dataset/tcga-feature/"
    OFFLINE_TRAIN_DIR = "/data2/mengzibing/Amedicine/dataset/dataset_offline/"
    prepare_offline_data(RAW_TRAIN_DIR, OFFLINE_TRAIN_DIR)