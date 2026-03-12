import h5py
import torch
import numpy as np
import time
from torch.utils.data import Dataset
import torch.nn.functional as F

class PathNEPAFeatureDataset(Dataset):
    def __init__(self, h5_file_list, num_crops=64, crop_size=32, valid_ratio=0.5, mask_ratio=0.4):
        self.h5_files = h5_file_list
        self.num_crops = num_crops
        self.crop_size = crop_size
        self.valid_ratio = valid_ratio
        self.mask_ratio = mask_ratio
        self.threshold_count = (crop_size * crop_size) * valid_ratio
        
        # --- 初始化日志 ---
        print("="*50)
        print(f"📦 [Dataset Init] 初始化 PathNEPAFeatureDataset")
        print(f"   -> 找到 .h5 文件总数: {len(self.h5_files)}")
        print(f"   -> 每次切图数量: {self.num_crops} 张")
        print(f"   -> 单张切图尺寸: {self.crop_size}x{self.crop_size}")
        print(f"   -> 有效区域阈值: >= {self.threshold_count} 个有效 Patch ({valid_ratio*100}%)")
        print(f"   -> 预训练遮挡率: {mask_ratio*100}%")
        print("="*50)

    def __len__(self):
        return len(self.h5_files)

    def _infer_patch_size(self, coords):
        x_unique = np.unique(coords[:, 0])
        x_diffs = np.diff(np.sort(x_unique))
        x_diffs = x_diffs[x_diffs > 0]
        return int(np.min(x_diffs)) if len(x_diffs) > 0 else 256

    def __getitem__(self, idx):
        start_time = time.time()
        
        h5_path = self.h5_files[idx]
        file_name = h5_path.split('/')[-1]  # 提取文件名用于打印
        
        print(f"🔄 [DataLoader] 正在处理 第 {idx+1}/{len(self.h5_files)} 个文件: {file_name}")
        
        # 1. 记录读 H5 的耗时
        t0 = time.time()
        with h5py.File(h5_path, 'r') as f:
            coords = f['coords_patching'][:] 
            features = f['features'][0]       
        t1 = time.time()
        
        D = features.shape[1] # (N,1536)
        patch_size = self._infer_patch_size(coords)
        grid_coords = coords // patch_size 
        
        min_x, min_y = grid_coords.min(axis=0)
        max_x, max_y = grid_coords.max(axis=0)
        
        W = int(max_x - min_x + 1)
        H = int(max_y - min_y + 1)
        
        dense_grid = np.zeros((H, W, D), dtype=np.float32)
        valid_mask = np.zeros((H, W), dtype=bool) 
        
        norm_coords = (grid_coords - [min_x, min_y]).astype(int)
        dense_grid[norm_coords[:, 1], norm_coords[:, 0]] = features
        valid_mask[norm_coords[:, 1], norm_coords[:, 0]] = True

        pad_h = max(0, self.crop_size - H)
        pad_w = max(0, self.crop_size - W)
        if pad_h > 0 or pad_w > 0:
            valid_mask = np.pad(valid_mask, ((0, pad_h), (0, pad_w)), mode='constant')
            dense_grid = np.pad(dense_grid, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            
        mask_tensor = torch.from_numpy(valid_mask).float().unsqueeze(0).unsqueeze(0)
        kernel = torch.ones((1, 1, self.crop_size, self.crop_size), dtype=torch.float32)
        
        # 2. 记录卷积滑动窗口计算的耗时
        t2 = time.time()
        valid_sums = F.conv2d(mask_tensor, kernel, stride=self.crop_size, padding=0).squeeze(0).squeeze(0)
        valid_positions = torch.nonzero(valid_sums >= self.threshold_count)
        t3 = time.time()
        
        if len(valid_positions) == 0:
            print(f"   ⚠️ [Dataset Warning] {file_name} 找不到 >= {self.valid_ratio*100}% 的区域，自动降级采样最大面积区域。")
            max_val = valid_sums.max()
            valid_positions = torch.nonzero(valid_sums == max_val)

        sampled_indices = torch.randint(0, len(valid_positions), (self.num_crops,))
        sampled_positions = valid_positions[sampled_indices]

        crops_feat = []
        crops_masked_pos = []
        
        # 3. 记录切图与生成 Mask 的耗时
        t4 = time.time()
        for pos in sampled_positions:
            y, x = pos[0].item(), pos[1].item()
            crop_f = dense_grid[y : y + self.crop_size, x : x + self.crop_size].reshape(-1, D)
            crop_m = valid_mask[y : y + self.crop_size, x : x + self.crop_size].reshape(-1)
            
            valid_indices = np.where(crop_m)[0]
            num_mask = int(len(valid_indices) * self.mask_ratio)
            masked_pos = np.zeros(self.crop_size * self.crop_size, dtype=bool)
            
            if num_mask > 0:
                masked_idx = np.random.choice(valid_indices, num_mask, replace=False)
                masked_pos[masked_idx] = True
                
            crops_feat.append(crop_f)
            crops_masked_pos.append(masked_pos)

        input_features = torch.tensor(np.stack(crops_feat))
        t5 = time.time()
        
        total_time = t5 - start_time
        print(f"   ✅ [切片完成] 耗时: {total_time:.3f}秒 (读H5: {t1-t0:.3f}s | 卷积: {t3-t2:.3f}s | 裁剪: {t5-t4:.3f}s)")
        
        return {
            "input_features": input_features,                         # (64, 1024, 1536)
            "label_features": input_features.clone(),                 # 预测目标
            "bool_masked_pos": torch.tensor(np.stack(crops_masked_pos)) # (64, 1024)
        }