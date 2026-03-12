import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
from torch.utils.data import Dataset

class PathNEPAFeatureDataset_Test(Dataset):
    def __init__(self, h5_file_list, num_crops=64, crop_size=32, valid_ratio=0.5):
        self.h5_files = h5_file_list
        self.num_crops = num_crops
        self.crop_size = crop_size
        self.threshold_count = (crop_size * crop_size) * valid_ratio

    def __len__(self):
        return len(self.h5_files)

    def _infer_patch_size(self, coords):
        x_unique = np.unique(coords[:, 0])
        x_diffs = np.diff(np.sort(x_unique))
        x_diffs = x_diffs[x_diffs > 0]
        return int(np.min(x_diffs)) if len(x_diffs) > 0 else 256

    def __getitem__(self, idx):
        h5_path = self.h5_files[idx]
        
        with h5py.File(h5_path, 'r') as f:
            coords = f['coords_patching'][:] 
            features = f['features'][0]       

        D = features.shape[1]
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
            
        mask_tensor = torch.from_numpy(valid_mask).float().unsqueeze(0).unsqueeze(0)
        kernel = torch.ones((1, 1, self.crop_size, self.crop_size), dtype=torch.float32)
        
        valid_sums = F.conv2d(mask_tensor, kernel).squeeze(0).squeeze(0)
        valid_positions = torch.nonzero(valid_sums >= self.threshold_count)
        
        if len(valid_positions) == 0:
            max_val = valid_sums.max()
            valid_positions = torch.nonzero(valid_sums == max_val)

        sampled_indices = torch.randint(0, len(valid_positions), (self.num_crops,))
        sampled_positions = valid_positions[sampled_indices]

        crops_mask = []
        for pos in sampled_positions:
            y, x = pos[0].item(), pos[1].item()
            crop_m = valid_mask[y : y + self.crop_size, x : x + self.crop_size]
            crops_mask.append(crop_m)

        return {
            "global_mask": valid_mask,
            "sampled_positions": sampled_positions.numpy(),
            "crops_mask": crops_mask,
            "crop_size": self.crop_size
        }

if __name__ == "__main__":
    # 替换为你的 h5 文件绝对路径
    test_file = '/data2/mengzibing/medicine/datasets/dataset_o/Sub-typing/BRCA/TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.h5'
    
    dataset = PathNEPAFeatureDataset_Test([test_file], num_crops=64, crop_size=32, valid_ratio=0.5)
    
    data = dataset[0]
    global_mask = data["global_mask"]
    sampled_positions = data["sampled_positions"]
    crops_mask = data["crops_mask"]
    crop_size = data["crop_size"]
    
    fig = plt.figure(figsize=(20, 8))
    
    # 1. 全局图
    ax1 = plt.subplot(1, 3, (1, 2))
    ax1.imshow(global_mask, cmap='Blues', origin='upper')
    ax1.set_title("Global WSI Valid Mask with 64 Sampled 32x32 Crops", fontsize=16)
    
    for pos in sampled_positions:
        y, x = pos[0], pos[1]
        rect = patches.Rectangle((x, y), crop_size, crop_size, linewidth=1.5, edgecolor='red', facecolor='none', alpha=0.6)
        ax1.add_patch(rect)
        
    # 2. 局部图
    ax2 = plt.subplot(2, 3, 3)
    ax2.imshow(crops_mask[0], cmap='Blues', origin='upper')
    valid_ratio_0 = crops_mask[0].sum() / (crop_size * crop_size) * 100
    ax2.set_title(f"Crop 1 (Valid: {valid_ratio_0:.1f}%)", fontsize=12)
    
    ax3 = plt.subplot(2, 3, 6)
    ax3.imshow(crops_mask[1], cmap='Blues', origin='upper')
    valid_ratio_1 = crops_mask[1].sum() / (crop_size * crop_size) * 100
    ax3.set_title(f"Crop 2 (Valid: {valid_ratio_1:.1f}%)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("dataset_crops_visualization.png", dpi=300)
    print("可视化图像已成功保存为 dataset_crops_visualization.png")