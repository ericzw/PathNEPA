import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os
import glob

import os
import torch
import glob
from torch.utils.data import Dataset

import os
import torch
import glob
import pandas as pd
from torch.utils.data import Dataset

class DownstreamNepaDataset(Dataset):
    def __init__(self, data_dir, label_file):
        """
        Args:
            data_dir: 存放 .pt 文件的文件夹 (例如 .../TCGA-BLCA-PT)
            label_file: 包含 'slide_id' 和 'label' 两列的 CSV 文件路径
        """
        self.data_dir = data_dir
        
        # 1. 读取标签表 (CSV)
        # 假设 CSV 长这样:
        # slide_id,       label
        # TCGA-BLCA-001,  0
        # TCGA-BLCA-002,  1
        df = pd.read_csv(label_file)
        
        # 创建一个快速查找字典: {'TCGA-BLCA-001': 0, ...}
        # 确保 slide_id 是字符串，label 是整数
        self.label_map = dict(zip(df['slide_id'].astype(str), df['label'].astype(int)))
        
        # 2. 过滤有效文件
        # 只加载那些既在文件夹里有 .pt，又在 CSV 里有标签的文件
        all_pt_files = glob.glob(os.path.join(data_dir, "*.pt"))
        self.files = []
        
        for f in all_pt_files:
            # 从文件名提取 ID: "path/to/TCGA-BLCA-001.pt" -> "TCGA-BLCA-001"
            slide_id = os.path.basename(f).replace('.pt', '')
            
            if slide_id in self.label_map:
                self.files.append(f)
            else:
                # 可以在这里打印 warning，提示有些文件没标签
                pass

        print(f"✅ 下游数据集加载完成！")
        print(f"   - 文件夹文件总数: {len(all_pt_files)}")
        print(f"   - CSV 标签总数: {len(self.label_map)}")
        print(f"   - 最终有效样本数 (交集): {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        # 3. 加载特征 (跟预训练一样)
        data = torch.load(file_path, map_location="cpu", weights_only=True)
        features = data['grid_32']  # (1024, 1536)
        
        # 4. 获取标签
        slide_id = os.path.basename(file_path).replace('.pt', '')
        label = self.label_map[slide_id]
        
        # 5. 返回微调格式
        # 注意：这里不再需要 bool_masked_pos，因为我们要给模型看完整的图
        return {
            "input_features": features,                  # 完整的 32x32 特征
            "labels": torch.tensor(label, dtype=torch.long) # 必须是 long 类型 (0, 1, 2...)
        }

class OfflineNepaDataset(Dataset):
    def __init__(self, data_dir, mask_ratio=0.4):
        """
        Args:
            data_dir: prepare_offline_data.py 生成的 .pt 文件夹路径
            mask_ratio: 预训练遮挡比例，默认 0.4 (即遮住 40% 的有效 Patch)
        """
        # 1. 自动搜索路径下所有的 .pt 文件
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        self.mask_ratio = mask_ratio
        
        if len(self.files) == 0:
            raise FileNotFoundError(f"错误：在路径 {data_dir} 下未找到 .pt 文件。请检查路径或先运行预处理脚本。")
        
        print(f"成功加载离线数据集，共包含 {len(self.files)} 个样本。")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 2. 直接读取离线生成的特征和掩码
        # 使用 weights_only=True 是更安全的做法
        data = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        
        # grid_32: (1024, 1536) - 展平后的 32x32 网格
        # mask_32: (1024,) - 布尔值，True 代表该位置有组织特征
        features = data['grid_32']
        valid_mask = data['mask_32']
        
        # 3. 生成自监督训练用的随机 Mask 
        valid_indices = torch.where(valid_mask)[0]
        num_valid = len(valid_indices)
        
        # 初始化全为 False 的 Mask
        bool_masked_pos = torch.zeros(features.shape[0], dtype=torch.bool)
        
        if num_valid > 0:
            # 计算需要遮挡的数量
            num_to_mask = int(num_valid * self.mask_ratio)
            # 随机打乱索引并取前 N 个
            perm = torch.randperm(num_valid)
            masked_indices = valid_indices[perm[:num_to_mask]]
            # 将这些位置标记为 True (即这些位置会被模型替换为 [MASK] Token)
            bool_masked_pos[masked_indices] = True

        # 4. 返回模型需要的三个核心字段
        return {
            "input_features": features,        # 输入特征 (模型会根据 bool_masked_pos 自动进行涂黑)
            "bool_masked_pos": bool_masked_pos, # 掩码位置清单
            "label_features": features.clone()  # 标签：未被涂黑的原始特征
        }

class H5FeatureGridDataset(Dataset):
    def __init__(self, h5_dir, target_size=32, feature_dim=1536, mask_ratio=0.4):
        """
        h5_dir: 存放 .h5 文件的路径
        target_size: 网格大小 (32x32)
        feature_dim: 输入特征维度 (ResNet=1024, UNI=1536)
        mask_ratio: 预训练时的掩码比例
        """
        self.h5_files = glob.glob(os.path.join(h5_dir, "*.h5"))
        self.target_size = target_size
        self.feature_dim = feature_dim
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        h5_path = self.h5_files[idx]
        
        with h5py.File(h5_path, 'r') as f:
            # 这里的 Key 根据你之前检查的结果：'features' 和 'coords'
            features = f['features'][0] # (N, feature_dim)
            coords = f['coords'][0]     # (N, 2)

        # 1. 坐标归一化到 [0, target_size-1]
        c_min = coords.min(axis=0)
        c_max = coords.max(axis=0)
        # 避免除以 0
        denom = (c_max - c_min)
        denom[denom == 0] = 1.0
        
        # 映射坐标到 32x32 的整数索引
        normalized_coords = (coords - c_min) / denom * (self.target_size - 1)
        grid_indices = normalized_coords.astype(int)

        # 2. 构造 32x32 特征网格
        # 初始化为 0，形状为 (32, 32, feature_dim)
        grid_features = torch.zeros((self.target_size, self.target_size, self.feature_dim))
        # 构造有效载荷掩码 (哪些格子有组织)
        valid_mask = torch.zeros((self.target_size, self.target_size), dtype=torch.bool)

        for i in range(len(features)):
            x, y = grid_indices[i]
            # 如果多个 patch 映射到同一个格，简单覆盖
            grid_features[y, x] = torch.from_numpy(features[i]).float()
            valid_mask[y, x] = True

        # 3. 展平为序列 (1024, feature_dim)
        flat_features = grid_features.view(-1, self.feature_dim)
        flat_valid_mask = valid_mask.view(-1)

        # 4. 生成随机掩码位置 (针对预训练)
        # 我们只在有组织 (valid_mask 为 True) 的地方进行遮挡
        num_valid = flat_valid_mask.sum().item()
        num_mask = int(num_valid * self.mask_ratio)
        
        # 找出所有有效位置的索引
        valid_indices = torch.where(flat_valid_mask)[0]
        # 随机抽取要被 mask 的索引
        perm = torch.randperm(num_valid)
        masked_indices = valid_indices[perm[:num_mask]]
        
        # 构造 bool_masked_pos (1024,)
        # 这是模型 Embeddings 层需要的参数
        bool_masked_pos = torch.zeros(self.target_size * self.target_size, dtype=torch.bool)
        bool_masked_pos[masked_indices] = True

        return {
            "input_features": flat_features,      # 对应模型 forward 的参数
            "bool_masked_pos": bool_masked_pos,    # 对应模型 forward 的参数
            "label_features": flat_features.clone() # 预训练的目标通常是原始特征
        }


if __name__ == "__main__":
    dataset = H5FeatureGridDataset(h5_dir="/root/autodl-tmp/nepa/datas/TCGA/TCGA-BLCA")
    sample = dataset[0]
    print(sample["input_features"].shape) # 应该是 torch.Size([1024, 1024])
    print(sample["bool_masked_pos"].sum()) # 应该大约是 1024 * 0.4
