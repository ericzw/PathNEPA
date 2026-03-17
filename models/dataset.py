# ==============================================================================
# dataset.py
# PathNEPA 核心数据引擎 (究极 I/O 优化版)
# 特性: 零大内存分配、HDF5 Fancy Indexing、searchsorted 极速映射
# ==============================================================================
import time
import h5py
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

class PathNEPAFeatureDataset(Dataset):
    def __init__(self, h5_file_list, num_crops=64, crop_size=32, valid_ratio=0.5, mask_ratio=0.4):
        self.h5_files = h5_file_list
        self.num_crops = num_crops
        self.crop_size = crop_size
        self.valid_ratio = valid_ratio
        self.mask_ratio = mask_ratio
        self.threshold_count = (crop_size * crop_size) * valid_ratio
        
        # 将 kernel 提前放到类变量中，避免每次 __getitem__ 都重新申请内存
        self.kernel = torch.ones((1, 1, self.crop_size, self.crop_size), dtype=torch.float32)
        
        print("="*50)
        print(f"📦 [Dataset Init] 极速版 PathNEPAFeatureDataset 引擎启动")
        print(f"   -> 挂载 .h5 文件总数: {len(self.h5_files)}")
        print(f"   -> 单 WSI 切图策略: {self.num_crops} 张 ({self.crop_size}x{self.crop_size})")
        print(f"   -> 有效区域阈值: >= {self.threshold_count} Patch ({valid_ratio*100}%)")
        print(f"   -> 预训练遮挡率: {mask_ratio*100}%")
        print("="*50)

    def __len__(self):
        return len(self.h5_files)

    def _infer_patch_size(self, coords):
        """自动推断原始切片的 Patch Size (通常是 256)"""
        x_unique = np.unique(coords[:, 0])
        x_diffs = np.diff(np.sort(x_unique))
        x_diffs = x_diffs[x_diffs > 0]
        return int(np.min(x_diffs)) if len(x_diffs) > 0 else 256

    def __getitem__(self, idx):
        start_time = time.time()
        
        h5_path = self.h5_files[idx]
        file_name = h5_path.split('/')[-1]
        
        try:
            # --- 1. 仅读取坐标和维度，决不盲目读取 features (极速) ---
            t0 = time.time()
            with h5py.File(h5_path, 'r', swmr=True, libver='latest') as f:
                coords = f['coords_patching'][:] 
                t0_1 = time.time()
                D = f['features'].shape[2]  # 获取特征维度，如 1536
                t0_2 = time.time()
                print(f"📂 [{file_name}] 读取坐标耗时: {t0_1 - t0:.3f}s, 获取维度耗时: {t0_2 - t0_1:.3f}s")
                
                patch_size = self._infer_patch_size(coords)
                grid_coords = coords // patch_size 
                
                min_x, min_y = grid_coords.min(axis=0)
                max_x, max_y = grid_coords.max(axis=0)
                
                W = int(max_x - min_x + 1)
                H = int(max_y - min_y + 1)
                
                # --- 2. 建立极轻量级索引网格 (背景填 -1) ---
                index_grid = np.full((H, W), -1, dtype=np.int32) 
                norm_coords = (grid_coords - [min_x, min_y]).astype(int)
                index_grid[norm_coords[:, 1], norm_coords[:, 0]] = np.arange(len(coords))

                pad_h = max(0, self.crop_size - H)
                pad_w = max(0, self.crop_size - W)
                if pad_h > 0 or pad_w > 0:
                    index_grid = np.pad(index_grid, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=-1)
                    
                valid_mask = (index_grid != -1)
                mask_tensor = torch.from_numpy(valid_mask).float().unsqueeze(0).unsqueeze(0)
                
                # --- 3. 二维卷积雷达扫图，寻找含肉量达标的区域 ---
                t1 = time.time()
                valid_sums = F.conv2d(mask_tensor, self.kernel, stride=self.crop_size, padding=0).squeeze(0).squeeze(0)
                valid_positions = torch.nonzero(valid_sums >= self.threshold_count)
                
                if len(valid_positions) == 0:
                    valid_positions = torch.nonzero(valid_sums == valid_sums.max())

                sampled_indices = torch.randint(0, len(valid_positions), (self.num_crops,))
                sampled_positions = valid_positions[sampled_indices]
                
                # --- 4. 汇总这 64 个切片到底需要哪些 Feature (开购物清单) ---
                t2 = time.time()
                crop_idx_arrays = []
                for pos in sampled_positions:
                    y_start = pos[0].item() * self.crop_size
                    x_start = pos[1].item() * self.crop_size
                    crop_idx_2d = index_grid[y_start : y_start + self.crop_size, x_start : x_start + self.crop_size]
                    crop_idx_arrays.append(crop_idx_2d.flatten())

                all_needed_indices = np.concatenate(crop_idx_arrays)
                valid_locs_global = all_needed_indices[all_needed_indices != -1]
                unique_indices = np.unique(valid_locs_global) # 自动去重并从小到大排序

                # --- 5. 💥 HDF5 按需拉取 (Fancy Indexing) 💥 ---
                t3 = time.time()
                if len(unique_indices) > 0:
                    # 只从硬盘读取必需的这部分数据，彻底释放 IO 总线！
                    loaded_features = f['features'][0, unique_indices, :]
                else:
                    loaded_features = np.zeros((0, D), dtype=np.float32)

                # --- 6. 将拉取到的数据拼装回 64 个 Tensor 里 ---
                t4 = time.time()
                crops_feat = np.zeros((self.num_crops, self.crop_size * self.crop_size, D), dtype=np.float32)
                crops_masked_pos = np.zeros((self.num_crops, self.crop_size * self.crop_size), dtype=bool)
                
                for i, crop_idx_1d in enumerate(crop_idx_arrays):
                    valid_locs = np.where(crop_idx_1d != -1)[0]
                    real_indices = crop_idx_1d[valid_locs]

                    if len(real_indices) > 0:
                        # 用 searchsorted 将全局 index 瞬间映射到 loaded_features 的本地 index
                        mapped_indices = np.searchsorted(unique_indices, real_indices)
                        crops_feat[i, valid_locs, :] = loaded_features[mapped_indices]

                    # 动态生成完形填空 Mask
                    num_mask = int(len(valid_locs) * self.mask_ratio)
                    if num_mask > 0:
                        masked_idx = valid_locs[np.random.permutation(len(valid_locs))[:num_mask]]
                        crops_masked_pos[i, masked_idx] = True

                input_features = torch.from_numpy(crops_feat)
                bool_masked_pos = torch.from_numpy(crops_masked_pos)
                t5 = time.time()

            # --- 耗时日志监控 (性能调优时可解除注释) ---
            total_time = t5 - start_time
            print(f"✅ [{file_name}] 耗时: {total_time:.3f}s (坐标:{t1-t0:.3f}s|找图:{t2-t1:.3f}s|清单:{t3-t2:.3f}s|H5读取:{t4-t3:.3f}s|拼装:{t5-t4:.3f}s)")

        except Exception as e:
            print(f"❌ 读取文件失败: {h5_path}, error: {e}")
            input_features = torch.zeros((self.num_crops, self.crop_size * self.crop_size, 1536)).float()
            bool_masked_pos = torch.zeros((self.num_crops, self.crop_size * self.crop_size)).bool()

        return {
            "input_features": input_features,                         # (64, 1024, 1536)
            "label_features": input_features.clone(),                 # (64, 1024, 1536)
            "bool_masked_pos": bool_masked_pos                        # (64, 1024)
        }

class FastOfflineMILDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, label2id, file_to_label_dict, num_crops=64, mask_ratio=0.4):
        # 依然读取 .h5 文件
        self.h5_files = glob.glob(os.path.join(root_dir, "**", "*.h5"), recursive=True)
        self.label2id = label2id
        self.num_crops = num_crops
        self.mask_ratio = mask_ratio
        
        self.files = []
        self.labels = []
        
        for f in self.h5_files:
            basename = os.path.basename(f)
            file_key = basename.replace(".h5", "") 
            
            matched_label = file_to_label_dict.get(file_key)
            if matched_label is None:
                for k in file_to_label_dict.keys():
                    if k in basename: matched_label = file_to_label_dict[k]; break
            
            if matched_label is not None and str(matched_label) in self.label2id:
                self.files.append(f)
                self.labels.append(self.label2id[str(matched_label)])

        logger.info(f"⚡ 成功挂载 {len(self.files)} 个纯净版离线 H5 文件")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        
        # 1. 极速读取离线处理好的 H5 特征
        try:
            with h5py.File(file_path, 'r', swmr=True, libver='latest') as f:
                # 直接全部读取到内存，因为经过筛选，现在的数据量非常小
                all_crops_np = f['features'][:] 
                
            all_crops = torch.from_numpy(all_crops_np).float()
            M = all_crops.shape[0]
            
            # 2. 随机采样 64 个
            if M >= self.num_crops:
                indices = torch.randperm(M)[:self.num_crops]
            else:
                indices = torch.randint(0, M, (self.num_crops,))
                
            input_features = all_crops[indices] 
            
            # 3. 动态生成 Mask
            bool_masked_pos = torch.zeros((self.num_crops, 1024), dtype=torch.bool)
            if self.mask_ratio > 0:
                for i in range(self.num_crops):
                    valid_locs = torch.where(input_features[i].abs().sum(dim=-1) > 0)[0]
                    num_mask = int(len(valid_locs) * self.mask_ratio)
                    if num_mask > 0:
                        masked_idx = valid_locs[torch.randperm(len(valid_locs))[:num_mask]]
                        bool_masked_pos[i, masked_idx] = True

        except Exception as e:
            logger.error(f"读取异常 {file_path}: {e}")
            input_features = torch.zeros((self.num_crops, 1024, 1536)).float()
            bool_masked_pos = torch.zeros((self.num_crops, 1024)).bool()

        return {
            "input_features": input_features,                         
            "label_features": input_features.clone(),                 
            "bool_masked_pos": bool_masked_pos,
            "labels": label
        }



# ==============================================================================
# 1. 新增：专为离线特征打造的极速预训练 Dataset
# ==============================================================================
class OfflinePretrainDataset(Dataset):
    """
    专为 run_nepa_h5.py (无监督预训练) 设计的极速离线 Dataset。
    直接读取榨汁好的纯净版 .h5 特征，无需任何坐标计算和图像切分！
    """
    # ⚠️ 加入 **kwargs 神器，完美吸收掉 run_nepa_h5.py 传过来的所有多余参数(如 valid_ratio)，绝不报错！
    def __init__(self, h5_file_list, num_crops=64, mask_ratio=0.4, **kwargs):
        self.h5_files = h5_file_list
        self.num_crops = num_crops
        self.mask_ratio = mask_ratio
        
        print("="*50)
        print(f"⚡ [极速预训练引擎] 成功挂载 {len(self.h5_files)} 个离线纯净版 H5 文件")
        print("="*50)

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        file_path = self.h5_files[idx]
        
        try:
            # 1. 极速读取离线 H5 特征 (形状: M, 1024, 1536)
            with h5py.File(file_path, 'r', driver='core', backing_store=False, libver='latest') as f:
                all_crops_np = f['features'][:] 
                
            all_crops = torch.from_numpy(all_crops_np).float()
            M = all_crops.shape[0]
            
            # 2. 随机采样 64 个区域 (Data Augmentation)
            if M >= self.num_crops:
                indices = torch.randperm(M)[:self.num_crops]
            else:
                # 如果该切片有效区域不足 64 个，允许重复采样补齐
                indices = torch.randint(0, M, (self.num_crops,))
                
            input_features = all_crops[indices] # 瞬间拿到 (64, 1024, 1536)
            
            # 3. 动态生成预训练专属的完形填空 Mask
            bool_masked_pos = torch.zeros((self.num_crops, 1024), dtype=torch.bool)
            if self.mask_ratio > 0:
                for i in range(self.num_crops):
                    # 排除掉补零的空白区域 (全0的特征向量)
                    valid_locs = torch.where(input_features[i].abs().sum(dim=-1) > 0)[0]
                    num_mask = int(len(valid_locs) * self.mask_ratio)
                    if num_mask > 0:
                        masked_idx = valid_locs[torch.randperm(len(valid_locs))[:num_mask]]
                        bool_masked_pos[i, masked_idx] = True

        except Exception as e:
            print(f"❌ 读取异常 {file_path}: {e}")
            input_features = torch.zeros((self.num_crops, 1024, 1536)).float()
            bool_masked_pos = torch.zeros((self.num_crops, 1024)).bool()

        # 预训练不需要分类 label，只返回重建任务需要的三件套
        return {
            "input_features": input_features,                         
            "label_features": input_features.clone(),                 
            "bool_masked_pos": bool_masked_pos                        
        }