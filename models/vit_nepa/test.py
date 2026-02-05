import torch
import h5py
import os
import numpy as np

# 导入你刚才修改好的模型
# 注意：确保 modeling_vit_nepa.py 和 configuration_vit_nepa.py 都在当前目录下
from modeling_vit_nepa import ViTNepaForImageClassification
from configuration_vit_nepa import ViTNepaConfig as ViTConfig

# ================= 配置区域 =================
# 替换为你真实的 .h5 文件路径
H5_FILE_PATH = "/root/autodl-tmp/nepa/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.h5" 
# (如果找不到这个特定文件，找一个你目录下存在的替换)

# 你的特征维度 (UNI 是 1536, ResNet 是 1024)
FEATURE_DIM = 1536 
# ===========================================

def load_h5_features(path):
    print(f"📂 正在读取: {path}")
    with h5py.File(path, 'r') as f:
        if 'features' not in f.keys():
            raise ValueError(f"❌ 文件中没有 'features' 键，只有: {list(f.keys())}")
        
        # 读取数据 (通常是 float16 或 float32)
        features = f['features'][:]
        print(f"   原始形状: {features.shape}")
        
    return features

def run_test():
    # 1. 准备数据
    if not os.path.exists(H5_FILE_PATH):
        print(f"⚠️ 找不到文件: {H5_FILE_PATH}")
        print("   -> 将使用随机生成的假数据进行测试...")
        # 模拟一个 (1, 3000, 1536) 的特征
        features_np = np.random.randn(1, 3000, FEATURE_DIM).astype(np.float32)
    else:
        features_np = load_h5_features(H5_FILE_PATH)

    # 2. 数据预处理
    # 转为 Tensor
    features_tensor = torch.from_numpy(features_np).float()
    
    # 【关键】确保维度是 [Batch_Size, Num_Patches, Feat_Dim]
    # 你的 h5 原始形状是 (1, 36164, 1536)，这已经是 [B, N, D] 了，不需要动
    # 如果原始形状是 (36164, 1536)，需要 unsqueeze(0) 变成 (1, 36164, 1536)
    if features_tensor.ndim == 2:
        features_tensor = features_tensor.unsqueeze(0)
    
    print(f"✅ 输入 Tensor 形状: {features_tensor.shape}")

    # 3. 初始化模型
    print("🤖 初始化模型...")
    config = ViTConfig()
    
    # 【重点】告诉模型输入的特征维度是多少，它会自动创建投影层 (1536 -> hidden_size)
    config.input_feat_dim = FEATURE_DIM 
    config.num_labels = 2 # 假设二分类
    
    model = ViTNepaForImageClassification(config)
    model.eval() # 评估模式

    # 4. 前向传播
    print("🚀 开始推理 (Forward Pass)...")
    with torch.no_grad():
        # 注意：这里使用 input_features 参数，而不是 pixel_values
        outputs = model(input_features=features_tensor)

    # 5. 查看结果
    logits = outputs.logits
    print("-" * 30)
    print(f"🎉 运行成功！")
    print(f"   Logits 输出形状: {logits.shape} (预期: [1, 2])")
    print(f"   Logits 值: {logits}")
    print("-" * 30)

if __name__ == "__main__":
    run_test()