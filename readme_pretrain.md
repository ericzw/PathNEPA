# PathNEPA: Offline Feature Pre-training Pipeline 🚀

本项目是基于 PathNEPA 的极速离线预训练/微调管道。代码已针对病理全切片图像（WSI）的离线提取特征（如 1536 维的 UNI 特征）进行了深度重构与 I/O 优化，支持在大显存 GPU 集群上进行高效的端到端训练。

## 🌟 核心特性 (Key Features)

1. **纯离线 H5 引擎**：直接读取预提取的 `.h5` 特征文件，动态进行固定长度（如 1024）的随机采样，完美避开海量 RGB 图像预处理造成的 I/O 瓶颈和显存溢出。
2. **LP-FT (两阶段解冻) 策略**：
   - **Phase 1 (预热阶段)**：前 2 个 Epoch 冻结官方预训练的 ViT 主干，仅训练随机初始化的 Decoder 和特征投影层（Feature Projection），有效防止灾难性遗忘。
   - **Phase 2 (协同微调)**：自动解冻所有网络层，进行端到端的深度特征微调。
3. **EMA (指数移动平均)**：内置 EMA 权重平滑更新机制，提升下游任务（如亚型分类、生存预测）的泛化能力。

---

## 🛠️ 1. 环境准备 (Environment Setup)

> **⚠️ 【重要警告】关于大算力显卡 (A100 / H100 等)**
> 如果您使用的是最新架构的显卡，请**务必先手动安装匹配您服务器 CUDA 版本（通常为 11.8 或 12.1+）的 PyTorch**，然后再安装 `requirements.txt` 中的其他依赖，否则极易发生环境不兼容或无法调用 GPU 的问题。

推荐的安装流程：

```bash
# 1. 创建干净的 Conda 环境
conda create -n nepa python=3.9 -y
conda activate nepa

# 2. 安装匹配您机器 CUDA 版本的 PyTorch (示例为 CUDA 11.8)
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. 一键安装核心依赖项
pip install -r requirements.txt

```

## 数据集准备
```
# 文件路径 /scripts/dataset/prepare_offline_features.py
# 在文件内修改好路径之后 运行
python /scripts/dataset/prepare_offline_features.py
```

## RUN
```
# 需要进行调整的参数 都在里面 freeze_epoch lr等等
chmod +x pretrain_run.sh
./pretrain_run.sh 
```