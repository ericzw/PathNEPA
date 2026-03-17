import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import glob
import torch
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
# === 导入你自己定义的模块 ===
# 假设你把模型代码保存为了 modeling_vit_nepa.py
from models.vit_nepa import ViTNepaForPreTraining, ViTNepaConfig
# 假设你把修复好的 Dataset 保存为了 dataset.py
# from models.dataset import PathNEPAFeatureDataset
# # # from models.dataset import  FastOfflineMILDataset as PathNEPAFeatureDataset
from models.dataset import OfflinePretrainDataset as PathNEPAFeatureDataset
# 检查版本
check_min_version("4.28.0")
logger = logging.getLogger(__name__)

# =============================================================================
# 1. 移植过来的 EnhancedTrainer (包含 EMA 和 分层学习率)
# =============================================================================
class EnhancedTrainer(Trainer):
    def __init__(
        self,
        *args,
        embed_lr=None,     # 特征投影层的特定学习率
        ema_decay=0.9999,  # EMA 衰减率，越接近 1 更新越慢越平滑
        use_ema=True,      # 是否开启 EMA
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embed_lr = embed_lr
        self.ema_decay = ema_decay
        self.use_ema = use_ema
        self.ema_model = None

    def get_decay_parameter_names(self, model) -> list[str]:
        forbidden_name_patterns = [r"bias", r"layernorm", r"rmsnorm", r"layer_scale", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
        decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm], forbidden_name_patterns)
        return decay_parameters

    def create_optimizer(self):
        """
        自定义优化器：允许给 embedding 层设置不同的学习率
        """
        if self.optimizer is not None:
            return self.optimizer

        # 如果没有指定特殊的 embed_lr，就用默认逻辑
        if self.embed_lr is None:
            return super().create_optimizer()

        decay_params = set(self.get_decay_parameter_names(self.model))
        
        # 这里的名字要匹配 ViTNepaForPreTraining 里的结构
        # self.model.vit_nepa.embeddings 包含了我们新加的 projection 层
        embed_params = set(
            f"vit_nepa.embeddings.{n}"
            for n, _ in self.model.vit_nepa.embeddings.named_parameters()
        )

        wd = self.args.weight_decay
        base_lr = self.args.learning_rate

        # 将参数分成 4 组：
        # 1. Embedding 层 (有 Decay)
        # 2. Embedding 层 (无 Decay - 如 Bias/Norm)
        # 3. 其他层 (有 Decay)
        # 4. 其他层 (无 Decay)
        groups = [
            {"params": [], "weight_decay": wd,  "lr": self.embed_lr},
            {"params": [], "weight_decay": 0.0, "lr": self.embed_lr},
            {"params": [], "weight_decay": wd,  "lr": base_lr},
            {"params": [], "weight_decay": 0.0, "lr": base_lr},
        ]

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_decay = name in decay_params
            is_embed = name in embed_params

            if is_embed and is_decay:
                groups[0]["params"].append(p)
            elif is_embed and not is_decay:
                groups[1]["params"].append(p)
            elif (not is_embed) and is_decay:
                groups[2]["params"].append(p)
            else:
                groups[3]["params"].append(p)

        optimizer_grouped_parameters = [g for g in groups if g["params"]]

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    # --- EMA 核心逻辑 ---
    def _init_ema_model(self):
        if self.ema_model is None:
            import copy
            # 深拷贝当前模型
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.eval()
            # 确保 EMA 模型在同样的设备和精度上
            for p in self.ema_model.parameters():
                p.requires_grad_(False) # EMA 不求导

    def _update_ema(self):
        if not self.use_ema:
            return
        if self.ema_model is None:
            self._init_ema_model()
        
        # 移动平均公式: theta_new = decay * theta_old + (1 - decay) * theta_current
        with torch.no_grad():
            msd = self.model.state_dict()
            for k, v in self.ema_model.state_dict().items():
                if k in msd:
                    model_param = msd[k].to(v.device).type(v.dtype)
                    v.mul_(self.ema_decay).add_(model_param, alpha=1.0 - self.ema_decay)

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        # 每次 Step 更新时，顺便更新 EMA
        if self.state.global_step > getattr(self, "_ema_global_step", 0):
            self._update_ema()
            self._ema_global_step = self.state.global_step
        super()._maybe_log_save_evaluate(*args, **kwargs)

    def save_model(self, output_dir=None, _internal_call=False):
        # 保存时，额外保存 EMA 模型
        super().save_model(output_dir, _internal_call)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if self.use_ema and self.ema_model is not None and self.args.should_save:
            ema_path = f"{output_dir}/pytorch_model_ema.bin"
            os.makedirs(os.path.dirname(ema_path), exist_ok=True)
            torch.save(self.ema_model.state_dict(), ema_path)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        # 尝试加载 EMA 权重
        if self.use_ema:
            ema_ckpt = os.path.join(resume_from_checkpoint, "pytorch_model_ema.bin")
            if os.path.exists(ema_ckpt):
                if self.ema_model is None:
                    self._init_ema_model()
                state_dict = torch.load(ema_ckpt, map_location="cpu")
                self.ema_model.load_state_dict(state_dict, strict=False)
                print(f"[EMA] Loaded EMA checkpoint from {ema_ckpt}")

# =============================================================================
# 2. 参数定义
# =============================================================================
@dataclass
class ModelArguments:
    model_config_name: str = field(
        default="google/vit-base-patch16-224", 
        metadata={"help": "基础配置文件 (用于初始化架构参数)"}
    )
    model_name_or_path: str = field(
        default=None, 
        metadata={"help": "加载官方预训练权重的路径"})
    input_feat_dim: int = field(
        default=1536, 
        metadata={"help": "H5 文件中的特征维度 (ResNet=1024, UNI=1536)"}
    )
    embed_lr: Optional[float] = field(
        default=None,
        metadata={"help": "特征投影层的专属学习率 (通常比主干网络小一点)"}
    )
    freeze_epochs: int = field(
        default=2, 
        metadata={"help": "预热阶段冻结主干的 Epoch 数量"}
    )

@dataclass
class DataArguments:
    h5_dir: str = field(
        default="/root/autodl-tmp/nepa/datas/TCGA/TCGA-BLCA", 
        metadata={"help": "存放 .h5 文件的文件夹路径"}
    )
    mask_ratio: float = field(
        default=0.4, 
        metadata={"help": "预训练 Masking 比例"}
    )
    val_ratio: float = field(
        default=0.1,
        metadata={"help": "验证集切分比例"}
    )
def collate_fn(examples):
    # 将 B 和 64 融合
    batch_features = torch.stack([ex["input_features"] for ex in examples]) # (B, 64, 1024, 1536)
    batch_labels = torch.stack([ex["label_features"] for ex in examples])
    
    B, num_crops, seq_len, D = batch_features.shape
    
    res = {
        "input_features": batch_features.view(B * num_crops, seq_len, D),
        "label_features": batch_labels.view(B * num_crops, seq_len, D)
    }
    
    if "bool_masked_pos" in examples[0]:
        batch_mask = torch.stack([ex["bool_masked_pos"] for ex in examples])
        res["bool_masked_pos"] = batch_mask.view(B * num_crops, seq_len)
        
    return res
# =============================================================================
# 3. 主函数
# =============================================================================
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 1. 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    
    logger.info(f"Training parameters: {training_args}")

    set_seed(training_args.seed)

    # 2. 加载 Dataset 
    logger.info(f"Loading H5 data from {data_args.h5_dir}...")

    # 1. 递归查找文件夹下所有的 .h5 文件，生成一个列表
    h5_files = glob.glob(os.path.join(data_args.h5_dir, "**", "*.h5"), recursive=True)
    
    # 加个保险，防止路径填错导致空列表
    if len(h5_files) == 0:
        raise FileNotFoundError(f"错误：在 {data_args.h5_dir} 及其子文件夹下没有找到任何 .h5 文件！")
    
    logger.info(f"成功找到 {len(h5_files)} 个 .h5 文件，正在初始化 Dataset...")

    # 2. 把文件列表传给 h5_file_list
    full_dataset = PathNEPAFeatureDataset(
        h5_file_list=h5_files,       # <--- 注意这里参数名是 h5_file_list，传入的是刚才搜出来的列表
        num_crops=64,                # 每次随机切 64 张图 (觉得显存不够可以改小，比如 16, 32)
        crop_size=32,                # 32x32 的特征图
        valid_ratio=0.5,             # 保证有效组织 >= 50%
        mask_ratio=data_args.mask_ratio
    )

    # 划分 Train / Val
    val_size = int(len(full_dataset) * data_args.val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    logger.info(f"Dataset loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 3. 初始化模型
    # 加载基础配置
    config = ViTNepaConfig.from_pretrained(model_args.model_config_name)
    config.input_feat_dim = model_args.input_feat_dim  # 1536
    if model_args.model_name_or_path is not None:
        logger.info(f"🚀 正在加载官方预训练底座: {model_args.model_name_or_path}")
        model = ViTNepaForPreTraining.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            ignore_mismatched_sizes=True # 容忍 Decoder 未初始化的维度警告
        )
    else:
        logger.info("🌱 未指定权重，从头开始随机初始化模型")
        model = ViTNepaForPreTraining(config)

    
    # 打印模型结构确认一下
    logger.info(f"Model initialized. Input Feature Dim: {config.input_feat_dim}")

    # =============================================================================
    # 4. 进阶：两阶段训练 (LP-FT 策略) 与 EnhancedTrainer 初始化
    # =============================================================================
    freeze_epochs = model_args.freeze_epochs

    if training_args.do_train:
        # 判断：如果总 Epoch 数大于冻结的 Epoch 数，才进行两阶段训练
        if freeze_epochs > 0 and (training_args.num_train_epochs > freeze_epochs):
            logger.info("="*60)
            logger.info(f"❄️ [第一阶段] 冻结官方预训练主干，先让 Decoder 和特征降维层预热 {freeze_epochs} 个 Epoch...")
            logger.info("="*60)

            # 1. 冻结 Encoder 主干
            for param in model.vit_nepa.parameters():
                param.requires_grad = False
            
            # 2. 强制解冻新初始化的组件 (投影层和 mask_token)
            for param in model.vit_nepa.embeddings.feature_projection.parameters():
                param.requires_grad = True
            if hasattr(model.vit_nepa.embeddings, 'mask_token'):
                model.vit_nepa.embeddings.mask_token.requires_grad = True

            # 备份原始的 epochs
            total_epochs = training_args.num_train_epochs
            training_args.num_train_epochs = freeze_epochs

            # 实例化第一阶段的 EnhancedTrainer
            trainer_phase1 = EnhancedTrainer(
                model=model,
                args=training_args,
                # ⚠️ 修复了你原本代码里把 full_dataset 传进去的 Bug，改为 train_dataset
                train_dataset=train_dataset, 
                eval_dataset=val_dataset if training_args.do_eval else None,
                data_collator=collate_fn,  
                embed_lr=model_args.embed_lr,
            )
            
            # 第一阶段只做预热，通常不建议直接 resume
            trainer_phase1.train()

            logger.info("="*60)
            logger.info("🔥 [第二阶段] 新手预热完毕！全面解冻老手主干，开始端到端协同微调...")
            logger.info("="*60)

            # 3. 全面解冻老手！
            for param in model.vit_nepa.parameters():
                param.requires_grad = True
                
            # 恢复剩余的 Epoch 数量 (比如总共 10 个，预热了 2 个，剩下跑 8 个)
            training_args.num_train_epochs = total_epochs - freeze_epochs

            # 💥 极其关键：必须重新实例化第二阶段的 EnhancedTrainer！
            # 因为你的 EnhancedTrainer 重写了 create_optimizer()，
            # 只有重新实例化并调用 train()，它才会把刚刚解冻的 Encoder 参数重新加入到优化器中！
            trainer_phase2 = EnhancedTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset if training_args.do_eval else None,
                data_collator=collate_fn,  
                embed_lr=model_args.embed_lr,
            )
            
            trainer_phase2.train()
            
            # 5. 保存最终结果
            trainer_phase2.save_model()
            trainer_phase2.save_state()

        else:
            # 如果不满足两阶段条件（比如 freeze_epochs 设为 0），走标准单阶段
            logger.info("🌱 进行标准的单阶段全参训练...")
            trainer = EnhancedTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset if training_args.do_eval else None,
                data_collator=collate_fn,  
                embed_lr=model_args.embed_lr,
            )
            
            if training_args.resume_from_checkpoint:
                logger.info(f"Resuming from checkpoint: {training_args.resume_from_checkpoint}")
                trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
            else:
                trainer.train()

            # 5. 保存最终结果
            trainer.save_model()
            trainer.save_state()

if __name__ == "__main__":
    main()