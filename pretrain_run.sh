#!/bin/bash

# ==========================================
# ⚙️ 第一部分：系统与环境配置
# ==========================================
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3            # 指定使用的 GPU 卡号 (例如 "0,1,2,3" 多卡)

# ==========================================
# 📂 第二部分：路径与目录配置
# ==========================================
ROOT_DIR="/data2/mengzibing/medicine"
CODE_DIR="${ROOT_DIR}/PathNEPA"
DATA_DIR="${ROOT_DIR}/datasets/dataset_offline/Sub-typing"
OUTPUT_DIR="./output_pretrain_32x32_ema_6"
LOG_FILE="${OUTPUT_DIR}/train.log"

# ==========================================
# 🧠 第三部分：模型与特征配置
# ==========================================
MODEL_NAME_OR_PATH="SixAILab/nepa-base-patch14-224"  # 官方预训练底座
INPUT_DIM=1536                                       # 特征维度 (UNI=1536, ResNet=1024)
MASK_RATIO=0.4                                       # MAE 遮掩比例

# ==========================================
# 🚀 第四部分：核心训练超参数
# ==========================================
EPOCHS=800                                # 总训练轮数
FREEZE_EPOCHS=2                          # 前几个 Epoch 冻结主干
BATCH_SIZE=3                             # 训练 Batch Size (大卡可改成 16/32/64)
EVAL_BATCH_SIZE=2                        # 验证 Batch Size
LEARNING_RATE=1.5e-4                     # 主干学习率 (Base LR)
EMBED_LR=2.0e-4                          # 投影层专属学习率 (Head/Embed LR)
WARMUP_RATIO=0.05                        # 预热步数比例
WEIGHT_DECAY=0.05                        # 权重衰减
GRAD_CHECKPOINTING="True"                # 梯度检查点 (省显存神器)
USE_BF16="True"                          # 是否使用 BF16 半精度 (A100/H100 强烈建议 True，老卡用 fp16)

# ==========================================
# 📊 第五部分：数据加载与日志策略
# ==========================================
NUM_WORKERS=8                            # DataLoader 的 CPU 线程数
LOGGING_STEPS=10                         # 每隔多少步打印一次日志
SAVE_STRATEGY="epoch"
SAVE_TOTAL_LIMIT=3                       # 保存的模型总数
EVAL_STRATEGY="epoch"                    # 验证策略 ("steps" 或 "epoch")
REPORT_TO="tensorboard"                  # 可视化工具 ("tensorboard" 或 "wandb")
REMOVE_UNUSED_COLUMNS="False"            # 是否移除未使用的列 (自定义 Dataset 必须设为 False)

# ==========================================
# 🎬 启动执行区 (无需修改)
# ==========================================
# 确保输出文件夹存在，避免日志文件创建失败
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "🚀 准备启动离线特征预训练..."
echo "🖥️  使用 GPU: $CUDA_VISIBLE_DEVICES"
echo "📂 数据路径: $DATA_DIR"
echo "🔢 特征维度: $INPUT_DIM"
echo "📦 Batch Size: $BATCH_SIZE"
echo "💾 输出路径: $OUTPUT_DIR"
echo "📝 日志文件: $LOG_FILE"
echo "========================================="

# 切换到代码目录
cd "$CODE_DIR"

# 放到后台运行，并将所有输出重定向到 LOG_FILE
nohup python run_nepa_h5.py \
    --output_dir "$OUTPUT_DIR" \
    --h5_dir "$DATA_DIR" \
    --do_train \
    --input_feat_dim "$INPUT_DIM" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --num_train_epochs "$EPOCHS" \
    --freeze_epochs "$FREEZE_EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
    --gradient_checkpointing "$GRAD_CHECKPOINTING" \
    --learning_rate "$LEARNING_RATE" \
    --embed_lr "$EMBED_LR" \
    --warmup_ratio "$WARMUP_RATIO" \
    --weight_decay "$WEIGHT_DECAY" \
    --logging_steps "$LOGGING_STEPS" \
    --save_strategy "$SAVE_STRATEGY" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --eval_strategy "$EVAL_STRATEGY" \
    --bf16 "$USE_BF16" \
    --dataloader_num_workers "$NUM_WORKERS" \
    --remove_unused_columns "$REMOVE_UNUSED_COLUMNS" \
    --report_to "$REPORT_TO" \
    --overwrite_output_dir \
    --mask_ratio "$MASK_RATIO" > "$LOG_FILE" 2>&1 &

echo "✅ 训练已成功在后台启动！"
echo "💡 提示：你可以使用以下命令实时查看训练进度："
echo "tail -f $LOG_FILE"