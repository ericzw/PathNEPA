#!/bin/bash

# ================= 配置区域 =================
# 项目路径 (根据你的实际路径修改)
ROOT_DIR="/data2/mengzibing/medicine"
CODE_DIR="$ROOT_DIR/PathNEPA"
DATA_DIR="$ROOT_DIR/datasets/dataset_o/Sub-typing"
OUTPUT_DIR="./output_pretrain_32x32_ema_1"

# 新增：定义日志文件的保存路径
LOG_FILE="${OUTPUT_DIR}/train.log"

# 解决显存碎片化问题
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES=3

# 关键参数
INPUT_DIM=1536              # 特征维度 (UNI=1536, ResNet=1024)
BATCH_SIZE=1                # 
ACCUMULATION_STEPS=32       # ⚠️ 累加 32 次
LEARNING_RATE=1.5e-4        # 主干学习率
EMBED_LR=2.0e-4             # 投影层学习率
EPOCHS=50                   # 训练轮数

# ================= 启动命令 =================
# 确保输出文件夹存在，避免日志文件创建失败
mkdir -p "$OUTPUT_DIR"

echo "🚀 准备启动训练..."
echo "📂 数据路径: $DATA_DIR"
echo "🔢 特征维度: $INPUT_DIM"
echo "💾 输出路径: $OUTPUT_DIR"
echo "📝 日志文件将保存至: $LOG_FILE"

# 切换到代码目录
cd $CODE_DIR

# ⚠️ 关键修改：使用 nohup 放到后台运行，并将所有输出重定向到 LOG_FILE
nohup python run_nepa_h5.py \
    --output_dir "$OUTPUT_DIR" \
    --h5_dir "$DATA_DIR" \
    --do_train \
    --input_feat_dim $INPUT_DIM \
    --model_config_name "./configs/pretrain/nepa-base-patch14-224/config.json" \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $ACCUMULATION_STEPS \
    --gradient_checkpointing True \
    --learning_rate $LEARNING_RATE \
    --embed_lr $EMBED_LR \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --logging_steps 10 \
    --save_strategy "epoch" \
    --eval_strategy "epoch" \
    --bf16 True \
    --dataloader_num_workers 0 \
    --remove_unused_columns False \
    --report_to "tensorboard" \
    --overwrite_output_dir \
    --mask_ratio 0.4 > "$LOG_FILE" 2>&1 &

echo "✅ 训练已成功在后台启动！"
echo "💡 提示：你可以使用以下命令实时查看训练进度："
echo "tail -f $LOG_FILE"