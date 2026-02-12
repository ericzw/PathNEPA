#!/bin/bash

# ================= 配置区域 =================
# 项目路径 (根据你的实际路径修改)
ROOT_DIR="/root/autodl-tmp/nepa"
CODE_DIR="$ROOT_DIR/codes/nepa-main"
DATA_DIR="$ROOT_DIR/datas/TCGA/TCGA-BLCA-PT"
OUTPUT_DIR="./output_pretrain_32x32_ema_1"

# 显卡设置 (如果你有多卡，设置为 0,1,2,3)
export CUDA_VISIBLE_DEVICES=0

# 关键参数
INPUT_DIM=1536              # 特征维度 (UNI=1536, ResNet=1024)
BATCH_SIZE=32             # 单卡 Batch Size
LEARNING_RATE=1.5e-4        # 主干学习率
EMBED_LR=2.0e-4             # 投影层学习率 (稍大一点以便快速对齐)
EPOCHS=50                   # 训练轮数

# ================= 启动命令 =================
echo "🚀 开始训练..."
echo "📂 数据路径: $DATA_DIR"
echo "🔢 特征维度: $INPUT_DIM"
echo "💾 输出路径: $OUTPUT_DIR"

# 切换到代码目录
cd $CODE_DIR

python run_nepa_h5.py \
    --output_dir "$OUTPUT_DIR" \
    --h5_dir "$DATA_DIR" \
    --input_feat_dim $INPUT_DIM \
    --model_config_name "./configs/pretrain/nepa-base-patch14-224/config.json" \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --embed_lr $EMBED_LR \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --logging_steps 10 \
    --save_strategy "epoch" \
    --eval_strategy "epoch" \
    --fp16 True \
    --dataloader_num_workers 8 \
    --remove_unused_columns False \
    --report_to "tensorboard" \
    --overwrite_output_dir \
    --mask_ratio 0.4

echo "✅ 训练结束！"