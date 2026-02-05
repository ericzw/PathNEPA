#!/bin/bash

# ================= 环境变量配置 =================
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

# 【新增】解决显存碎片化问题 (根据报错建议添加)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ================= 路径配置 =================
# ✅ 路径已修正：指向上一级目录
DATA_DIR="/root/autodl-tmp/nepa/datas/TCGA" 
MODEL_PATH="SixAILab/nepa-base-patch14-224-sft"
OUTPUT_DIR="./output_tcga_finetune_optimized"

# ================= 显存急救配置 =================
#  1. 降级序列长度：从 4096 -> 2048 (如果还爆，请改成 1024)
MAX_SEQ_LEN=1024

FEAT_DIM=1536
BATCH_SIZE=1
#  2. 增加累积步数：因为长度变短了，batch更小了，所以多累积一点
ACCUMULATION_STEPS=32

EPOCHS=10
LR=2e-5

echo "----------------------------------------------------------------"
echo "🚀 开始训练 (防爆显存模式)..."
echo "📉 采样长度: ${MAX_SEQ_LEN}"
echo "⚡ 混合精度: FP16 (开启)"
echo "----------------------------------------------------------------"

python run_image_classification.py \
    --model_name_or_path "${MODEL_PATH}" \
    --train_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --input_feat_dim ${FEAT_DIM} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
    --gradient_checkpointing True \
    --learning_rate ${LR} \
    --weight_decay 0.05 \
    --do_train \
    --train_val_split 0.2 \
    --save_strategy "epoch" \
    --logging_steps 10 \
    --remove_unused_columns False \
    --dataloader_pin_memory False \
    --dataloader_num_workers 4 \
    --ignore_mismatched_sizes \
    --overwrite_output_dir \
    --fp16  # FP16 混合精度

echo "------ 训练结束！-------"