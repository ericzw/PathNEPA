#!/bin/bash

# ==========================================
# ⚙️ 第一部分：系统与环境配置
# ==========================================
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3           # 指定使用的 GPU 卡号

# ==========================================
# 📂 第二部分：路径与目录配置
# ==========================================
DATASET_NAME="BRCA_Survival"             # 数据集名称
ROOT_DIR="/data2/mengzibing/medicine"
OUTPUT_DIR="./output_${DATASET_NAME}2"
CODE_DIR="${ROOT_DIR}/PathNEPA"
DATA_DIR="${ROOT_DIR}/datasets/dataset_o/Survival_Prediction/BRCA"
CLINICAL_FILE="/data2/mengzibing/medicine/datasets/dataset_o/A-source_label/survival_prediction/bins/BRCA.csv" #
LOG_FILE="${OUTPUT_DIR}/${DATASET_NAME}_cv.log"

# ==========================================
# 🧠 第三部分：模型与特征配置
# ==========================================
PRETRAINED_WEIGHTS="SixAILab/nepa-base-patch14-224" 
NUM_BINS=4                              # ⚠️ 替换为你数据中的离散区间数(如4)

# ==========================================
# 🚀 第四部分：核心训练超参数 (Survival Prediction)
# ==========================================
EPOCHS=30                                
BATCH_SIZE=1                             
EVAL_BATCH_SIZE=1                        
GRAD_ACCUM_STEPS=16                      
LEARNING_RATE=0.0001                     # 💡 生存预测学习率稍微调小
WEIGHT_DECAY=0.1                         # 💡 生存预测极易过拟合，正则化加大到 0.1
MAX_GRAD_NORM=1.0                        
USE_BF16="True"                          
SEED=42                                  

# ==========================================
# 📊 第五部分：数据加载与日志策略
# ==========================================
NUM_WORKERS=8                            
LOGGING_STEPS=10                         
SAVE_STRATEGY="epoch"                    
EVAL_STRATEGY="epoch"                    
SAVE_TOTAL_LIMIT=1                       
REPORT_TO="tensorboard"                  
REMOVE_UNUSED_COLUMNS="False"            

# ==========================================
# 🎬 启动执行区 (⚠️ 这里的 \ 后面绝对不能有注释或空格)
# ==========================================
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "🚀 准备启动下游 Survival 5-Fold 交叉验证..."
echo "🖥️  使用 GPU: $CUDA_VISIBLE_DEVICES"
echo "📂 数据路径: $DATA_DIR"
echo "🏷️  标签文件: $CLINICAL_FILE"
echo "🔢 Bin 数量: $NUM_BINS"
echo "💾 输出路径: $OUTPUT_DIR"
echo "📝 日志文件: $LOG_FILE"
echo "========================================="

cd "$CODE_DIR"

nohup python run_surv_h5.py \
    --model_name_or_path "$PRETRAINED_WEIGHTS" \
    --data_dir "$DATA_DIR" \
    --clinical_file "$CLINICAL_FILE" \
    --num_classes "$NUM_BINS" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --logging_steps "$LOGGING_STEPS" \
    --save_strategy "$SAVE_STRATEGY" \
    --eval_strategy "$EVAL_STRATEGY" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --load_best_model_at_end "True" \
    --metric_for_best_model "c_index" \
    --greater_is_better "True" \
    --bf16 "$USE_BF16" \
    --dataloader_num_workers "$NUM_WORKERS" \
    --remove_unused_columns "$REMOVE_UNUSED_COLUMNS" \
    --report_to "$REPORT_TO" \
    --seed "$SEED" > "$LOG_FILE" 2>&1 &

echo "✅ 生存预测 5-Fold 训练已成功在后台启动！"
echo "💡 查看日志命令: tail -f $LOG_FILE"