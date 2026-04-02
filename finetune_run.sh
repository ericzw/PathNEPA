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
DATASET_NAME="RCC"            # 数据集名称 (仅用于日志和输出文件夹命名)
ROOT_DIR="/data2/mengzibing/medicine"
CODE_DIR="${ROOT_DIR}/PathNEPA"
DATA_DIR="${ROOT_DIR}/datasets/dataset_o/Sub-typing/RCC"
CLINICAL_FILE="/data2/mengzibing/medicine/datasets/dataset_o/A-source_label/subtyping/rcc_subtyping_labels.csv" # 替换为真实的临床标签 CSV 路径
OUTPUT_DIR="./output_${DATASET_NAME}2"
LOG_FILE="${OUTPUT_DIR}/${DATASET_NAME}_cv.log"

# ==========================================
# 🧠 第三部分：模型与特征配置
# ==========================================
PRETRAINED_WEIGHTS="SixAILab/nepa-base-patch14-224" # ⚠️ 聚合网络是随机初始化的，这个参数目前仅作占位符防报错
NUM_CLASSES=3                            # 下游亚型分类的类别数

# ==========================================
# 🚀 第四部分：核心训练超参数 (Downstream MIL Aggregation)
# ==========================================
EPOCHS=30                                # MIL 通常需要稍微多一点 Epoch (建议 30-50)
BATCH_SIZE=1                             # ⚠️ 训练 Batch Size 强锁 1，防止动态序列导致 OOM
EVAL_BATCH_SIZE=1                        # ⚠️ 验证 Batch Size 强锁 1，防止评测时 Padding 导致维度崩溃
GRAD_ACCUM_STEPS=16                      # 💡 梯度累加：逻辑 Batch Size = 1 * 16 = 16，保证收敛极其稳定
LEARNING_RATE=0.0002                     # 下游聚合网络是从头训练的，LR 设为 2e-4 比较合适
WEIGHT_DECAY=0.05                        # 权重衰减，防止小样本医学数据过拟合
MAX_GRAD_NORM=1.0                        # 💡 极其重要：序列长达上万，极易梯度爆炸，必须强制裁剪梯度
USE_BF16="True"                          # 是否使用 BF16 半精度 
SEED=42                                  # 随机种子，确保 5-fold 划分可复现

# ==========================================
# 📊 第五部分：数据加载与日志策略
# ==========================================
NUM_WORKERS=8                            # DataLoader 的 CPU 线程数
LOGGING_STEPS=10                         # 每隔多少步打印一次日志
SAVE_STRATEGY="epoch"                    # 保存策略
EVAL_STRATEGY="epoch"                    # 验证策略
SAVE_TOTAL_LIMIT=1                       # ⚠️ 每折只保留最好的一组权重，节省硬盘空间
REPORT_TO="tensorboard"                  # 可视化工具
REMOVE_UNUSED_COLUMNS="False"            # 自定义 Dataset 必须设为 False

# ==========================================
# 🎬 启动执行区 (无需修改)
# ==========================================
# 确保输出文件夹存在，避免日志文件创建失败
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "🚀 准备启动下游 Subtyping 5-Fold 交叉验证..."
echo "🖥️  使用 GPU: $CUDA_VISIBLE_DEVICES"
echo "📂 数据路径: $DATA_DIR"
echo "🏷️  标签文件: $CLINICAL_FILE"
echo "🔢 类别数量: $NUM_CLASSES"
echo "📦 物理 BS: $BATCH_SIZE | 逻辑 BS (含累加): $(($BATCH_SIZE * $GRAD_ACCUM_STEPS))"
echo "💾 输出路径: $OUTPUT_DIR"
echo "📝 日志文件: $LOG_FILE"
echo "========================================="

# 切换到代码目录
cd "$CODE_DIR"

# 放到后台运行，并将所有输出重定向到 LOG_FILE
nohup python run_downstream_h5.py \
    --model_name_or_path "$PRETRAINED_WEIGHTS" \
    --data_dir "$DATA_DIR" \
    --clinical_file "$CLINICAL_FILE" \
    --num_classes "$NUM_CLASSES" \
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
    --metric_for_best_model "f1_macro" \
    --greater_is_better "True" \
    --bf16 "$USE_BF16" \
    --dataloader_num_workers "$NUM_WORKERS" \
    --remove_unused_columns "$REMOVE_UNUSED_COLUMNS" \
    --report_to "$REPORT_TO" \
    --seed "$SEED" > "$LOG_FILE" 2>&1 &

echo "✅ 5-Fold 训练已成功在后台启动！"
echo "💡 提示：你可以使用以下命令实时查看训练进度（将会看到每一折的 F1 成绩跑分）："
echo "tail -f $LOG_FILE"