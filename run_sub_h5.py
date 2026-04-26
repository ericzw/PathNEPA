import os
import sys
import glob
import evaluate
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from sklearn.model_selection import KFold
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed
)

# 导入你自定义的类
from models.dataset import FastOfflineMILDataset
from models.vit_nepa.configuration_vit_nepa import ViTNepaConfig
from models.vit_nepa.modeling_vit_nepa import ViTNepaForSubtypingClassification
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 定义参数类 (Model & Data Arguments)
# ==========================================
@dataclass
class ModelArguments:
    """关于模型和权重的参数"""
    model_name_or_path: str = field(
        metadata={"help": "预训练模型的路径 (例如 ./output_pretrain/checkpoint-xxx)"}
    )
    num_classes: int = field(
        default=3,
        metadata={"help": "下游任务的类别数量"}
    )
    ignore_mismatched_sizes: bool = field(
        default=True,
        metadata={"help": "是否忽略尺寸不匹配的权重 (微调分类头时必须为 True)"}
    )

@dataclass
class DataArguments:
    """关于数据和加载的参数"""
    data_dir: str = field(
        metadata={"help": "存放离线 .h5 特征文件的主目录"}
    )
    clinical_file: str = field(
        default=None,
        metadata={"help": "包含 patient_id 和 label 的 CSV 临床文件路径"}
    )
    
def main():
    # ==========================================
    # 2. 解析命令行参数
    # ==========================================
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 设置随机种子，保证可复现性
    set_seed(training_args.seed)

    # ==========================================
    # 3. 临床数据与标签挂载
    # ==========================================
    print(f"📊 正在加载临床标签数据...")
    
    # 建立标签映射字典
    label2id = {f"{i}": i for i in range(model_args.num_classes)}
    id2label = {v: k for k, v in label2id.items()}

    if data_args.clinical_file and os.path.exists(data_args.clinical_file):
        df = pd.read_csv(data_args.clinical_file)
        file_to_label_dict = dict(zip(df['patient_id'], df['label']))
    else:
        print("⚠️ 未提供 clinical_file，使用 Mock 数据进行测试...")
        all_patients = [f"patient_{str(i).zfill(3)}" for i in range(1, 101)]
        mock_labels = np.random.choice(list(label2id.keys()), size=100)
        file_to_label_dict = {p: l for p, l in zip(all_patients, mock_labels)}
    
    patient_ids = np.array(list(file_to_label_dict.keys()))

    # ==========================================
    # 4. 评测指标与 Collate 函数
    # ==========================================
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    # 💡 修改这里：增加 AUC 计算
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        
        # 1. 计算硬标签的 Acc 和 F1
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
        
        # 2. 计算软概率的 AUC
        probs = softmax(logits, axis=-1)
        num_classes = logits.shape[-1] # 自动获取当前任务的类别数
        
        try:
            if num_classes == 2:
                # 🌟 二分类专属通道：只传入正类的概率 (索引为 1 的那一列)
                auc = roc_auc_score(y_true=labels, y_score=probs[:, 1])
            else:
                # 🌟 多分类专属通道：传入整个矩阵，并指定 ovr 和所有类别标签
                all_classes = list(range(num_classes))
                auc = roc_auc_score(
                    y_true=labels, 
                    y_score=probs, 
                    multi_class='ovr', 
                    average='macro',
                    labels=all_classes
                )
        except Exception as e:
            # 打印致命报错，防止静默失败变 0
            print(f"\n⚠️ [DEBUG] AUC 计算报错: {e}") 
            auc = 0.0

        return {"accuracy": acc, "f1_macro": f1, "auc_macro": auc}

    def feature_collate_fn(examples):
        batch_labels = [ex["labels"] for ex in examples]
        MAX_SEQ_LENGTH = 10000  # 防 OOM 截断长度
        
        features_list = []
        coords_list = [] 
        
        for ex in examples:
            feat = ex["input_features"] 
            # 1. 强制展平特征：去掉多余的 batch 维度，变成纯粹的 [M, 1536]
            feat_flat = feat.view(-1, feat.shape[-1])
            
            coord = ex["coords"]
            # 💡【核心修复】2. 强制展平坐标：变成纯粹的 [M, 2]
            coord_flat = coord.view(-1, coord.shape[-1])
            
            # 💡【安全网】如果 H5 里压根没存坐标导致长度不对，自动生成全 0 坐标防崩溃
            if coord_flat.shape[0] != feat_flat.shape[0]:
                coord_flat = torch.zeros((feat_flat.shape[0], 2), dtype=torch.float32)
            
            # 3. 同步截断
            if feat_flat.shape[0] > MAX_SEQ_LENGTH:
                feat_flat = feat_flat[:MAX_SEQ_LENGTH, :]
                coord_flat = coord_flat[:MAX_SEQ_LENGTH, :] 
                
            features_list.append(feat_flat)
            coords_list.append(coord_flat)

        # 动态 Padding
        if len(features_list) == 1:
            batch_features = features_list[0].unsqueeze(0) 
            batch_coords = coords_list[0].unsqueeze(0) # 💡 现在稳定是 3D: [1, M, 2]
            attention_mask = torch.ones(1, batch_features.size(1), dtype=torch.long)
        else:
            batch_features = torch.nn.utils.rnn.pad_sequence(features_list, batch_first=True)
            batch_coords = torch.nn.utils.rnn.pad_sequence(coords_list, batch_first=True) 
            
            attention_mask = torch.zeros(batch_features.shape[:2], dtype=torch.long)
            for i, feat in enumerate(features_list):
                attention_mask[i, :feat.size(0)] = 1

        return {
            "input_features": batch_features,
            "coords": batch_coords, # 喂给模型
            "attention_mask": attention_mask,
            "labels": torch.stack(batch_labels)
        }

    # ==========================================
    # 5. 全局扫盘与文件匹配 (仅执行一次)
    # ==========================================
    print("\n🔍 正在执行全局硬盘扫描 ...")
    all_h5_files = glob.glob(os.path.join(data_args.data_dir, "**", "*.h5"), recursive=True)
    
    print(f"📂 硬盘共找到 {len(all_h5_files)} 个 .h5 文件")
    if len(all_h5_files) > 0:
        print(f"📄 样例 H5 文件名: {os.path.basename(all_h5_files[0])}")
    print(f"🧑‍⚕️ 样例 CSV 病人 ID: {patient_ids[0]}")

    
    print("🧩 正在进行全局文件与标签匹配...")
    patient_to_files = {pid: [] for pid in patient_ids}
    matched_file_count = 0
    
    # ================= 💡 终极 DEBUG 打印 =================
    print("\n" + "!"*50)
    print(f"【DEBUG】从 CSV 提取的头 3 个原始 ID: {patient_ids[:3]}")
    if len(all_h5_files) > 0:
        print(f"【DEBUG】从硬盘读取的头 3 个 H5 文件名: {[os.path.basename(f) for f in all_h5_files[:3]]}")
    print("!"*50 + "\n")
    # ======================================================

    for f in all_h5_files:
        basename = os.path.basename(f)
        for pid in patient_ids:
            # 💡【核心修复】：
            # 1. str(pid) 转字符串
            # 2. .strip() 去除首尾所有看不见的空格、换行符
            # 3. .upper() 强制大写，防止 tcga 和 TCGA 的差异
            # 4. [:12] 截取前 12 位
            core_pid = str(pid).strip().upper()[:12] 
            
            # basename 同样转大写后进行匹配
            if core_pid in basename.upper():
                patient_to_files[pid].append(f)
                matched_file_count += 1
                break # 找到归属即跳出内层循环
                
    print(f"✅ 匹配完成！共成功为病人匹配到了 {matched_file_count} 个 H5 文件。")
    if matched_file_count == 0:
        raise ValueError("❌ 匹配失败！没有任何 H5 文件和 CSV 里的病人 ID 对应上，请检查上面的打印格式！")
        
    valid_patient_ids = [pid for pid in patient_ids if len(patient_to_files[pid]) > 0]
    print(f"🧹 数据清洗：CSV 共有 {len(patient_ids)} 人，实际拥有 H5 文件的有 {len(valid_patient_ids)} 人。")
    
    # 覆盖原来的 patient_ids，只用有效病人做后续的 5-Fold
    patient_ids = np.array(valid_patient_ids)
    
    if len(patient_ids) == 0:
        raise ValueError("❌ 过滤后可用病人数量为 0，请检查匹配逻辑！")
    # ==========================================
    # 6. 5-Fold 交叉验证主循环
    # ==========================================
    kf = KFold(n_splits=5, shuffle=True, random_state=training_args.seed)
    fold_results = []
    
    base_output_dir = training_args.output_dir # 记住外层传入的输出根目录

    print(f"\n🚀 开始 5-Fold 下游任务验证 (总样本数: {len(patient_ids)})")

    for fold, (train_idx, val_idx) in enumerate(kf.split(patient_ids)):
        print("\n" + "="*50)
        print(f"🔥 正在训练 Fold {fold + 1} / 5")
        print("="*50)
        
        train_patients = patient_ids[train_idx]
        val_patients = patient_ids[val_idx]
        
        # 💡 瞬间组装当前折的训练集纯列表
        train_files, train_labels = [], []
        for pid in train_patients:
            label = file_to_label_dict[pid]
            # 确保将字符串标签转为 ID 整数
            label_id = label2id.get(str(label), label) if isinstance(label, str) else label
            for f in patient_to_files[pid]:
                train_files.append(f)
                train_labels.append(label_id)
                
        # 💡 瞬间组装当前折的验证集纯列表
        val_files, val_labels = [], []
        for pid in val_patients:
            label = file_to_label_dict[pid]
            label_id = label2id.get(str(label), label) if isinstance(label, str) else label
            for f in patient_to_files[pid]:
                val_files.append(f)
                val_labels.append(label_id)
        
        # 使用组装好的纯列表实例化 Dataset
        # 1. 实例化极简的 Dataset (删掉 num_crops 和 mask_ratio)
        train_dataset = FastOfflineMILDataset(
            file_paths=train_files,
            labels=train_labels
        )
        
        val_dataset = FastOfflineMILDataset(
            file_paths=val_files,
            labels=val_labels
        )
        
        # 2. 实例化我们新写的聚合模型
        config = ViTNepaConfig(
            input_feat_dim=1536,
            hidden_size=1536,
            num_hidden_layers=2,
            num_attention_heads=12,
            intermediate_size=1536 * 4,
            num_labels=model_args.num_classes,
        )

        model = ViTNepaForSubtypingClassification(config)
        
        # 修改当前 Fold 的专属输出路径
        training_args.output_dir = os.path.join(base_output_dir, f"fold_{fold + 1}")
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=feature_collate_fn,
        )
        
        # 开始训练
        trainer.train()
        
        # 评估并记录成绩
        eval_metrics = trainer.evaluate()
        
        # 💡 修改这里：加上 AUC 的打印
        print(f"\n✅ Fold {fold + 1} 最佳结果 | Acc: {eval_metrics['eval_accuracy']:.4f} | F1: {eval_metrics['eval_f1_macro']:.4f} | AUC: {eval_metrics['eval_auc_macro']:.4f}")
        
        fold_results.append({
            "fold": fold + 1,
            "accuracy": eval_metrics['eval_accuracy'],
            "f1_macro": eval_metrics['eval_f1_macro'],
            "auc_macro": eval_metrics['eval_auc_macro'] # 💡 记录 AUC
        })

    # ==========================================
    # 7. 5-Fold 成绩汇总输出
    # ==========================================
    print("\n" + "*"*50)
    print("🎉 5-Fold Cross Validation 全部完成！")
    
    acc_list = [res["accuracy"] for res in fold_results]
    f1_list = [res["f1_macro"] for res in fold_results]
    auc_list = [res["auc_macro"] for res in fold_results] # 💡 提取 AUC 列表
    
    for res in fold_results:
        print(f"Fold {res['fold']}: Acc: {res['accuracy']:.4f} | Macro F1: {res['f1_macro']:.4f} | Macro AUC: {res['auc_macro']:.4f}")
        
    print("-" * 50)
    print(f"✅ 平均 Accuracy : {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"✅ 平均 Macro F1 : {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
    print(f"✅ 平均 Macro AUC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}") # 💡 打印平均 AUC
    print("*"*50)
    
if __name__ == "__main__":
    main()