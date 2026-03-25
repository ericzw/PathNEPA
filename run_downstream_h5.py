import os
import sys
import glob
import torch
import evaluate
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from sklearn.model_selection import KFold

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed
)

# 导入你自定义的类
from models.dataset import FastOfflineMILDataset
from models.vit_nepa import ViTNepaForImageClassification

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
    num_crops: int = field(
        default=16,
        metadata={"help": "每个 WSI 采样的特征 Patch 数量 (影响显存)"}
    )
    mask_ratio: float = field(
        default=0.0,
        metadata={"help": "下游分类任务通常不需要 Mask，默认为 0.0"}
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
        # 假设 csv 中有 'patient_id' 和 'label' 两列
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

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
        return {"accuracy": acc, "f1_macro": f1}

    def feature_collate_fn(examples):
        batch_features = []
        batch_labels = []
        for example in examples:
            feat = example["input_features"] 
            feat_flat = feat.view(-1, feat.shape[-1]) # 展平为长序列
            batch_features.append(feat_flat)
            batch_labels.append(torch.tensor(example["labels"], dtype=torch.long))

        return {
            "input_features": torch.stack(batch_features), 
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
    
    for f in all_h5_files:
        basename = os.path.basename(f)
        for pid in patient_ids:
            # 💡截取前 12 位进行匹配，防止后缀干扰
            core_pid = str(pid)[:12] 
            if core_pid in basename:
                patient_to_files[pid].append(f)
                matched_file_count += 1
                break # 找到归属即跳出内层循环
                
    print(f"✅ 匹配完成！共成功为病人匹配到了 {matched_file_count} 个 H5 文件。")
    if matched_file_count == 0:
        raise ValueError("❌ 匹配失败！没有任何 H5 文件和 CSV 里的病人 ID 对应上，请检查上面的打印格式！")

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
        train_dataset = FastOfflineMILDataset(
            file_paths=train_files,
            labels=train_labels,
            num_crops=data_args.num_crops,
            mask_ratio=data_args.mask_ratio
        )
        
        val_dataset = FastOfflineMILDataset(
            file_paths=val_files,
            labels=val_labels,
            num_crops=data_args.num_crops,
            mask_ratio=data_args.mask_ratio
        )
        
        # 重新初始化模型，加载预训练权重并替换分类头
        model = ViTNepaForImageClassification.from_pretrained(
            model_args.model_name_or_path,
            num_labels=model_args.num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes 
        )
        
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
        print(f"\n✅ Fold {fold + 1} 最佳结果 | Acc: {eval_metrics['eval_accuracy']:.4f} | F1: {eval_metrics['eval_f1_macro']:.4f}")
        
        fold_results.append({
            "fold": fold + 1,
            "accuracy": eval_metrics['eval_accuracy'],
            "f1_macro": eval_metrics['eval_f1_macro']
        })

    # ==========================================
    # 7. 5-Fold 成绩汇总输出
    # ==========================================
    print("\n" + "*"*50)
    print("🎉 5-Fold Cross Validation 全部完成！")
    
    acc_list = [res["accuracy"] for res in fold_results]
    f1_list = [res["f1_macro"] for res in fold_results]
    
    for res in fold_results:
        print(f"Fold {res['fold']}: Acc: {res['accuracy']:.4f} | Macro F1: {res['f1_macro']:.4f}")
        
    print("-" * 50)
    print(f"✅ 平均 Accuracy : {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"✅ 平均 Macro F1 : {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
    print("*"*50)

if __name__ == "__main__":
    main()