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
from lifelines.utils import concordance_index
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed
)

# 导入你自定义的类
from models.dataset import DatasetForSur as FastOfflineMILDataset
# 把 import CleanDownstreamMIL 改为：
from models.downstream_surv import SurvDownstreamMIL

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
        file_to_label_dict = {}
        for _, row in df.iterrows():
            # 获取病人 ID
            pid = row['diagnoses.submitter_id']
            # 将 (survival_bin, status, time_months) 打包存入字典
            file_to_label_dict[pid] = (row['survival_bin'], row['status'], row['time_months'])
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
        hazards, labels_combined = eval_pred
        
        # 拆解标签: labels_combined 形状是 [Batch, 3]
        status = labels_combined[:, 1]       # c (事件是否发生: 1或0)
        event_time = labels_combined[:, 2]   # 真实发生时间 (time_months)
        
        # 根据 hazards 计算累积生存概率
        survival = np.cumprod(1 - hazards, axis=1)
        
        # 计算风险得分 Risk Score (生存概率的总和的负数)
        # 模型预测某人活得越久，risk_score 越小
        risk_score = -np.sum(survival, axis=1)
        
        try:
            # concordance_index (真实生存时间, 风险得分, 是否发生事件)
            c_index = concordance_index(event_time, risk_score, status)
        except Exception as e:
            # 若整个 batch 全是删失(0)无死亡(1)，C-Index 会抛出异常
            c_index = 0.5 
            
        return {"c_index": c_index}

    def feature_collate_fn(examples):
        batch_labels = [torch.as_tensor(ex["labels"], dtype=torch.float32) for ex in examples]
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
        # 💡 瞬间组装当前折的训练集纯列表
        train_files, train_labels = [], []
        for pid in train_patients:
            label_tuple = file_to_label_dict[pid] # 这是一个 tuple: (bin, status, time)
            for f in patient_to_files[pid]:
                train_files.append(f)
                train_labels.append(list(label_tuple)) # 💡 【修复】直接转成 list 放进去
                
        # 💡 瞬间组装当前折的验证集纯列表
        val_files, val_labels = [], []
        for pid in val_patients:
            label_tuple = file_to_label_dict[pid]
            for f in patient_to_files[pid]:
                val_files.append(f)
                val_labels.append(list(label_tuple)) # 💡 【修复】直接转成 list 放进去
        
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
        model = SurvDownstreamMIL(
            hidden_size=1536, 
            num_bins=model_args.num_classes, # 💡 【修复】这里把参数名改成 num_bins
            num_layers=2,  
            num_heads=12
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
        
        eval_metrics = trainer.evaluate()
        # 💡 【修复】读取 C-Index 而不是 accuracy
        print(f"\n✅ Fold {fold + 1} 最佳结果 | C-Index: {eval_metrics['eval_c_index']:.4f}")
        
        fold_results.append({
            "fold": fold + 1,
            "c_index": eval_metrics['eval_c_index'] # 💡 记录 C-Index
        })

    # ==========================================
    # 7. 5-Fold 成绩汇总输出
    # ==========================================
    print("\n" + "*"*50)
    print("🎉 5-Fold Cross Validation 全部完成！")
    
    # 💡 提取并打印 C-Index 汇总
    c_index_list = [res["c_index"] for res in fold_results]
    
    for res in fold_results:
        print(f"Fold {res['fold']}: C-Index: {res['c_index']:.4f}")
        
    print("-" * 50)
    print(f"✅ 平均 C-Index : {np.mean(c_index_list):.4f} ± {np.std(c_index_list):.4f}")
    print("*"*50)

if __name__ == "__main__":
    main()