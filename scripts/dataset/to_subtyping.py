import pandas as pd
from pathlib import Path
import os
import sys

# ================= 1. 路径与参数配置 =================
# BRCA 通常合并在一个单一的 TCGA 项目中
tsv_files_list = [
    "/data2/mengzibing/Amedicine/dataset/tcga-label/tsv/tcga-brca.tsv"
]
# 你的 .h5 特征目录 (包含软连接)
h5_dataset_dir = "/data2/mengzibing/Amedicine/dataset/tcga-feature-clean/Sub-typing"
# 最终生成的全局标签文件
output_csv = "/data2/mengzibing/Amedicine/dataset/tcga-label-cleaned/subtyping/BRCA_global_labels.csv"

id_col = "cases.submitter_id"            
label_col = "diagnoses.primary_diagnosis" 

def process_brca_labels():
    # ================= 2. 读取临床标签 (TSV) =================
    print("1. 正在读取 BRCA 临床标签数据...")
    df_list = []
    for tsv_path_str in tsv_files_list:
        file_path = Path(tsv_path_str)
        if file_path.exists():
            df_temp = pd.read_csv(file_path, sep='\t', low_memory=False)
            if id_col in df_temp.columns and label_col in df_temp.columns:
                df_list.append(df_temp[[id_col, label_col]])
        else:
            print(f"   [警告] 找不到文件: {file_path}")
                
    if not df_list:
        print("[错误] 未读取到任何有效的 TSV 数据，程序终止。")
        sys.exit(1)

    df_labels = pd.concat(df_list, ignore_index=True)
    df_labels = df_labels.drop_duplicates(subset=[id_col])
    df_labels = df_labels[df_labels[label_col] != "'--"]
    df_labels.rename(columns={id_col: 'patient_id', label_col: 'raw_diagnosis'}, inplace=True)

    # ================= 3. 强制跟随软连接扫描 .h5 =================
    print("\n2. 正在穿透软连接扫描特征文件夹，并过滤有病(01)样本...")
    h5_records = []
    
    for root, dirs, files in os.walk(h5_dataset_dir, followlinks=True):
        for filename in files:
            if filename.endswith('.h5'):
                patient_id = filename[:12]         
                sample_type = filename[13:15]      
                
                # 只保留 '01' 原发实体瘤
                if sample_type == '01':
                    h5_records.append({'slide_id': filename, 'patient_id': patient_id})
            
    df_h5 = pd.DataFrame(h5_records, columns=['slide_id', 'patient_id'])
    print(f"   扫描完成，共找到 {len(df_h5)} 个属于 '01' 的 .h5 样本。")
    
    if len(df_h5) == 0:
        print("[错误] 穿透软连接后依然为 0！请检查源文件是否丢失或后缀名不匹配。")
        sys.exit(1)

    # ================= 4. 匹配与 BRCA 二分类映射 =================
    print("\n3. 正在进行交集匹配并生成 0/1 标签 (IDC vs ILC)...")
    df_merged = pd.merge(df_h5, df_labels, on='patient_id', how='inner')
    
    def map_brca_subtype(diag):
        diag_lower = str(diag).lower()
        
        # 严格过滤：排除混合型癌 (同时包含 duct 和 lobular，或者写了 mixed)
        if 'mixed' in diag_lower or ('duct' in diag_lower and 'lobular' in diag_lower):
            return -1
            
        # 浸润性导管癌 (IDC) -> 匹配 'duct' 即可囊括 "Infiltrating duct carcinoma"
        if 'duct' in diag_lower:
            return 0  
        # 浸润性小叶癌 (ILC) -> 匹配 'lobular' 囊括 "Lobular carcinoma"
        elif 'lobular' in diag_lower:
            return 1  
        else:
            # 其他罕见类型剔除
            return -1
            
    df_merged['target'] = df_merged['raw_diagnosis'].apply(map_brca_subtype)
    df_clean = df_merged[df_merged['target'] != -1].copy()
    df_clean['target'] = df_clean['target'].astype(int)

    # ================= 5. 直接输出供 Dataloader 用的单文件 =================
    final_df = df_clean[['slide_id', 'target']]
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, index=False)

    print("\n================ BRCA 处理完成 ================")
    print(f"有效切片总数: {len(final_df)} 张")
    print(f" - 类别 0 (IDC 浸润性导管癌): {len(final_df[final_df['target'] == 0])} 张")
    print(f" - 类别 1 (ILC 浸润性小叶癌): {len(final_df[final_df['target'] == 1])} 张")
    
    dropped_mixed = len(df_merged[df_merged['target'] == -1])
    if dropped_mixed > 0:
        print(f" [注] 为了保证纯度，已自动剔除 {dropped_mixed} 张混合型或其他罕见亚型的切片。")
        
    print(f"结果已保存至: {output_csv}")

if __name__ == "__main__":
    process_brca_labels()