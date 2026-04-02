import os
import pandas as pd
import numpy as np
from pathlib import Path

def generate_tcga_survival_tables(input_dir='.', output_dir='./processed_csv', num_bins=4):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 明确你需要的最终队列，以及它们对应的文件名关键字
    target_cohorts = {
        'BLCA': ['blca'],
        'BRCA': ['brca'],
        'GBMLGG': ['gbm', 'lgg'],     # 合并项
        'LUAD': ['luad'],
        'PAAD': ['paad'],
        'UCEC': ['ucec'],
        'COADREAD': ['coad', 'read']  # 合并项
    }
    
    # 获取当前目录下所有的 .tsv 文件
    all_tsv_files = list(Path(input_dir).glob('*.tsv'))
    
    if not all_tsv_files:
        print(f"文件夹 {input_dir} 下未找到任何 .tsv 文件。")
        return

    print("开始匹配、合并并生成生存标签表...\n")

    for cohort, keywords in target_cohorts.items():
        print(f"====== 正在处理队列: {cohort} ======")
        
        # 1. 查找匹配当前队列的所有源文件
        matched_files = []
        for kw in keywords:
            for file_path in all_tsv_files:
                # 忽略大小写进行匹配，例如 'tcga-gbm.tsv' 会被匹配到
                if kw in file_path.name.lower():
                    matched_files.append(file_path)
                    
        if not matched_files:
            print(f"  -> [跳过] 未找到 {cohort} 的相关源文件。\n")
            continue
            
        print(f"  -> 找到源文件: {[f.name for f in matched_files]}")
        
        # 2. 读取并合并数据
        df_list = []
        for f in matched_files:
            try:
                # low_memory=False 防止存在混合数据类型时报警告
                df_temp = pd.read_csv(f, sep='\t', low_memory=False)
                df_list.append(df_temp)
            except Exception as e:
                print(f"  -> [错误] 读取 {f.name} 失败: {e}")
                
        if not df_list:
            continue
            
        df = pd.concat(df_list, ignore_index=True)
        
        # 3. 提取核心列 (更新为 GDC 官方字典表头，加入 submitter_id)
        cols_to_keep = [
            'diagnoses.submitter_id', 
            'demographic.vital_status', 
            'demographic.days_to_death', 
            'diagnoses.days_to_last_follow_up'
        ]
        
        missing_cols = [col for col in cols_to_keep if col not in df.columns]
        if missing_cols:
            print(f"  -> [跳过] 合并后的数据缺失关键列: {missing_cols}\n")
            continue
            
        df_surv = df[cols_to_keep].copy()
        
        # 4. 数据清洗与格式化
        # 扩大异常值的替换范围，把带单引号的情况也加进去（虽然下面用了 to_numeric，但多做一步清洗更好）
        df_surv = df_surv.replace(['--', "'--", 'Not Reported', 'Not Applicable'], np.nan)
        
        # 转换状态标签
        df_surv['status'] = df_surv['demographic.vital_status'].str.lower().map({'alive': 0, 'dead': 1})
        
        # 【核心修改】使用 pd.to_numeric 强制转换，任何无法识别的字符串（比如 "'--" 或其他乱码）都会变成 NaN
        death_days = pd.to_numeric(df_surv['demographic.days_to_death'], errors='coerce')
        follow_up_days = pd.to_numeric(df_surv['diagnoses.days_to_last_follow_up'], errors='coerce')
        
        df_surv['time_days'] = np.where(
            df_surv['status'] == 1,
            death_days,
            follow_up_days
        )
        
        # 剔除无效行并转换时间为月 (因为上面 errors='coerce' 产生了 NaN，这里 dropna 就能完美过滤掉那些乱码样本)
        df_surv = df_surv.dropna(subset=['status', 'time_days'])
        df_surv = df_surv[df_surv['time_days'] >= 0]
        df_surv['time_months'] = df_surv['time_days'] / 30.4
            
        # 5. 基于合并后的数据统一进行分箱 (Bins = 4)
        try:
            df_surv['survival_bin'], bin_edges = pd.qcut(
                df_surv['time_months'], 
                q=num_bins, 
                labels=False, 
                retbins=True,
                duplicates='drop'
            )
        except Exception as e:
            print(f"  -> [错误] {cohort} 分箱失败: {e}\n")
            continue
        
        # 6. 保存为最终的目标文件
        final_file = out_path / f"{cohort}.csv"
        final_cols = ['diagnoses.submitter_id', 'demographic.vital_status', 'status', 'time_months', 'survival_bin']
        df_surv[final_cols].to_csv(final_file, index=False)
        
        print(f"  -> [成功] 共保留 {len(df_surv)} 个有效样本。")
        print(f"  -> {num_bins}个Bins的切分边界(月): {np.round(bin_edges, 2)}")
        print(f"  -> 已生成: {final_file}\n")

if __name__ == "__main__":
    generate_tcga_survival_tables(
        input_dir='/data2/mengzibing/medicine/datasets/dataset_o/A-source_label/tsv',
        output_dir='/data2/mengzibing/medicine/datasets/dataset_o/A-source_label/survival_prediction/bins/',
        num_bins=4
    )