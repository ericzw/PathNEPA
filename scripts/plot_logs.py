import re
import ast
import matplotlib.pyplot as plt

def parse_and_plot_log(log_file_path):
    # 存储解析后的数据
    fold_data = {}
    current_fold = 0
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 1. 匹配当前是第几折
            fold_match = re.search(r'🔥 正在训练 Fold (\d+) / 5', line)
            if fold_match:
                current_fold = int(fold_match.group(1))
                fold_data[current_fold] = {
                    'train_epochs': [], 'train_loss': [],
                    'eval_epochs': [], 'eval_loss': [], 'eval_acc': [], 'eval_f1': []
                }
                continue
                
            # 如果还没开始第一折，跳过
            if current_fold == 0:
                continue
                
            # 2. 匹配 Training Loss 字典
            if line.startswith("{'loss':"):
                try:
                    data_dict = ast.literal_eval(line)
                    fold_data[current_fold]['train_epochs'].append(data_dict['epoch'])
                    fold_data[current_fold]['train_loss'].append(data_dict['loss'])
                except:
                    pass
                    
            # 3. 匹配 Evaluation 字典
            elif line.startswith("{'eval_loss':"):
                try:
                    data_dict = ast.literal_eval(line)
                    fold_data[current_fold]['eval_epochs'].append(data_dict['epoch'])
                    fold_data[current_fold]['eval_loss'].append(data_dict['eval_loss'])
                    fold_data[current_fold]['eval_acc'].append(data_dict['eval_accuracy'])
                    fold_data[current_fold]['eval_f1'].append(data_dict['eval_f1_macro'])
                except:
                    pass

    # ================= 画图部分 =================
    num_folds = len(fold_data)
    if num_folds == 0:
        print("没有在 Log 中找到有效数据！")
        return

    fig, axes = plt.subplots(nrows=2, ncols=num_folds, figsize=(6 * num_folds, 10))
    if num_folds == 1:
        axes = axes.reshape(2, 1)

    for i, fold in enumerate(sorted(fold_data.keys())):
        d = fold_data[fold]
        
        # --- 第一排：Loss 曲线 ---
        ax_loss = axes[0, i]
        ax_loss.plot(d['train_epochs'], d['train_loss'], label='Train Loss', color='#1f77b4', alpha=0.8, linewidth=1.5)
        if d['eval_epochs']:
            ax_loss.plot(d['eval_epochs'], d['eval_loss'], label='Eval Loss', color='#ff4b33', linewidth=3)
        
        ax_loss.set_title(f'Fold {fold}: Loss Curve', fontsize=12, fontweight='bold')
        ax_loss.set_xlabel('Epoch', fontsize=11)
        ax_loss.set_ylabel('Loss (Log Scale)', fontsize=11)
        
        # ✅ 核心：对数坐标 + 足够大的纵轴范围，能看到底部所有点
        ax_loss.set_yscale('log')
        ax_loss.set_ylim(bottom=0.001, top=100)  # 从 0.001 ~ 100，覆盖你所有loss范围
        # # ax_loss.set_yscale('log')
        # ax_loss.set_ylim(bottom=1e-5, top=3)  # 只看0.0001~2的区间，放大底部
        ax_loss.legend(fontsize=10)
        ax_loss.grid(True, linestyle='--', alpha=0.6)   

        # --- 第二排：Accuracy / F1 曲线 ---
        ax_score = axes[1, i]
        if d['eval_epochs']:
            ax_score.plot(d['eval_epochs'], d['eval_acc'], label='Eval Accuracy', marker='o', color='#2ca02c', linewidth=2)
            ax_score.plot(d['eval_epochs'], d['eval_f1'], label='Eval F1 Macro', marker='s', color='#ff7f0e', linewidth=2)
        ax_score.set_title(f'Fold {fold}: Evaluation Scores', fontsize=12, fontweight='bold')
        ax_score.set_xlabel('Epoch', fontsize=11)
        ax_score.set_ylabel('Score (0 ~ 1)', fontsize=11)
        ax_score.set_ylim([0, 1.05])
        ax_score.legend(fontsize=10)
        ax_score.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('kfold_training_curves.png', dpi=300, bbox_inches='tight')
    print("✅ 画图完成！已保存为 kfold_training_curves.png")

# 运行
if __name__ == "__main__":
    parse_and_plot_log('/data2/mengzibing/Amedicine/PathNEPA/output_RCC2/RCC2_cv.log')