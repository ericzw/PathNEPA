import re
import ast
import matplotlib.pyplot as plt

def parse_and_plot_log(log_file_path):
    fold_data = {}
    current_fold = -1
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # --- 1. 匹配当前是第几折 ---
            fold_match = re.search(r'Fold (\d+) \|', line)
            if not fold_match:
                fold_match = re.search(r'Fold (\d+) / 5', line)
                
            if fold_match:
                current_fold = int(fold_match.group(1))
                if current_fold not in fold_data:
                    fold_data[current_fold] = {
                        'eval_epochs': [], 'eval_loss': [], 'eval_acc': [], 'eval_f1': [],
                        '_step_dict': {}
                    }
                    
            if current_fold == -1:
                continue
                
            # --- 2. 提取 Train Loss ---
            tqdm_match = re.search(r'Epoch (\d+)/\d+: .*?(\d+)/(\d+) \[.*loss=([0-9.]+)', line)
            if tqdm_match:
                current_ep = int(tqdm_match.group(1))
                step = int(tqdm_match.group(2))
                total_steps = int(tqdm_match.group(3))
                loss = float(tqdm_match.group(4))
                exact_epoch = (current_ep - 1) + (step / total_steps)
                fold_data[current_fold]['_step_dict'][exact_epoch] = loss
                
            # --- 3. 提取 Eval 数据 ---
            dict_match = re.search(r'(\{.*?\})', line)
            if dict_match:
                try:
                    data_dict = ast.literal_eval(dict_match.group(1))
                    if 'eval_loss' in data_dict and 'epoch' in data_dict:
                        fold_data[current_fold]['eval_epochs'].append(data_dict['epoch'])
                        fold_data[current_fold]['eval_loss'].append(data_dict['eval_loss'])
                        fold_data[current_fold]['eval_acc'].append(data_dict.get('eval_accuracy', 0))
                        fold_data[current_fold]['eval_f1'].append(data_dict.get('eval_f1_macro', 0))
                except:
                    pass

    # 组装数据
    for fold in fold_data:
        d = fold_data[fold]
        sorted_epochs = sorted(d['_step_dict'].keys())
        d['train_epochs'] = sorted_epochs
        d['train_loss'] = [d['_step_dict'][e] for e in sorted_epochs]

    num_folds = len(fold_data)
    if num_folds == 0 or all(len(d['train_loss']) == 0 for d in fold_data.values()):
        print("❌ 没有提取到数据")
        return

    # 画图
    fig, axes = plt.subplots(nrows=2, ncols=num_folds, figsize=(6 * num_folds, 10))
    if num_folds == 1:
        axes = axes.reshape(2, 1)

    for i, fold in enumerate(sorted(fold_data.keys())):
        d = fold_data[fold]
        
        # ==================== Loss 曲线（对数坐标 + 大范围）====================
        ax_loss = axes[0, i]
        ax_loss.plot(d['train_epochs'], d['train_loss'], label='Train Loss', linewidth=1.5, alpha=0.8)
        if d['eval_epochs']:
            ax_loss.plot(d['eval_epochs'], d['eval_loss'], label='Eval Loss', color='red', linewidth=3, markersize=8)
        
        ax_loss.set_title(f'Fold {fold}: Loss Curve')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss (Log Scale)')
        
        # ✅ 核心：对数坐标 + 超大范围，能看到底部收敛
        ax_loss.set_yscale('log')
        ax_loss.set_ylim(bottom=0.0001, top=100)  # 从 0.0001 ~ 100，全覆盖
        
        ax_loss.legend()
        ax_loss.grid(True, linestyle='--', alpha=0.6)

        # ==================== 指标曲线 ====================
        ax_score = axes[1, i]
        if d['eval_epochs']:
            ax_score.plot(d['eval_epochs'], d['eval_acc'], label='Eval Accuracy', marker='o', color='green')
            ax_score.plot(d['eval_epochs'], d['eval_f1'], label='Eval F1 Macro', marker='s', color='orange')
        ax_score.set_title(f'Fold {fold}: Evaluation Scores')
        ax_score.set_xlabel('Epoch')
        ax_score.set_ylabel('Score')
        ax_score.set_ylim(0, 1.05)
        ax_score.legend()
        ax_score.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('kfold_training_curves.png', dpi=300, bbox_inches='tight')
    print(f"✅ 画图完成！已生成对数坐标清晰图")

if __name__ == "__main__":
    parse_and_plot_log('/data2/mengzibing/Amedicine/PathAILab/train_trans.log')