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
            # 兼容 "🔥 正在训练 Fold 1 / 5" 或 "Fold 0 | Epoch..."
            fold_match = re.search(r'Fold (\d+) \|', line)
            if not fold_match:
                fold_match = re.search(r'Fold (\d+) / 5', line)
                
            if fold_match:
                current_fold = int(fold_match.group(1))
                if current_fold not in fold_data:
                    fold_data[current_fold] = {
                        'eval_epochs': [], 'eval_loss': [], 'eval_acc': [], 'eval_f1': [],
                        '_step_dict': {} # 💡 神奇的去重字典：{ exact_epoch: loss }
                    }
                    
            if current_fold == -1:
                continue
                
            # --- 2. 提取 Train Loss (专门针对进度条格式) ---
            # 匹配: "Epoch 1/50:  24%|...| 157/656 [06:53<20:50,  2.51s/it, loss=0.0105]"
            tqdm_match = re.search(r'Epoch (\d+)/\d+: .*?(\d+)/(\d+) \[.*loss=([0-9.]+)', line)
            if tqdm_match:
                current_ep = int(tqdm_match.group(1))
                step = int(tqdm_match.group(2))
                total_steps = int(tqdm_match.group(3))
                loss = float(tqdm_match.group(4))
                
                # 计算出极其精确的 X 轴坐标 (例如 Epoch 1 的第 157/656 步 = X轴 0.239)
                exact_epoch = (current_ep - 1) + (step / total_steps)
                
                # 写入字典！同一个 Step 重复打印时，后面的会自动覆盖前面的（完美解决重影）
                fold_data[current_fold]['_step_dict'][exact_epoch] = loss
                
            # --- 3. 提取 Eval 字典 (不变，用于画验证集红点) ---
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

    # ================= 数据组装与画图部分 =================
    num_folds = len(fold_data)
    
    # 转换字典数据为列表
    for fold in fold_data:
        d = fold_data[fold]
        sorted_epochs = sorted(d['_step_dict'].keys())
        d['train_epochs'] = sorted_epochs
        d['train_loss'] = [d['_step_dict'][e] for e in sorted_epochs]

    if num_folds == 0 or all(len(d['train_loss']) == 0 for d in fold_data.values()):
        print("❌ 依然没有提取到数据，请把最新的 Log 截取几行发来看看！")
        return

    # 创建画布
    fig, axes = plt.subplots(nrows=2, ncols=num_folds, figsize=(6 * num_folds, 10))
    if num_folds == 1:
        axes = axes.reshape(2, 1)

    for i, fold in enumerate(sorted(fold_data.keys())):
        d = fold_data[fold]
        
        if not d['train_epochs']:
            continue
            
        # --- 第一排：Loss 曲线 ---
        ax_loss = axes[0, i]
        ax_loss.plot(d['train_epochs'], d['train_loss'], label='Train Loss', marker='o', markersize=4, alpha=0.7)
        if d['eval_epochs']:
            ax_loss.plot(d['eval_epochs'], d['eval_loss'], label='Eval Loss', marker='s', color='red', markersize=6, linewidth=0)
        ax_loss.set_title(f'Fold {fold}: Loss Curve')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True, linestyle='--', alpha=0.6)

        # --- 第二排：Accuracy / F1 曲线 ---
        ax_score = axes[1, i]
        if d['eval_epochs']:
            ax_score.plot(d['eval_epochs'], d['eval_acc'], label='Eval Accuracy', marker='^', color='green')
            ax_score.plot(d['eval_epochs'], d['eval_f1'], label='Eval F1 Macro', marker='v', color='orange')
        ax_score.set_title(f'Fold {fold}: Evaluation Scores')
        ax_score.set_xlabel('Epoch')
        ax_score.set_ylabel('Score (0 to 1)')
        ax_score.set_ylim([0, 1.05])
        ax_score.legend()
        ax_score.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('kfold_training_curves.png', dpi=300)
    print(f"✅ 画图完成！共成功解析了 {num_folds} 个 Fold 的数据。已保存为 kfold_training_curves.png")

# 运行脚本
if __name__ == "__main__":
    # 换成你的 log 路径
    parse_and_plot_log('/data2/mengzibing/Amedicine/PathAILab/train_trans.log')