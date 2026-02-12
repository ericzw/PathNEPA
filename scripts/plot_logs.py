import json
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg') # 适配 AutoDL 无显示器环境

def plot_training_results(json_path):
    if not os.path.exists(json_path):
        print(f"❌ 找不到文件: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    log_history = data.get("log_history", [])
    print(f"📊 总共读取到 {len(log_history)} 条日志记录")

    train_steps, train_loss, lrs = [], [], []
    eval_steps, eval_loss = [], []

    for entry in log_history:
        step = entry.get("step")
        # 兼容处理：检查多种可能的 Loss 键名
        t_loss = entry.get("loss") or entry.get("train_loss")
        e_loss = entry.get("eval_loss")
        
        if t_loss is not None:
            train_steps.append(step)
            train_loss.append(t_loss)
            if "learning_rate" in entry:
                lrs.append(entry["learning_rate"])
        
        if e_loss is not None:
            eval_steps.append(step)
            eval_loss.append(e_loss)

    print(f"📈 提取到训练点数: {len(train_steps)}")
    print(f"📉 提取到验证点数: {len(eval_steps)}")

    if len(train_steps) == 0 and len(eval_steps) == 0:
        print("⚠️ 警告：没有提取到任何 Loss 数据，请检查 JSON 内容！")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- 左轴：Loss ---
    ax1.set_xlabel('Global Step')
    ax1.set_ylabel('Loss', color='tab:red')
    
    # 使用 marker='o' 确保即使点很少也能看见
    if train_loss:
        ax1.plot(train_steps, train_loss, color='tab:red', linestyle='--', 
                 marker='.', alpha=0.6, label='Train Loss')
    if eval_loss:
        ax1.plot(eval_steps, eval_loss, color='tab:red', marker='o', 
                 markersize=8, linewidth=2, label='Eval Loss')
    
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, linestyle=':', alpha=0.5)

    # --- 右轴：LR ---
    if lrs:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Learning Rate', color='tab:blue')
        # 补齐 LR 的步数（有些 log 没带 LR）
        ax2.plot(train_steps[:len(lrs)], lrs, color='tab:blue', label='LR')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('NEPA Training Debug Plot')
    fig.tight_layout()
    
    output_name = "debug_training_curves.png"
    plt.savefig(output_name, dpi=300)
    print(f"✅ 图片已更新并保存为: {output_name}")

if __name__ == "__main__":
    plot_training_results('/root/autodl-tmp/nepa/codes/nepa-main/output_pretrain_32x32_ema_1/trainer_state.json')