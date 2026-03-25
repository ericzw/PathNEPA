import os

def count_h5_files(target_dir):
    all_physical_paths = []
    unique_filenames = set()

    print(f"🔍 正在扫描 (包含软链接): {target_dir}")

    # followlinks=True 是处理软链接文件夹的关键
    for root, dirs, files in os.walk(target_dir, followlinks=True):
        for file in files:
            if file.endswith('.h5'):
                # 记录物理路径
                full_path = os.path.join(root, file)
                all_physical_paths.append(full_path)
                
                # 记录文件名（用于去重）
                unique_filenames.add(file)

    total_count = len(all_physical_paths)
    unique_count = len(unique_filenames)

    print("\n" + "="*40)
    print(f"📊 物理文件总数: {total_count}")
    print(f"📊 唯一文件名总数: {unique_count}")
    print(f"⚠️  重名冲突文件数: {total_count - unique_count}")
    print("="*40)

    if total_count > unique_count:
        print("\n🚩 结论：存在重名！")
        print(f"如果你直接用 basename 保存到同一个文件夹，最终只会剩下 {unique_count} 个文件。")
    else:
        print("\n✅ 结论：没有重名冲突。")

if __name__ == "__main__":
    # 替换成你的原始数据路径
    MY_DIR = "/data2/mengzibing/medicine/datasets/dataset_o/"
    count_h5_files(MY_DIR)