import os

root_dir = '/data2/mengzibing/medicine/datasets/dataset_o'

print(f"🚀 开始强制穿透扫描 (开启 followlinks): {root_dir}")
print("-" * 60)

grand_total = 0

if not os.path.exists(root_dir):
    print("❌ 根路径不存在！")
else:
    for root, dirs, files in os.walk(root_dir, followlinks=True):
        count_in_dir = 0
        for file in files:
            if file.lower().endswith('.h5'):
                count_in_dir += 1
                
        if count_in_dir > 0:
            print(f"✅ 在 {root.split('/')[-1]} 中找到: {count_in_dir} 个")
            grand_total += count_in_dir

print("-" * 60)
print(f"🎉 最终总计: {grand_total} 个 .h5 文件。")

# ------------------------------------------------------------
# ✅ 在 BRCA 中找到: 838 个
# ✅ 在 TCGA-KIRC 中找到: 519 个
# ✅ 在 TCGA-KICH 中找到: 109 个
# ✅ 在 TCGA-KIRP 中找到: 297 个
# ✅ 在 TCGA-LUAD 中找到: 531 个
# ✅ 在 TCGA-LUSC 中找到: 512 个
# ✅ 在 BRCA 中找到: 838 个
# ✅ 在 LUAD 中找到: 531 个
# ✅ 在 KIRC 中找到: 519 个
# ✅ 在 TCGA-LGG 中找到: 844 个
# ✅ 在 TCGA-GBM 中找到: 858 个
# ✅ 在 HNSC 中找到: 472 个
# ✅ 在 UCEC 中找到: 566 个
# ✅ 在 TCGA-COAD 中找到: 442 个
# ✅ 在 TCGA-READ 中找到: 158 个
# ✅ 在 BRCA 中找到: 838 个
# ✅ 在 LUAD 中找到: 531 个
# ✅ 在 TCGA-LGG 中找到: 844 个
# ✅ 在 TCGA-GBM 中找到: 858 个
# ✅ 在 BLCA 中找到: 457 个
# ✅ 在 UCEC 中找到: 566 个
# ✅ 在 PAAD 中找到: 203 个
# ------------------------------------------------------------
# 🎉 最终总计: 12331 个 .h5 文件。