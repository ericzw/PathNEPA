#!/bin/bash

# 你的路径
inputdir="/data2/mengzibing/Amedicine/dataset/tcga-feature"
outputdir="/data2/mengzibing/Amedicine/dataset/tcga-feature-clean"

# ==================== Sub-typing ====================
mkdir -p "$outputdir/Sub-typing/NSCLC"
ln -sfn "$inputdir/LUSC"  "$outputdir/Sub-typing/NSCLC/LUSC"
ln -sfn "$inputdir/LUAD"  "$outputdir/Sub-typing/NSCLC/LUAD"

mkdir -p "$outputdir/Sub-typing/RCC"
ln -sfn "$inputdir/KIRC"  "$outputdir/Sub-typing/RCC/KIRC"
ln -sfn "$inputdir/KICH"  "$outputdir/Sub-typing/RCC/KICH"
ln -sfn "$inputdir/KIRP"  "$outputdir/Sub-typing/RCC/KIRP"

mkdir -p "$outputdir/Sub-typing/BRCA"
ln -sfn "$inputdir/BRCA-IDC"    "$outputdir/Sub-typing/BRCA/BRCA-IDC"
ln -sfn "$inputdir/BRCA-OTHERS" "$outputdir/Sub-typing/BRCA/BRCA-OTHERS"

# ==================== Survival Prediction ====================
mkdir -p "$outputdir/Survival Prediction/BRCA"
ln -sfn "$inputdir/BRCA-IDC"    "$outputdir/Survival Prediction/BRCA/BRCA-IDC"
ln -sfn "$inputdir/BRCA-OTHERS" "$outputdir/Survival Prediction/BRCA/BRCA-OTHERS"

ln -sfn "$inputdir/BLCA"  "$outputdir/Survival Prediction/BLCA"

mkdir -p "$outputdir/Survival Prediction/GBMLGG"
ln -sfn "$inputdir/GBM"  "$outputdir/Survival Prediction/GBMLGG/GBM"
ln -sfn "$inputdir/LGG"  "$outputdir/Survival Prediction/GBMLGG/LGG"

ln -sfn "$inputdir/LUAD"  "$outputdir/Survival Prediction/LUAD"
ln -sfn "$inputdir/PAAD"  "$outputdir/Survival Prediction/PAAD"
ln -sfn "$inputdir/UCEC"  "$outputdir/Survival Prediction/UCEC"

mkdir -p "$outputdir/Survival Prediction/COADREAD"
ln -sfn "$inputdir/COAD"  "$outputdir/Survival Prediction/COADREAD/COAD"
ln -sfn "$inputdir/READ"  "$outputdir/Survival Prediction/COADREAD/READ"

# ==================== Gene Mutation Prediction ====================
mkdir -p "$outputdir/Gene Mutation Prediction/BRCA"
ln -sfn "$inputdir/BRCA-IDC"    "$outputdir/Gene Mutation Prediction/BRCA/BRCA-IDC"
ln -sfn "$inputdir/BRCA-OTHERS" "$outputdir/Gene Mutation Prediction/BRCA/BRCA-OTHERS"

mkdir -p "$outputdir/Gene Mutation Prediction/GBMLGG"
ln -sfn "$inputdir/GBM"  "$outputdir/Gene Mutation Prediction/GBMLGG/GBM"
ln -sfn "$inputdir/LGG"  "$outputdir/Gene Mutation Prediction/GBMLGG/LGG"

ln -sfn "$inputdir/HNSC"  "$outputdir/Gene Mutation Prediction/HNSC"
ln -sfn "$inputdir/KIRC"  "$outputdir/Gene Mutation Prediction/KIRC"
ln -sfn "$inputdir/LUAD"  "$outputdir/Gene Mutation Prediction/LUAD"
ln -sfn "$inputdir/UCEC"  "$outputdir/Gene Mutation Prediction/UCEC"

echo "✅ 暴力创建完成！全部纯净无括号！"