# NEPA: Next-Embedding Prediction Makes Strong Vision Learners

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/abs/2512.16922)
[![Project Page](https://img.shields.io/badge/Project-Website-5B7493?logo=googlechrome&logoColor=5B7493)](https://sihanxu.github.io/nepa)
[![Hugging Model Card](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/collections/SixAILab/nepa)

This is a PyTorch/GPU re-implementation of Next-Embedding Prediction Makes Strong Vision Learners.

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/63f233820a16587ea967adc2/f3ybK_7Mf7rMekc05AcWH.png" width="350">
</p>

Next-Embedding Predictive Autoregression. An image is split into patches and embedded into a sequence. An autoregressive model predicts the next embedding from previous ones.

```
@article{six2025nepa,
  title={Next-Embedding Prediction Makes Strong Vision Learners},
  author={Sihan Xu and Ziqiao Ma and Wenhao Chai and Xuweiyi Chen and Weiyang Jin and Joyce Chai and Saining Xie and Stella X. Yu},
  journal={arXiv preprint arXiv: 2512.16922},
  year={2025}
}
```

## TODO List

- [ x ] 基础特征提取：提取所有WSI的patches features并保存坐标信息
- [ x ] 离线数据生成：编写脚本预先根据坐标裁剪并保存32x32及全图的feature文件为独立数据集 (这一步需要考虑如何实现，特别的，对于被mask的位置的处理)
- [ x ] Dataset适配：实现新的Dataset类以直接读取离线预处理好的特征文件
- [ x ] Embedding层修改：将PatchEmbed替换为Linear Projection以适配特征输入
- [ x ] 位置编码修改：PosEmbed修改以适配变长输入
- [ x ] 训练流程配置：跑通基于feature的流程
- [ ] 第一轮训练：基于32x32数据进行模型预训练
- [ ] 第一轮测试：基于32x32训练得到的model进行第一次downstream测试(subtyping任务)，以检查训练是否正常
- [ ] 第二轮训练：基于全图数据进行模型预训练
- [ ] 第二轮测试：基于全图数据训练得到的model进行第二次downstream测试
- [ ] 后续调优


## Environment

The codebase has been tested with the following environment:

- Python 3.10
- PyTorch 2.8.0
- Transformers 4.56.2

### Installation

First, clone the repository:

```bash
git clone https://github.com/SihanXU/nepa
cd nepa
```

Then, create a conda environment and install dependencies:

```bash
conda env create -f environment.yml
conda activate nepa
```

Alternatively, you can install the dependencies manually:

```bash
pip install -r requirements.txt
```

## Quick Start

Here's a simple example to run inference with a pretrained NEPA model:

```python
from transformers import AutoImageProcessor
from models.vit_nepa import ViTNepaForImageClassification
from PIL import Image
import requests

url = 'https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('SixAILab/nepa-large-patch14-224-sft')
model = ViTNepaForImageClassification.from_pretrained('SixAILab/nepa-large-patch14-224-sft')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

## Setup Huggingface Token

To download pretrained models from Hugging Face Hub, you need to authenticate with your Hugging Face account:

```bash
hf auth login
```


## [Optional] Setup Wandb Token

We use Wandb to track experiments. You may want to use [Weights & Biases](https://wandb.ai/) to log and track your experiments:

```bash
pip install wandb
wandb login
```

## Prepare ImageNet-1k Dataset

We use the ImageNet-1k dataset for training and evaluation. To download the dataset via Hugging Face Datasets:

```bash
python download_dataset.py
```

This script will download and prepare the ImageNet-1k dataset. Note that this requires approximately 150GB of disk space. You may need to accept the dataset terms on [Hugging Face](https://huggingface.co/datasets/ILSVRC/imagenet-1k) before downloading.

## Evaluate Nepa for Image Classification

We provide pretrained checkpoints for NEPA models. The following table compares our reproduced results with the paper:

| Model  | SwiGLU (paper) | GeLU (reproduce) |
|--------|---------------:|-----------------:|
| Nepa-B | 83.8           | 83.75            |
| Nepa-L | 85.3           | 85.40            |

To evaluate the base model on ImageNet-1k validation set:

```bash
bash scripts/eval/nepa_b_sft_eval.sh 
```

This should give:
```
***** eval metrics *****
  eval_accuracy               =     0.8375
  eval_loss                   =     0.7169
```

To evaluate the large model:

```bash
bash scripts/eval/nepa_l_sft_eval.sh 
```

This should give:
```
***** eval metrics *****
  eval_accuracy               =      0.854
  eval_loss                   =     0.6371
```

## Fine-tune

To fine-tune a pretrained NEPA model on ImageNet-1k for image classification:

For the base model:

```bash
bash scripts/finetune/nepa_b_sft.sh 
```

For the large model:

```bash
bash scripts/finetune/nepa_l_sft.sh 
```

You can modify the training hyperparameters (learning rate, batch size, epochs, etc.) in the corresponding script files.

## Pretrain

To pretrain NEPA from scratch on ImageNet-1k:

For the base model:

```bash
bash scripts/pretrain/nepa_b.sh
```

For the large model:

```bash
bash scripts/pretrain/nepa_l.sh
```

Pretraining typically requires multiple GPUs. We recommend using at least 8 A100 GPUs for the large model.

## Convert a Pretrained Model to Classification Model

After pretraining, you can convert the pretrained model to a classification model by initializing a classification head. Use the `init_nepa_cls_from_pretrain.py` script:

Here is an example:
```
python init_nepa_cls_from_pretrain.py \
  --pretrained_model_id SixAILab/nepa-base-patch14-224 \
  --config_model_id configs/finetune/nepa-base-patch14-224-sft \
  --pretrained_revision main \
  --save_local \
  --local_dir ./nepa-base-patch14-224-sft
```

## Acknowledgements

We gratefully acknowledge the developers of [Transformers](https://github.com/huggingface/transformers), [Datasets](https://github.com/huggingface/datasets), [Evaluate](https://github.com/huggingface/evaluate), and [timm](https://github.com/huggingface/pytorch-image-models) for their excellent open-source contributions.

## Contact

Feel free to contact me through email (sihanxu@umich.edu). Enjoy!
