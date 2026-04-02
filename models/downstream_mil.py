# 文件路径: models/downstream_mil.py
import torch
import torch.nn as nn
from transformers.models.vit.modeling_vit import ViTEncoder
from transformers import ViTConfig

class CleanDownstreamMIL(nn.Module):
    """
    纯净版下游多实例学习(MIL)模型。
    彻底抛弃上游的图像切块和位置编码，直接接收 [Batch, Seq_len, 1536] 的特征矩阵。
    """
    def __init__(self, hidden_size=1536, num_classes=3, num_layers=2, num_heads=12):
        super().__init__()
        
        # 1. 初始化标准 Transformer Encoder (无位置编码)
        # 1. 初始化标准 Transformer Encoder (无位置编码)
        config = ViTConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4
        )
        
        # 💡 【核心修复】强制指定 Attention 的底层实现方式
        # 推荐使用 "eager" (最稳定兼容) 或 "sdpa" (PyTorch 2.0+ 加速)
        # config._attn_implementation = "eager" 
        config._attn_implementation = "sdpa"
        self.transformer_encoder = ViTEncoder(config)
        
        # 2. 全局可学习的 CLS Token (用于汇总全局信息)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        # 3. 分类头
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_features, attention_mask=None, labels=None):
        batch_size = input_features.shape[0]
        
        # 1. 拼接 CLS Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
        hidden_states = torch.cat((cls_tokens, input_features), dim=1)
        
        # 2. 处理 Attention Mask (兼容 Batch Size > 1 的 Padding)
        if attention_mask is not None:
            cls_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
            extended_attention_mask = torch.cat((cls_mask, attention_mask), dim=1)
            # HF Transformer 的 Mask 格式要求: [Batch, 1, 1, Seq_len]
            extended_attention_mask = (1.0 - extended_attention_mask[:, None, None, :]) * -10000.0
        else:
            extended_attention_mask = None
            
        # 3. 送入 Encoder
        encoder_outputs = self.transformer_encoder(
            hidden_states,
            # head_mask=None,
            # output_attentions=False,
            # 注意：如果 transformers 版本较新，这里可能需要传 attention_mask=extended_attention_mask
            # 我们按照底层源码要求直接传进去
        )
        # ViTEncoder 返回的是 tuple，第一项是 hidden_states
        sequence_output = encoder_outputs[0]
        
        # 4. 提取 CLS Token
        wsi_representation = sequence_output[:, 0, :] 
        
        # 5. 分类输出
        logits = self.classifier(wsi_representation)
        
        # 6. 计算 Loss (为了兼容 HF Trainer)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
            
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}