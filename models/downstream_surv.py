# 文件路径: models/downstream_surv.py
import torch
import torch.nn as nn
from transformers.models.vit.modeling_vit import ViTEncoder
from transformers import ViTConfig
from transformers.modeling_outputs import SequenceClassifierOutput # 💡 添加导入

# 生存分析专用的 NLL 损失函数
def nll_loss(hazards, survival, Y, c, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1).long()  
    c = c.view(batch_size, 1).to(hazards.dtype)
    
    if survival is None:
        survival = torch.cumprod(1 - hazards, dim=1)
        
    survival_padded = torch.cat([torch.ones_like(c), survival], 1)
    
    uncensored_loss = -c * (
        torch.log(torch.gather(survival_padded, 1, Y).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -(1-c) * torch.log(
        torch.gather(survival_padded, 1, Y + 1).clamp(min=eps)
    )
    
    batch_loss = uncensored_loss + censored_loss
    return batch_loss.mean()

# 2D 位置编码函数 (和分类任务复用相同的逻辑)
def get_2d_sincos_pos_embed(embed_dim, coords):
    b, m, _ = coords.shape
    emb_dim_half = embed_dim // 2
    x, y = coords[:, :, 0] / 256.0, coords[:, :, 1] / 256.0
    omega = torch.arange(emb_dim_half // 2, dtype=torch.float32, device=coords.device) / (emb_dim_half / 2.)
    omega = 1. / (10000 ** omega) 
    out_x = torch.einsum('bm,d->bmd', x, omega)
    emb_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=-1) 
    out_y = torch.einsum('bm,d->bmd', y, omega)
    emb_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=-1) 
    return torch.cat([emb_x, emb_y], dim=-1)

class SurvDownstreamMIL(nn.Module):
    def __init__(self, hidden_size=1536, num_bins=4, num_layers=2, num_heads=12):
        super().__init__()
        # 这里的 num_bins 对应你的离散时间区间数量
        config = ViTConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4
        )
        config._attn_implementation = "sdpa" # 开启 FlashAttention
        self.transformer_encoder = ViTEncoder(config)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.classifier = nn.Linear(hidden_size, num_bins)
        
    def forward(self, input_features, attention_mask=None, labels=None, coords=None, **kwargs):
        batch_size = input_features.shape[0]
        
        # 1. 坐标位置编码
        if coords is not None and not torch.all(coords == 0):
            pos_embed = get_2d_sincos_pos_embed(input_features.shape[-1], coords)
            input_features = input_features + pos_embed
            
        # 2. 拼接 CLS 并进入 Transformer
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
        hidden_states = torch.cat((cls_tokens, input_features), dim=1)
        
        if attention_mask is not None:
            cls_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
            extended_attention_mask = torch.cat((cls_mask, attention_mask), dim=1)
            extended_attention_mask = (1.0 - extended_attention_mask[:, None, None, :]) * -10000.0
        else:
            extended_attention_mask = None
            
        encoder_outputs = self.transformer_encoder(hidden_states)
        wsi_representation = encoder_outputs[0][:, 0, :] 
        
        # 3. 生存风险计算 (Hazards & Survival)
        logits = self.classifier(wsi_representation)
        
        # 4. 组合标签拆解与 Loss 计算
        loss = None
        if labels is not None:
            # 💡 将网络输出强制转为 float32 防止精度报错
            logits_fp32 = logits.to(torch.float32)
            
            # 使用 float32 的 logits 重新计算 hazards 和 survival，用于算 Loss
            hazards_fp32 = torch.sigmoid(logits_fp32)
            survival_fp32 = torch.cumprod(1 - hazards_fp32, dim=1)
            
            # 从外部传入的组合标签 [Batch, 3] 中拆解出 nll_loss 需要的元素
            # labels[:, 0] 是 survival_bin (也就是 Y)
            # labels[:, 1] 是 status (事件是否发生，也就是 c)
            Y = labels[:, 0]
            c = labels[:, 1]
            
            # 💡 调用真实的 nll_loss 函数
            loss = nll_loss(hazards_fp32, survival_fp32, Y, c)
            
        # 为了评测，返回模型自身的 hazards (通常评测时不在乎精度冲突)
        # 用未转换的 logits 计算返回结果，保持模型输出 dtype 一致性
        hazards = torch.sigmoid(logits)    
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=hazards # 这里把 hazards 当做 logits 返回，方便 evaluate 计算 c-index
        )