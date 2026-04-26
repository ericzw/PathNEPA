import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

class CleanDownstreamMIL(nn.Module):
    def __init__(self, hidden_size=1536, num_classes=2, num_layers=2, num_heads=12):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_size * 2, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ABMIL 注意力
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.loss_fct = CrossEntropyLoss()

    def forward(self, input_features, labels=None, attention_mask=None, **kwargs):
        # 1. 特征提取
        feat = self.transformer(input_features)  # [B, N, D]

        # 2. ABMIL 注意力加权
        attn = self.attention(feat)
        attn = torch.softmax(attn, dim=1)
        bag_feat = (feat * attn).sum(dim=1)

        # 3. 分类
        logits = self.classifier(bag_feat)

        # 4. 有标签就计算loss（ Trainer 必须要这个！）
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, logits
        
        return logits