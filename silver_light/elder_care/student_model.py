import torch
import torch.nn as nn

class MLPWithAttention(nn.Module):
    """
    轻量级学生模型：MLP + Self-Attention
    输入：11维结构化特征
    输出：8类健康状态软标签分布
    """
    def __init__(self, input_dim=11, hidden_dim=64, num_classes=8, num_layers=2, dropout=0.2):
        super(MLPWithAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 特征映射层
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

        # 自注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (batch_size, input_dim=11)
        x = self.mlp(x)

        # 添加序列维度，用于 Attention
        x = x.unsqueeze(1)

        # 自注意力
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)  # 残差连接
        x = x.squeeze(1)

        # 分类
        logits = self.classifier(x)
        return logits

# 使用示例
if __name__ == "__main__":
    model = MLPWithAttention(input_dim=11, hidden_dim=64, num_classes=8)
    x = torch.randn(4, 11)  # 4 个样本
    logits = model(x)
    print(f"输入: {x.shape} -> 输出: {logits.shape}")
    