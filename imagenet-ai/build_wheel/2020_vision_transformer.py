"""
Vision Transformer 教学版实现。

这个版本用最小可讲明白结构展示 ViT：
1. 图像先被切成 patch。
2. patch 变成 token embedding。
3. 加上 cls token 和位置编码后送入 encoder。
"""

from __future__ import annotations

import torch
from torch import nn

from model_blocks import PatchEmbedding, TransformerEncoderBlock


class VisionTransformerModel(nn.Module):
    """教学版 ViT。"""

    def __init__(self, num_classes: int = 3, image_size: int = 64, patch_size: int = 8, embed_dim: int = 96):
        super().__init__()
        token_count = (image_size // patch_size) ** 2

        self.patch_embedding = PatchEmbedding(3, embed_dim, patch_size=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, token_count + 1, embed_dim))
        self.encoder = nn.Sequential(
            TransformerEncoderBlock(embed_dim, num_heads=4, mlp_ratio=4),
            TransformerEncoderBlock(embed_dim, num_heads=4, mlp_ratio=4),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        batch_size = x.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.position_embedding[:, : x.shape[1], :]
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def create_model(num_classes: int = 3, image_size: int = 64):
    return VisionTransformerModel(num_classes=num_classes, image_size=image_size)
