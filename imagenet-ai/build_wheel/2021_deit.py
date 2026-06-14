"""
DeiT 教学版实现。

教学上，这里重点保留 distillation token 这件事，
让读者知道 DeiT 并不是完全换了新骨干，而是在 ViT 上加入更现实的训练思路。
"""

from __future__ import annotations

import torch
from torch import nn

from model_blocks import PatchEmbedding, TransformerEncoderBlock


class DeiTModel(nn.Module):
    """教学版 DeiT。"""

    def __init__(self, num_classes: int = 3, image_size: int = 64, patch_size: int = 8, embed_dim: int = 96):
        super().__init__()
        token_count = (image_size // patch_size) ** 2

        self.patch_embedding = PatchEmbedding(3, embed_dim, patch_size=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.distill_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, token_count + 2, embed_dim))
        self.encoder = nn.Sequential(
            TransformerEncoderBlock(embed_dim, num_heads=4, mlp_ratio=4),
            TransformerEncoderBlock(embed_dim, num_heads=4, mlp_ratio=4),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.cls_head = nn.Linear(embed_dim, num_classes)
        self.distill_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        batch_size = x.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        distill_token = self.distill_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, distill_token, x], dim=1)
        x = x + self.position_embedding[:, : x.shape[1], :]
        x = self.encoder(x)
        x = self.norm(x)

        # 这里把 cls token 和 distillation token 的预测取平均，
        # 目的是让教学版训练流程保持简单，同时保留 DeiT 的结构差异。
        cls_logits = self.cls_head(x[:, 0])
        distill_logits = self.distill_head(x[:, 1])
        return (cls_logits + distill_logits) / 2.0


def create_model(num_classes: int = 3, image_size: int = 64):
    return DeiTModel(num_classes=num_classes, image_size=image_size)
